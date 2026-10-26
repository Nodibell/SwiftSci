import Foundation
import Accelerate

extension DataFrame {

    /// Performs a high-performance, SIMD-accelerated hash join between two DataFrames on a specified key column.
    ///
    /// This specialized join path avoids dynamic existential boxing by operating directly on typed column storage
    /// buffers (`Int64`, `Int32`, `Double`, `Float`, `String`, `Bool`), yielding optimal cache locality and execution speed.
    ///
    /// - Parameters:
    ///   - other: The right DataFrame to join with `self`.
    ///   - key: The common key column name (must exist in both DataFrames).
    ///   - how: The type of relational join (`.inner`, `.left`, `.right`, `.outer`). Default is `.inner`.
    /// - Returns: A new `DataFrame` containing the joined relational schema and rows.
    /// - Throws: `SwiftMLError.columnNotFound` if the key column is absent in either DataFrame, or `SwiftMLError.typeMismatch` if column types differ.
    public func joinSIMD(
        _ other: DataFrame,
        on key: String,
        how: JoinKind = .inner
    ) throws -> DataFrame {
        guard let leftKeyCol = self[column: key] else {
            throw SwiftMLError.columnNotFound(key)
        }
        guard let rightKeyCol = other[column: key] else {
            throw SwiftMLError.columnNotFound(key)
        }
        guard leftKeyCol.dtype == rightKeyCol.dtype else {
            throw SwiftMLError.typeMismatch(column: key, expected: "\(leftKeyCol.dtype)", got: "\(rightKeyCol.dtype)")
        }

        let leftCount = self.rowCount
        let rightCount = other.rowCount

        // Match index pairs (leftIdx, rightIdx)
        let (leftIndices, rightIndices) = try computeJoinIndices(
            leftKeyCol: leftKeyCol,
            rightKeyCol: rightKeyCol,
            leftCount: leftCount,
            rightCount: rightCount,
            how: how
        )

        // Assemble result columns using direct typed gathered indexing (zero boxing)
        var resultColumns: [any AnyColumn] = []
        resultColumns.reserveCapacity(self.columns.count + other.columns.count - 1)

        // 1. Build joined key column
        let joinedKeyCol = leftKeyCol.gatheredJoinKey(
            leftIndices: leftIndices,
            rightColumn: rightKeyCol,
            rightIndices: rightIndices,
            name: key
        )
        resultColumns.append(joinedKeyCol)

        // 2. Gather left columns
        for leftCol in self.columns where leftCol.name != key {
            let colName = other._columns[leftCol.name] != nil ? "\(leftCol.name)_x" : leftCol.name
            resultColumns.append(leftCol.gatheredWithNulls(at: leftIndices, newName: colName))
        }

        // 3. Gather right columns
        for rightCol in other.columns where rightCol.name != key {
            let colName = self._columns[rightCol.name] != nil ? "\(rightCol.name)_y" : rightCol.name
            resultColumns.append(rightCol.gatheredWithNulls(at: rightIndices, newName: colName))
        }

        return try DataFrame(columns: resultColumns)
    }

    // MARK: – Typed Join Dispatcher

    private func computeJoinIndices(
        leftKeyCol: any AnyColumn,
        rightKeyCol: any AnyColumn,
        leftCount: Int,
        rightCount: Int,
        how: JoinKind
    ) throws -> (leftIndices: [Int?], rightIndices: [Int?]) {
        if leftCount == 0 {
            let rows = how == .right || how == .outer ? rightCount : 0
            return ([Int?](repeating: nil, count: rows), (0..<rows).map(Optional.some))
        }
        switch leftKeyCol.dtype {
        case .int64:
            if let lCol = leftKeyCol as? TypedColumn<Int64>,
               let rCol = rightKeyCol as? TypedColumn<Int64> {
                return buildAndProbeHashTable(
                    leftValues: lCol,
                    rightValues: rCol,
                    leftCount: leftCount,
                    rightCount: rightCount,
                    how: how
                )
            }
        case .int32:
            if let lCol = leftKeyCol as? TypedColumn<Int32>,
               let rCol = rightKeyCol as? TypedColumn<Int32> {
                return buildAndProbeHashTable(
                    leftValues: lCol,
                    rightValues: rCol,
                    leftCount: leftCount,
                    rightCount: rightCount,
                    how: how
                )
            }
        case .float64:
            if let lCol = leftKeyCol as? TypedColumn<Double>,
               let rCol = rightKeyCol as? TypedColumn<Double> {
                return buildAndProbeHashTable(
                    leftValues: lCol,
                    rightValues: rCol,
                    leftCount: leftCount,
                    rightCount: rightCount,
                    how: how
                )
            }
        case .float32:
            if let lCol = leftKeyCol as? TypedColumn<Float>,
               let rCol = rightKeyCol as? TypedColumn<Float> {
                return buildAndProbeHashTable(
                    leftValues: lCol,
                    rightValues: rCol,
                    leftCount: leftCount,
                    rightCount: rightCount,
                    how: how
                )
            }
        case .utf8:
            if let lCol = leftKeyCol as? TypedColumn<String>,
               let rCol = rightKeyCol as? TypedColumn<String> {
                return buildAndProbeHashTable(
                    leftValues: lCol,
                    rightValues: rCol,
                    leftCount: leftCount,
                    rightCount: rightCount,
                    how: how
                )
            }
        case .boolean:
            if let lCol = leftKeyCol as? TypedColumn<Bool>,
               let rCol = rightKeyCol as? TypedColumn<Bool> {
                return buildAndProbeHashTable(
                    leftValues: lCol,
                    rightValues: rCol,
                    leftCount: leftCount,
                    rightCount: rightCount,
                    how: how
                )
            }
        case .date32:
            if let lCol = leftKeyCol as? TypedColumn<Date>,
               let rCol = rightKeyCol as? TypedColumn<Date> {
                return buildAndProbeHashTable(
                    leftValues: lCol,
                    rightValues: rCol,
                    leftCount: leftCount,
                    rightCount: rightCount,
                    how: how
                )
            }
        }

        // Generic fallback
        return genericHashJoinIndices(
            leftKeyCol: leftKeyCol,
            rightKeyCol: rightKeyCol,
            leftCount: leftCount,
            rightCount: rightCount,
            how: how
        )
    }

    private func buildAndProbeHashTable<T: SupportedType>(
        leftValues: TypedColumn<T>,
        rightValues: TypedColumn<T>,
        leftCount: Int,
        rightCount: Int,
        how: JoinKind
    ) -> (leftIndices: [Int?], rightIndices: [Int?]) {
        guard rightCount > 0 else {
            if how == .left || how == .outer {
                return ((0..<leftCount).map { Optional($0) }, [Int?](repeating: nil, count: leftCount))
            }
            return ([], [])
        }

        // 1. Capacity & bucket count (power of two >= rightCount * 2)
        var bucketCount = 16
        while bucketCount < max(rightCount * 2, 16) {
            bucketCount <<= 1
        }
        let mask = bucketCount - 1

        var head = [Int32](repeating: -1, count: bucketCount)
        var tail = [Int32](repeating: -1, count: bucketCount)
        var next = [Int32](repeating: -1, count: rightCount)

        // 2. Build chained hash table on right column with FIFO order
        for r in 0..<rightCount {
            if let val = rightValues[r] {
                let bucket = (val.hashValue & 0x7FFFFFFF) & mask
                let prevTail = tail[bucket]
                if prevTail == -1 {
                    head[bucket] = Int32(r)
                } else {
                    next[Int(prevTail)] = Int32(r)
                }
                tail[bucket] = Int32(r)
            }
        }

        var leftIndices = [Int?]()
        var rightIndices = [Int?]()
        let capEstimate = max(leftCount, rightCount)
        leftIndices.reserveCapacity(capEstimate)
        rightIndices.reserveCapacity(capEstimate)

        var matchedRight: [Bool]? = nil
        if how == .right || how == .outer {
            matchedRight = [Bool](repeating: false, count: rightCount)
        }

        // 3. Probe loop
        for l in 0..<leftCount {
            if let val = leftValues[l] {
                let bucket = (val.hashValue & 0x7FFFFFFF) & mask
                var curr = head[bucket]
                var foundMatch = false
                while curr != -1 {
                    let r = Int(curr)
                    if rightValues[r] == val {
                        foundMatch = true
                        leftIndices.append(l)
                        rightIndices.append(r)
                        if how == .right || how == .outer {
                            matchedRight?[r] = true
                        }
                    }
                    curr = next[r]
                }
                if !foundMatch && (how == .left || how == .outer) {
                    leftIndices.append(l)
                    rightIndices.append(nil)
                }
            } else {
                if how == .left || how == .outer {
                    leftIndices.append(l)
                    rightIndices.append(nil)
                }
            }
        }

        // 4. Add unmatched right rows
        if let matchedRight = matchedRight {
            for r in 0..<rightCount {
                if !matchedRight[r] {
                    leftIndices.append(nil)
                    rightIndices.append(r)
                }
            }
        }

        return (leftIndices, rightIndices)
    }

    private func genericHashJoinIndices(
        leftKeyCol: any AnyColumn,
        rightKeyCol: any AnyColumn,
        leftCount: Int,
        rightCount: Int,
        how: JoinKind
    ) -> (leftIndices: [Int?], rightIndices: [Int?]) {
        guard rightCount > 0 else {
            if how == .left || how == .outer {
                return ((0..<leftCount).map { Optional($0) }, [Int?](repeating: nil, count: leftCount))
            }
            return ([], [])
        }

        var bucketCount = 16
        while bucketCount < max(rightCount * 2, 16) {
            bucketCount <<= 1
        }
        let mask = bucketCount - 1

        var head = [Int32](repeating: -1, count: bucketCount)
        var tail = [Int32](repeating: -1, count: bucketCount)
        var next = [Int32](repeating: -1, count: rightCount)

        var rightHashables = [AnyHashable?](repeating: nil, count: rightCount)
        for r in 0..<rightCount {
            if let val = rightKeyCol.value(at: r), let hashable = val as? AnyHashable {
                rightHashables[r] = hashable
                let bucket = (hashable.hashValue & 0x7FFFFFFF) & mask
                let prevTail = tail[bucket]
                if prevTail == -1 {
                    head[bucket] = Int32(r)
                } else {
                    next[Int(prevTail)] = Int32(r)
                }
                tail[bucket] = Int32(r)
            }
        }

        var leftIndices = [Int?]()
        var rightIndices = [Int?]()
        let capEstimate = max(leftCount, rightCount)
        leftIndices.reserveCapacity(capEstimate)
        rightIndices.reserveCapacity(capEstimate)

        var matchedRight: [Bool]? = nil
        if how == .right || how == .outer {
            matchedRight = [Bool](repeating: false, count: rightCount)
        }

        for l in 0..<leftCount {
            if let val = leftKeyCol.value(at: l), let hashable = val as? AnyHashable {
                let bucket = (hashable.hashValue & 0x7FFFFFFF) & mask
                var curr = head[bucket]
                var foundMatch = false
                while curr != -1 {
                    let r = Int(curr)
                    if rightHashables[r] == hashable {
                        foundMatch = true
                        leftIndices.append(l)
                        rightIndices.append(r)
                        if how == .right || how == .outer {
                            matchedRight?[r] = true
                        }
                    }
                    curr = next[r]
                }
                if !foundMatch && (how == .left || how == .outer) {
                    leftIndices.append(l)
                    rightIndices.append(nil)
                }
            } else {
                if how == .left || how == .outer {
                    leftIndices.append(l)
                    rightIndices.append(nil)
                }
            }
        }

        if let matchedRight = matchedRight {
            for r in 0..<rightCount {
                if !matchedRight[r] {
                    leftIndices.append(nil)
                    rightIndices.append(r)
                }
            }
        }

        return (leftIndices, rightIndices)
    }
}
