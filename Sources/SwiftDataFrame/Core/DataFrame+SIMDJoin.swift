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

        if how == .inner {
            let (leftIndices, rightIndices, leftIsIdentity, rightIsIdentity) = try computeInnerJoinIndices(
                leftKeyCol: leftKeyCol,
                rightKeyCol: rightKeyCol,
                leftCount: leftCount,
                rightCount: rightCount
            )

            var resultColumns: [any AnyColumn] = []
            resultColumns.reserveCapacity(self.columns.count + other.columns.count - 1)

            // 1. Joined key column (inner join always uses left key values; preserve compact Double storage if either side is compact)
            if let leftD = leftKeyCol as? TypedColumn<Double>,
               let rightD = rightKeyCol as? TypedColumn<Double>,
               leftD.compactNumericStorage == nil,
               rightD.compactNumericStorage != nil {
                let output = CompactNumericColumn(count: leftIndices.count) { position in
                    leftD[leftIndices[position]]
                }
                resultColumns.append(TypedColumn<Double>(name: key, compactColumn: output))
            } else if leftIsIdentity {
                resultColumns.append(leftKeyCol)
            } else {
                let gatheredKey = leftKeyCol.gathered(at: leftIndices)
                resultColumns.append(gatheredKey.name == key ? gatheredKey : gatheredKey.renamed(to: key))
            }

            // 2. Gather left columns
            for leftCol in self.columns where leftCol.name != key {
                let colName = other._columns[leftCol.name] != nil ? "\(leftCol.name)_x" : leftCol.name
                if leftIsIdentity {
                    resultColumns.append(colName == leftCol.name ? leftCol : leftCol.renamed(to: colName))
                } else {
                    let gathered = leftCol.gathered(at: leftIndices)
                    resultColumns.append(colName == leftCol.name ? gathered : gathered.renamed(to: colName))
                }
            }

            // 3. Gather right columns
            for rightCol in other.columns where rightCol.name != key {
                let colName = self._columns[rightCol.name] != nil ? "\(rightCol.name)_y" : rightCol.name
                if rightIsIdentity {
                    resultColumns.append(colName == rightCol.name ? rightCol : rightCol.renamed(to: colName))
                } else {
                    let gathered = rightCol.gathered(at: rightIndices)
                    resultColumns.append(colName == rightCol.name ? gathered : gathered.renamed(to: colName))
                }
            }

            return try DataFrame(columns: resultColumns)
        }

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

    @inline(__always)
    private static func mixInt64(_ k: Int64) -> Int {
        var x = UInt64(bitPattern: k)
        x = (x ^ (x >> 30)) &* 0xbf58476d1ce4e5b9
        x = (x ^ (x >> 27)) &* 0x94d049bb133111eb
        x = x ^ (x >> 31)
        return Int(truncatingIfNeeded: x)
    }

    private func computeInnerJoinIndices(
        leftKeyCol: any AnyColumn,
        rightKeyCol: any AnyColumn,
        leftCount: Int,
        rightCount: Int
    ) throws -> (leftIndices: [Int], rightIndices: [Int], leftIsIdentity: Bool, rightIsIdentity: Bool) {
        if leftCount == 0 || rightCount == 0 {
            return ([], [], false, false)
        }
        switch leftKeyCol.dtype {
        case .int64:
            if let lCol = leftKeyCol as? TypedColumn<Int64>,
               let rCol = rightKeyCol as? TypedColumn<Int64> {
                if case .dense(let lv, validity: nil) = lCol.storage,
                   case .dense(let rv, validity: nil) = rCol.storage {
                    return buildAndProbeInnerInt64(leftValues: lv, rightValues: rv)
                }
                return buildAndProbeInnerHashTable(
                    leftValues: lCol.values,
                    rightValues: rCol.values,
                    leftCount: leftCount,
                    rightCount: rightCount
                )
            }
        case .int32:
            if let lCol = leftKeyCol as? TypedColumn<Int32>,
               let rCol = rightKeyCol as? TypedColumn<Int32> {
                return buildAndProbeInnerHashTable(
                    leftValues: lCol.values,
                    rightValues: rCol.values,
                    leftCount: leftCount,
                    rightCount: rightCount
                )
            }
        case .float64:
            if let lCol = leftKeyCol as? TypedColumn<Double>,
               let rCol = rightKeyCol as? TypedColumn<Double> {
                switch (lCol.compactNumericStorage, rCol.compactNumericStorage) {
                case let (left?, right?):
                    return buildAndProbeInnerHashTable(
                        leftValues: CompactColumnValues(column: left),
                        rightValues: CompactColumnValues(column: right),
                        leftCount: leftCount,
                        rightCount: rightCount
                    )
                case let (left?, nil):
                    return buildAndProbeInnerHashTable(
                        leftValues: CompactColumnValues(column: left),
                        rightValues: rCol.values,
                        leftCount: leftCount,
                        rightCount: rightCount
                    )
                case let (nil, right?):
                    return buildAndProbeInnerHashTable(
                        leftValues: lCol.values,
                        rightValues: CompactColumnValues(column: right),
                        leftCount: leftCount,
                        rightCount: rightCount
                    )
                case (nil, nil):
                    return buildAndProbeInnerHashTable(
                        leftValues: lCol.values,
                        rightValues: rCol.values,
                        leftCount: leftCount,
                        rightCount: rightCount
                    )
                }
            }
        case .float32:
            if let lCol = leftKeyCol as? TypedColumn<Float>,
               let rCol = rightKeyCol as? TypedColumn<Float> {
                return buildAndProbeInnerHashTable(
                    leftValues: lCol.values,
                    rightValues: rCol.values,
                    leftCount: leftCount,
                    rightCount: rightCount
                )
            }
        case .utf8:
            if let lCol = leftKeyCol as? TypedColumn<String>,
               let rCol = rightKeyCol as? TypedColumn<String> {
                return buildAndProbeInnerHashTable(
                    leftValues: lCol.values,
                    rightValues: rCol.values,
                    leftCount: leftCount,
                    rightCount: rightCount
                )
            }
        case .boolean:
            if let lCol = leftKeyCol as? TypedColumn<Bool>,
               let rCol = rightKeyCol as? TypedColumn<Bool> {
                return buildAndProbeInnerHashTable(
                    leftValues: lCol.values,
                    rightValues: rCol.values,
                    leftCount: leftCount,
                    rightCount: rightCount
                )
            }
        case .date32:
            if let lCol = leftKeyCol as? TypedColumn<Date>,
               let rCol = rightKeyCol as? TypedColumn<Date> {
                return buildAndProbeInnerHashTable(
                    leftValues: lCol.values,
                    rightValues: rCol.values,
                    leftCount: leftCount,
                    rightCount: rightCount
                )
            }
        }

        return genericInnerHashJoinIndices(
            leftKeyCol: leftKeyCol,
            rightKeyCol: rightKeyCol,
            leftCount: leftCount,
            rightCount: rightCount
        )
    }

    private func buildAndProbeInnerInt64(
        leftValues: [Int64],
        rightValues: [Int64]
    ) -> (leftIndices: [Int], rightIndices: [Int], leftIsIdentity: Bool, rightIsIdentity: Bool) {
        let leftCount = leftValues.count
        let rightCount = rightValues.count
        var bucketCount = 16
        while bucketCount < max(rightCount * 2, 16) {
            bucketCount <<= 1
        }
        let mask = bucketCount - 1

        var head = [Int32](repeating: -1, count: bucketCount)
        var tail = [Int32](repeating: -1, count: bucketCount)
        var next = [Int32](repeating: -1, count: rightCount)

        leftValues.withUnsafeBufferPointer { lBuf in
            rightValues.withUnsafeBufferPointer { rBuf in
                head.withUnsafeMutableBufferPointer { hBuf in
                    tail.withUnsafeMutableBufferPointer { tBuf in
                        next.withUnsafeMutableBufferPointer { nBuf in
                            guard let rPtr = rBuf.baseAddress,
                                  let hPtr = hBuf.baseAddress,
                                  let tPtr = tBuf.baseAddress,
                                  let nPtr = nBuf.baseAddress else { return }
                            for r in 0..<rightCount {
                                let bucket = Self.mixInt64(rPtr[r]) & mask
                                let prevTail = tPtr[bucket]
                                let ri = Int32(r)
                                if prevTail == -1 {
                                    hPtr[bucket] = ri
                                } else {
                                    nPtr[Int(prevTail)] = ri
                                }
                                tPtr[bucket] = ri
                            }
                        }
                    }
                }
            }
        }

        var leftIndices = [Int]()
        var rightIndices = [Int]()
        let capEstimate = max(leftCount, rightCount)
        leftIndices.reserveCapacity(capEstimate)
        rightIndices.reserveCapacity(capEstimate)

        var leftIsIdentity = true
        var expectedLeft = 0
        var rightIsIdentity = true
        var expectedRight = 0

        leftValues.withUnsafeBufferPointer { lBuf in
            rightValues.withUnsafeBufferPointer { rBuf in
                head.withUnsafeBufferPointer { hBuf in
                    next.withUnsafeBufferPointer { nBuf in
                        guard let lPtr = lBuf.baseAddress,
                              let rPtr = rBuf.baseAddress,
                              let hPtr = hBuf.baseAddress,
                              let nPtr = nBuf.baseAddress else { return }
                        for l in 0..<leftCount {
                            let val = lPtr[l]
                            let bucket = Self.mixInt64(val) & mask
                            var curr = hPtr[bucket]
                            while curr != -1 {
                                let r = Int(curr)
                                if rPtr[r] == val {
                                    leftIndices.append(l)
                                    rightIndices.append(r)
                                    if l != expectedLeft { leftIsIdentity = false }
                                    expectedLeft += 1
                                    if r != expectedRight { rightIsIdentity = false }
                                    expectedRight += 1
                                }
                                curr = nPtr[r]
                            }
                        }
                    }
                }
            }
        }

        return (
            leftIndices,
            rightIndices,
            leftIsIdentity && expectedLeft == leftCount,
            rightIsIdentity && expectedRight == rightCount
        )
    }

    private func buildAndProbeInnerHashTable<Left: RandomAccessCollection, Right: RandomAccessCollection, T: Hashable>(
        leftValues: Left,
        rightValues: Right,
        leftCount: Int,
        rightCount: Int
    ) -> (leftIndices: [Int], rightIndices: [Int], leftIsIdentity: Bool, rightIsIdentity: Bool)
    where Left.Index == Int, Right.Index == Int, Left.Element == T?, Right.Element == T? {
        var bucketCount = 16
        while bucketCount < max(rightCount * 2, 16) {
            bucketCount <<= 1
        }
        let mask = bucketCount - 1

        var head = [Int32](repeating: -1, count: bucketCount)
        var tail = [Int32](repeating: -1, count: bucketCount)
        var next = [Int32](repeating: -1, count: rightCount)

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

        var leftIndices = [Int]()
        var rightIndices = [Int]()
        let capEstimate = max(leftCount, rightCount)
        leftIndices.reserveCapacity(capEstimate)
        rightIndices.reserveCapacity(capEstimate)

        var leftIsIdentity = true
        var expectedLeft = 0
        var rightIsIdentity = true
        var expectedRight = 0

        for l in 0..<leftCount {
            if let val = leftValues[l] {
                let bucket = (val.hashValue & 0x7FFFFFFF) & mask
                var curr = head[bucket]
                while curr != -1 {
                    let r = Int(curr)
                    if rightValues[r] == val {
                        leftIndices.append(l)
                        rightIndices.append(r)
                        if l != expectedLeft { leftIsIdentity = false }
                        expectedLeft += 1
                        if r != expectedRight { rightIsIdentity = false }
                        expectedRight += 1
                    }
                    curr = next[r]
                }
            }
        }

        return (
            leftIndices,
            rightIndices,
            leftIsIdentity && expectedLeft == leftCount,
            rightIsIdentity && expectedRight == rightCount
        )
    }

    private func genericInnerHashJoinIndices(
        leftKeyCol: any AnyColumn,
        rightKeyCol: any AnyColumn,
        leftCount: Int,
        rightCount: Int
    ) -> (leftIndices: [Int], rightIndices: [Int], leftIsIdentity: Bool, rightIsIdentity: Bool) {
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

        var leftIndices = [Int]()
        var rightIndices = [Int]()
        let capEstimate = max(leftCount, rightCount)
        leftIndices.reserveCapacity(capEstimate)
        rightIndices.reserveCapacity(capEstimate)

        var leftIsIdentity = true
        var expectedLeft = 0
        var rightIsIdentity = true
        var expectedRight = 0

        for l in 0..<leftCount {
            if let val = leftKeyCol.value(at: l), let hashable = val as? AnyHashable {
                let bucket = (hashable.hashValue & 0x7FFFFFFF) & mask
                var curr = head[bucket]
                while curr != -1 {
                    let r = Int(curr)
                    if rightHashables[r] == hashable {
                        leftIndices.append(l)
                        rightIndices.append(r)
                        if l != expectedLeft { leftIsIdentity = false }
                        expectedLeft += 1
                        if r != expectedRight { rightIsIdentity = false }
                        expectedRight += 1
                    }
                    curr = next[r]
                }
            }
        }

        return (
            leftIndices,
            rightIndices,
            leftIsIdentity && expectedLeft == leftCount,
            rightIsIdentity && expectedRight == rightCount
        )
    }

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
                    leftValues: lCol.values,
                    rightValues: rCol.values,
                    leftCount: leftCount,
                    rightCount: rightCount,
                    how: how
                )
            }
        case .int32:
            if let lCol = leftKeyCol as? TypedColumn<Int32>,
               let rCol = rightKeyCol as? TypedColumn<Int32> {
                return buildAndProbeHashTable(
                    leftValues: lCol.values,
                    rightValues: rCol.values,
                    leftCount: leftCount,
                    rightCount: rightCount,
                    how: how
                )
            }
        case .float64:
            if let lCol = leftKeyCol as? TypedColumn<Double>,
               let rCol = rightKeyCol as? TypedColumn<Double> {
                switch (lCol.compactNumericStorage, rCol.compactNumericStorage) {
                case let (left?, right?):
                    return buildAndProbeHashTable(leftValues: CompactColumnValues(column: left),
                        rightValues: CompactColumnValues(column: right), leftCount: leftCount,
                        rightCount: rightCount, how: how)
                case let (left?, nil):
                    return buildAndProbeHashTable(leftValues: CompactColumnValues(column: left),
                        rightValues: rCol.values, leftCount: leftCount, rightCount: rightCount, how: how)
                case let (nil, right?):
                    return buildAndProbeHashTable(leftValues: lCol.values,
                        rightValues: CompactColumnValues(column: right), leftCount: leftCount,
                        rightCount: rightCount, how: how)
                case (nil, nil):
                    return buildAndProbeHashTable(leftValues: lCol.values,
                        rightValues: rCol.values, leftCount: leftCount, rightCount: rightCount, how: how)
                }
            }
        case .float32:
            if let lCol = leftKeyCol as? TypedColumn<Float>,
               let rCol = rightKeyCol as? TypedColumn<Float> {
                return buildAndProbeHashTable(
                    leftValues: lCol.values,
                    rightValues: rCol.values,
                    leftCount: leftCount,
                    rightCount: rightCount,
                    how: how
                )
            }
        case .utf8:
            if let lCol = leftKeyCol as? TypedColumn<String>,
               let rCol = rightKeyCol as? TypedColumn<String> {
                return buildAndProbeHashTable(
                    leftValues: lCol.values,
                    rightValues: rCol.values,
                    leftCount: leftCount,
                    rightCount: rightCount,
                    how: how
                )
            }
        case .boolean:
            if let lCol = leftKeyCol as? TypedColumn<Bool>,
               let rCol = rightKeyCol as? TypedColumn<Bool> {
                return buildAndProbeHashTable(
                    leftValues: lCol.values,
                    rightValues: rCol.values,
                    leftCount: leftCount,
                    rightCount: rightCount,
                    how: how
                )
            }
        case .date32:
            if let lCol = leftKeyCol as? TypedColumn<Date>,
               let rCol = rightKeyCol as? TypedColumn<Date> {
                return buildAndProbeHashTable(
                    leftValues: lCol.values,
                    rightValues: rCol.values,
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

    private func buildAndProbeHashTable<Left: RandomAccessCollection, Right: RandomAccessCollection, T: Hashable>(
        leftValues: Left,
        rightValues: Right,
        leftCount: Int,
        rightCount: Int,
        how: JoinKind
    ) -> (leftIndices: [Int?], rightIndices: [Int?])
    where Left.Index == Int, Right.Index == Int, Left.Element == T?, Right.Element == T? {
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
