import Foundation

/// Defines the strategy for joining two DataFrames.
public enum JoinKind: Sendable {
    /// Returns only rows with matching keys in both DataFrames.
    case inner
    /// Returns all rows from left DataFrame, and matching rows from right.
    case left
    /// Returns all rows from right DataFrame, and matching rows from left.
    case right
    /// Returns all rows when there is a match in either left or right DataFrame.
    case outer
}

extension DataFrame {
    /// Joins two DataFrames on a common key column using an efficient hash join algorithm.
    ///
    /// - Parameters:
    ///   - other: The DataFrame to join with `self`.
    ///   - key: The column name to join on (must exist in both DataFrames).
    ///   - how: The type of join (`.inner`, `.left`, `.right`, `.outer`). Default is `.inner`.
    /// - Returns: A new DataFrame containing the joined columns and matching rows.
    /// - Throws: `DataFrameError.columnNotFound` if key column is missing in either DataFrame.
    public func join(
        _ other: DataFrame,
        on key: String,
        how: JoinKind = .inner
    ) throws -> DataFrame {
        guard let leftKeyCol = self[column: key] else {
            throw DataFrameError.columnNotFound(key)
        }
        guard let rightKeyCol = other[column: key] else {
            throw DataFrameError.columnNotFound(key)
        }

        let leftCount = shape.rows
        let rightCount = other.shape.rows

        // 1. Build hash table on right DataFrame key column
        var rightHashTable = [AnyHashable: [Int]]()
        rightHashTable.reserveCapacity(rightCount)

        for r in 0..<rightCount {
            if let val = rightKeyCol.value(at: r), let hashable = val as? AnyHashable {
                rightHashTable[hashable, default: []].append(r)
            }
        }

        var leftIndices = [Int?]()
        var rightIndices = [Int?]()

        let estimateCap = max(leftCount, rightCount)
        leftIndices.reserveCapacity(estimateCap)
        rightIndices.reserveCapacity(estimateCap)

        var matchedRightIndices = Set<Int>()
        if how == .right || how == .outer {
            matchedRightIndices.reserveCapacity(rightCount)
        }

        // 2. Probe loop using left DataFrame
        for l in 0..<leftCount {
            if let val = leftKeyCol.value(at: l),
               let hashable = val as? AnyHashable,
               let matches = rightHashTable[hashable] {
                for r in matches {
                    leftIndices.append(l)
                    rightIndices.append(r)
                    if how == .right || how == .outer {
                        matchedRightIndices.insert(r)
                    }
                }
            } else {
                if how == .left || how == .outer {
                    leftIndices.append(l)
                    rightIndices.append(nil)
                }
            }
        }

        // 3. Add unmatched right rows for right and outer joins
        if how == .right || how == .outer {
            for r in 0..<rightCount {
                if !matchedRightIndices.contains(r) {
                    leftIndices.append(nil)
                    rightIndices.append(r)
                }
            }
        }

        let totalResultRows = leftIndices.count
        guard totalResultRows > 0 else { return DataFrame.empty }

        // 4. Assemble result columns
        var resultColumns = [any AnyColumn]()

        // Build key column first
        var joinedKeyValues = [Any?]()
        joinedKeyValues.reserveCapacity(totalResultRows)
        for i in 0..<totalResultRows {
            if let lIdx = leftIndices[i] {
                joinedKeyValues.append(leftKeyCol.value(at: lIdx))
            } else if let rIdx = rightIndices[i] {
                joinedKeyValues.append(rightKeyCol.value(at: rIdx))
            } else {
                joinedKeyValues.append(nil)
            }
        }
        resultColumns.append(makeColumn(name: key, dtype: leftKeyCol.dtype, rawValues: joinedKeyValues))

        // Gather remaining left columns
        for leftCol in self.columns where leftCol.name != key {
            let colName = other._columns[leftCol.name] != nil ? "\(leftCol.name)_x" : leftCol.name
            let rawVals: [Any?] = leftIndices.map { idx in
                guard let idx else { return nil }
                return leftCol.value(at: idx)
            }
            resultColumns.append(makeColumn(name: colName, dtype: leftCol.dtype, rawValues: rawVals))
        }

        // Gather remaining right columns
        for rightCol in other.columns where rightCol.name != key {
            let colName = self._columns[rightCol.name] != nil ? "\(rightCol.name)_y" : rightCol.name
            let rawVals: [Any?] = rightIndices.map { idx in
                guard let idx else { return nil }
                return rightCol.value(at: idx)
            }
            resultColumns.append(makeColumn(name: colName, dtype: rightCol.dtype, rawValues: rawVals))
        }

        return try DataFrame(columns: resultColumns)
    }
}
