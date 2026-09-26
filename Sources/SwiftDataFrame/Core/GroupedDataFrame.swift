import Foundation

/// Conditions used with `DataFrame.filter(column:where:)`.
public enum FilterCondition: Sendable {
    case equals(any Sendable)
    case notEquals(any Sendable)
    case greaterThan(any Sendable)
    case lessThan(any Sendable)
    case greaterThanOrEqual(any Sendable)
    case lessThanOrEqual(any Sendable)
    case isNull
    case isNotNull
    case contains(String)   // .utf8 columns only

    /// Evaluates the condition against a raw optional value.
    internal func evaluate(value: Any?) -> Bool {
        switch self {
        case .isNull:    return value == nil
        case .isNotNull: return value != nil
        default: break
        }
        guard let value else { return false }

        switch self {
        case .equals(let rhs):              return compare(value, rhs) == .orderedSame
        case .notEquals(let rhs):           return compare(value, rhs) != .orderedSame
        case .greaterThan(let rhs):         return compare(value, rhs) == .orderedDescending
        case .lessThan(let rhs):            return compare(value, rhs) == .orderedAscending
        case .greaterThanOrEqual(let rhs):  return compare(value, rhs) != .orderedAscending
        case .lessThanOrEqual(let rhs):     return compare(value, rhs) != .orderedDescending
        case .contains(let substring):
            guard let str = value as? String else { return false }
            return str.contains(substring)
        case .isNull, .isNotNull: return false // handled above
        }
    }

    private func compare(_ lhs: Any, _ rhs: Any) -> ComparisonResult {
        // Numeric comparisons promoted to Double
        if let l = toDouble(lhs), let r = toDouble(rhs) {
            return l < r ? .orderedAscending : l > r ? .orderedDescending : .orderedSame
        }
        // String comparison
        if let l = lhs as? String, let r = rhs as? String {
            return l.compare(r)
        }
        // Date comparison
        if let l = lhs as? Date, let r = rhs as? Date {
            return l < r ? .orderedAscending : l > r ? .orderedDescending : .orderedSame
        }
        // Bool comparison
        if let l = lhs as? Bool, let r = rhs as? Bool {
            return l == r ? .orderedSame : l ? .orderedDescending : .orderedAscending
        }
        return .orderedSame
    }

    private func toDouble(_ v: Any) -> Double? {
        switch v {
        case let x as Double: return x
        case let x as Float:  return Double(x)
        case let x as Int64:  return Double(x)
        case let x as Int32:  return Double(x)
        case let x as Int:    return Double(x)
        default: return nil
        }
    }
}

/// Aggregation functions available in `GroupedDataFrame`.
public enum Aggregation: Sendable {
    case sum, mean, min, max, count, first, last
}

/// The result of `DataFrame.groupBy(...)`.
public struct GroupedDataFrame: Sendable {

    internal let dataFrame: DataFrame
    internal let groupColumns: [String]

    init(dataFrame: DataFrame, groupColumns: [String]) {
        self.dataFrame    = dataFrame
        self.groupColumns = groupColumns
    }

    // MARK: – Public API

    /// Returns the count of rows per group.
    /// - Returns: A new `DataFrame` containing the transformed columns and computed results.
    public func count() -> DataFrame {
        aggregate(using: .count)
    }

    /// Returns the mean of each numeric column per group.
    /// - Returns: A new `DataFrame` containing the transformed columns and computed results.
    public func mean() -> DataFrame {
        aggregate(using: .mean)
    }

    /// Returns the sum of each numeric column per group.
    /// - Returns: A new `DataFrame` containing the transformed columns and computed results.
    public func sum() -> DataFrame {
        aggregate(using: .sum)
    }

    /// Returns the min of each numeric column per group.
    /// - Returns: A new `DataFrame` containing the transformed columns and computed results.
    public func min() -> DataFrame {
        aggregate(using: .min)
    }

    /// Returns the max of each numeric column per group.
    /// - Returns: A new `DataFrame` containing the transformed columns and computed results.
    public func max() -> DataFrame {
        aggregate(using: .max)
    }

    /// Apply multiple aggregations per column.
    /// - Parameters:
    ///   - aggregations: Dictionary mapping column names to aggregation functions.
    /// - Returns: A new `DataFrame` containing the transformed columns and computed results.
    public func agg(_ aggregations: [String: Aggregation]) -> DataFrame {
        let groups = buildGroups()
        var resultColumns: [any AnyColumn] = []

        // Group key columns
        for keyCol in groupColumns {
            guard let col = dataFrame[column: keyCol] else { continue }
            let keys = groups.representatives.map { col.value(at: $0) }
            let strKeys = keys.map { $0.map { "\($0)" } }
            resultColumns.append(TypedColumn<String>(name: keyCol, values: strKeys))
        }

        // Aggregated columns
        for (colName, agg) in aggregations {
            guard let col = dataFrame[column: colName], col.dtype.isNumeric else { continue }
            let aggValues = aggregateNumeric(col: col, groups: groups, agg: agg)
            resultColumns.append(TypedColumn<Double>(name: "\(colName)_\(aggLabel(agg))",
                                                      values: aggValues))
        }

        return (try? DataFrame(columns: resultColumns)) ?? DataFrame.empty
    }

    /// Applies aggregations per group and expands the aggregated values back to match the original DataFrame row count.
    /// - Parameters:
    ///   - aggregations: Dictionary mapping column names to aggregation functions.
    /// - Returns: A new `DataFrame` containing the transformed columns and computed results.
    public func transform(_ aggregations: [String: Aggregation]) -> DataFrame {
        let groups = buildGroups()
        var df = dataFrame

        for (colName, agg) in aggregations {
            guard let col = dataFrame[column: colName], col.dtype.isNumeric else { continue }
            let aggValues = aggregateNumeric(col: col, groups: groups, agg: agg)

            var expanded = [Double?](repeating: nil, count: dataFrame.shape.rows)
            for (row, group) in groups.rowGroups.enumerated() {
                expanded[row] = aggValues[group]
            }
            let newName = "\(colName)_group_\(aggLabel(agg))"
            let newCol = TypedColumn<Double>(name: newName, values: expanded)
            df = (try? df.withColumn(newName, column: newCol)) ?? df
        }
        return df
    }

    // MARK: – Private helpers

    private struct GroupIndex {
        var rowGroups: [Int] = []
        var representatives: [Int] = []
        var counts: [Int] = []
        var count: Int { representatives.count }

        init(rowCapacity: Int) {
            rowGroups.reserveCapacity(rowCapacity)
        }

        mutating func append(_ group: Int) {
            if group == count {
                representatives.append(rowGroups.count)
                counts.append(0)
            }
            rowGroups.append(group)
            counts[group] += 1
        }
    }

    /// Assigns a first-seen group ID to each row.
    private func buildGroups() -> GroupIndex {
        let rowCount = dataFrame.shape.rows
        guard rowCount > 0, !groupColumns.isEmpty else { return GroupIndex(rowCapacity: 0) }

        if groupColumns.count == 1 {
            let column = dataFrame[column: groupColumns[0]]
            if let typed = column as? TypedColumn<Int64> {
                return buildIntegerGroups(typed.values)
            }
            if let typed = column as? TypedColumn<Int32> {
                return buildIntegerGroups(typed.values)
            }
            if let typed = column as? TypedColumn<Int> {
                return buildIntegerGroups(typed.values)
            }
        }

        // Fast path: single utf8 key column (common category / groupBy case).
        if groupColumns.count == 1,
           let typed = dataFrame[column: groupColumns[0], as: String.self] {
            var groupMap: [String: Int] = [:]
            var groups = GroupIndex(rowCapacity: rowCount)
            groupMap.reserveCapacity(16)
            let vals = typed.values
            for row in 0..<vals.count {
                let key = vals[row] ?? "__null__"
                if let group = groupMap[key] {
                    groups.append(group)
                } else {
                    groupMap[key] = groups.count
                    groups.append(groups.count)
                }
            }
            return groups
        }

        var groupMap: [String: Int] = [:]
        var groups = GroupIndex(rowCapacity: rowCount)
        let keyCols = groupColumns.compactMap { dataFrame[column: $0] }

        for row in 0..<rowCount {
            var key = ""
            key.reserveCapacity(32)
            for (i, col) in keyCols.enumerated() {
                if i > 0 { key.append("||") }
                if let v = col.value(at: row) {
                    key.append("\(v)")
                } else {
                    key.append("null")
                }
            }

            if let group = groupMap[key] {
                groups.append(group)
            } else {
                groupMap[key] = groups.count
                groups.append(groups.count)
            }
        }

        return groups
    }

    private func buildIntegerGroups<T: FixedWidthInteger>(_ values: [T?]) -> GroupIndex {
        var groupIndices: [T?: Int] = [:]
        var groups = GroupIndex(rowCapacity: values.count)
        for key in values {
            if let group = groupIndices[key] {
                groups.append(group)
            } else {
                groupIndices[key] = groups.count
                groups.append(groups.count)
            }
        }
        return groups
    }

    private func aggregate(using agg: Aggregation) -> DataFrame {
        let groups = buildGroups()
        var resultColumns: [any AnyColumn] = []

        // Group key columns — one representative row per group
        for keyCol in groupColumns {
            guard let col = dataFrame[column: keyCol] else { continue }
            let repValues = groups.representatives.map { row in
                col.value(at: row).map { "\($0)" }
            }
            resultColumns.append(TypedColumn<String>(name: keyCol, values: repValues))
        }

        // Numeric value columns
        let valueColumns = dataFrame.columns.filter {
            !groupColumns.contains($0.name) && $0.dtype.isNumeric
        }

        for col in valueColumns {
            if agg == .count {
                let counts: [Int64?] = groups.counts.map { Int64($0) }
                resultColumns.append(TypedColumn<Int64>(name: col.name, values: counts))
            } else {
                let aggValues = aggregateNumeric(col: col, groups: groups, agg: agg)
                resultColumns.append(TypedColumn<Double>(name: col.name, values: aggValues))
            }
        }

        if agg == .count && valueColumns.isEmpty {
            let counts: [Int64?] = groups.counts.map { Int64($0) }
            resultColumns.append(TypedColumn<Int64>(name: "count", values: counts))
        }

        return (try? DataFrame(columns: resultColumns)) ?? DataFrame.empty
    }

    private func aggregateNumeric(col: any AnyColumn, groups: GroupIndex, agg: Aggregation) -> [Double?] {
        if let typed = col as? TypedColumn<Double> {
            return applyNumeric(agg, to: typed.values, groups: groups)
        }
        if let typed = col as? TypedColumn<Float> {
            return applyNumeric(agg, to: typed.values, groups: groups)
        }
        if let typed = col as? TypedColumn<Int64> {
            return applyNumeric(agg, to: typed.values, groups: groups)
        }
        if let typed = col as? TypedColumn<Int32> {
            return applyNumeric(agg, to: typed.values, groups: groups)
        }
        let values = groups.rowGroups.indices.map { col.value(at: $0).flatMap { toDouble($0) } }
        let result = applyNumeric(agg, to: values, groups: groups)
        // The fallback historically returns nil when no values convert to Double.
        return agg == .count ? result.map { $0 == 0 ? nil : $0 } : result
    }

    private func applyNumeric<T: SupportedType>(_ agg: Aggregation, to vals: [T?], groups: GroupIndex) -> [Double?] {
        let ids = groups.rowGroups
        switch agg {
        case .count:
            var counts = [Int](repeating: 0, count: groups.count)
            for row in ids.indices where vals[row] != nil { counts[ids[row]] += 1 }
            return counts.map { Double($0) }
        case .sum, .mean:
            var sums = [Double](repeating: 0, count: groups.count)
            var counts = [Int](repeating: 0, count: groups.count)
            for row in ids.indices {
                if let value = vals[row]?.doubleValue {
                    let group = ids[row]
                    sums[group] += value
                    counts[group] += 1
                }
            }
            return sums.indices.map { group in
                guard counts[group] > 0 else { return nil }
                return agg == .mean ? sums[group] / Double(counts[group]) : sums[group]
            }
        case .min, .max:
            var result = [Double?](repeating: nil, count: groups.count)
            for row in ids.indices {
                guard let value = vals[row]?.doubleValue else { continue }
                let group = ids[row]
                if result[group] == nil || (agg == .min ? value < result[group]! : value > result[group]!) {
                    result[group] = value
                }
            }
            return result
        case .first:
            var result = [Double?](repeating: nil, count: groups.count)
            var remaining = groups.count
            for row in ids.indices {
                let group = ids[row]
                if result[group] == nil, let value = vals[row]?.doubleValue {
                    result[group] = value
                    remaining -= 1
                    if remaining == 0 { break }
                }
            }
            return result
        case .last:
            var result = [Double?](repeating: nil, count: groups.count)
            var remaining = groups.count
            for row in ids.indices.reversed() {
                let group = ids[row]
                if result[group] == nil, let value = vals[row]?.doubleValue {
                    result[group] = value
                    remaining -= 1
                    if remaining == 0 { break }
                }
            }
            return result
        }
    }

    private func aggLabel(_ agg: Aggregation) -> String {
        switch agg {
        case .sum:   return "sum"
        case .mean:  return "mean"
        case .min:   return "min"
        case .max:   return "max"
        case .count: return "count"
        case .first: return "first"
        case .last:  return "last"
        }
    }

    private func toDouble(_ v: Any) -> Double? {
        switch v {
        case let x as Double: return x
        case let x as Float:  return Double(x)
        case let x as Int64:  return Double(x)
        case let x as Int32:  return Double(x)
        default: return nil
        }
    }
}
