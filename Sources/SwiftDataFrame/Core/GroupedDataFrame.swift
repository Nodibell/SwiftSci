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
    public func count() -> DataFrame {
        aggregate(using: .count)
    }

    /// Returns the mean of each numeric column per group.
    public func mean() -> DataFrame {
        aggregate(using: .mean)
    }

    /// Returns the sum of each numeric column per group.
    public func sum() -> DataFrame {
        aggregate(using: .sum)
    }

    /// Returns the min of each numeric column per group.
    public func min() -> DataFrame {
        aggregate(using: .min)
    }

    /// Returns the max of each numeric column per group.
    public func max() -> DataFrame {
        aggregate(using: .max)
    }

    /// Apply multiple aggregations per column.
    public func agg(_ aggregations: [String: Aggregation]) -> DataFrame {
        let groups = buildGroups()
        var resultColumns: [any AnyColumn] = []

        // Group key columns
        for keyCol in groupColumns {
            guard let col = dataFrame[column: keyCol] else { continue }
            let keys = groups.map { group in
                group.first.flatMap { col.value(at: $0) }
            }
            let strKeys = keys.map { $0.map { "\($0)" } }
            resultColumns.append(TypedColumn<String>(name: keyCol, values: strKeys))
        }

        // Aggregated columns
        for (colName, agg) in aggregations {
            guard let col = dataFrame[column: colName], col.dtype.isNumeric else { continue }
            let aggValues: [Double?] = groups.map { indices in
                let nums = indices.compactMap { col.value(at: $0).flatMap { toDouble($0) } }
                return apply(agg, to: nums)
            }
            resultColumns.append(TypedColumn<Double>(name: "\(colName)_\(aggLabel(agg))",
                                                      values: aggValues))
        }

        return (try? DataFrame(columns: resultColumns)) ?? DataFrame.empty
    }

    // MARK: – Private helpers

    /// Groups row indices by the unique combination of groupColumn values.
    private func buildGroups() -> [[Int]] {
        var groupMap: [String: [Int]] = [:]
        var order: [String] = []

        for row in 0..<dataFrame.shape.rows {
            let key = groupColumns.map { col in
                dataFrame[column: col].map { "\($0.value(at: row) ?? "null")" } ?? "null"
            }.joined(separator: "||")

            if groupMap[key] == nil {
                groupMap[key] = []
                order.append(key)
            }
            groupMap[key]?.append(row)
        }

        return order.compactMap { groupMap[$0] }
    }

    private func aggregate(using agg: Aggregation) -> DataFrame {
        let groups = buildGroups()
        var resultColumns: [any AnyColumn] = []

        // Group key columns — one representative row per group
        for keyCol in groupColumns {
            guard let col = dataFrame[column: keyCol] else { continue }
            let repValues = groups.map { indices -> String? in
                indices.first.flatMap { col.value(at: $0).map { "\($0)" } }
            }
            resultColumns.append(TypedColumn<String>(name: keyCol, values: repValues))
        }

        // Numeric value columns
        let valueColumns = dataFrame.columns.filter {
            !groupColumns.contains($0.name) && $0.dtype.isNumeric
        }

        for col in valueColumns {
            if agg == .count {
                let counts: [Int64?] = groups.map { Int64($0.count) }
                resultColumns.append(TypedColumn<Int64>(name: col.name, values: counts))
            } else {
                let aggValues: [Double?] = groups.map { indices in
                    let nums = indices.compactMap { col.value(at: $0).flatMap { toDouble($0) } }
                    return apply(agg, to: nums)
                }
                resultColumns.append(TypedColumn<Double>(name: col.name, values: aggValues))
            }
        }

        if agg == .count && valueColumns.isEmpty {
            // If no numeric columns, just return count column
            let counts: [Int64?] = groups.map { Int64($0.count) }
            resultColumns.append(TypedColumn<Int64>(name: "count", values: counts))
        }

        return (try? DataFrame(columns: resultColumns)) ?? DataFrame.empty
    }

    private func apply(_ agg: Aggregation, to nums: [Double]) -> Double? {
        guard !nums.isEmpty else { return nil }
        switch agg {
        case .sum:   return nums.reduce(0, +)
        case .mean:  return nums.reduce(0, +) / Double(nums.count)
        case .min:   return nums.min()
        case .max:   return nums.max()
        case .count: return Double(nums.count)
        case .first: return nums.first
        case .last:  return nums.last
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
