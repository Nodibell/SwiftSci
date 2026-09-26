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
    /// Uses compensated Double summation before dividing by the non-null count.
    /// Integer conversion and intermediate overflow retain Double semantics.
    /// - Returns: A new `DataFrame` containing the transformed columns and computed results.
    public func mean() -> DataFrame {
        aggregate(using: .mean)
    }

    /// Returns the sum of each numeric column per group.
    /// Uses Neumaier compensation to reduce rounding loss; all-null groups return nil.
    /// Results use Double, so large integers can lose precision and sums can overflow.
    /// - Returns: A new `DataFrame` containing the transformed columns and computed results.
    public func sum() -> DataFrame {
        aggregate(using: .sum)
    }

    /// Sums numeric columns while preserving exact integer totals.
    /// Int32, Int64 and Int values accumulate without floating-point conversion and
    /// return Int64 columns. A temporary total outside Int64 is allowed if the final
    /// total fits. Floating columns use the same compensated Double sums as `sum()`.
    /// Null values are skipped; groups with no non-null values return nil.
    /// Group keys and their order follow `sum()`.
    /// - Throws: `SwiftMLError.integerOverflow` if a final integer total cannot fit
    ///   in Int64. Its group index is zero-based, in first-appearance order.
    ///   Custom integer columns must expose Int32, Int64 or Int values; other values
    ///   throw `SwiftMLError.typeMismatch` rather than converting through Double.
    /// - Returns: A DataFrame with Int64 integer sums and Double floating sums.
    public func sumChecked() throws -> DataFrame {
        let groups = buildGroups()
        var resultColumns = groupKeyColumns(groups)
        for col in dataFrame.columns where !groupColumns.contains(col.name) && col.dtype.isNumeric {
            let integerValues: [Int64?]
            if let typed = col as? TypedColumn<Int64> {
                integerValues = try sumIntegers(typed.values, groups: groups, name: col.name)
            } else if let typed = col as? TypedColumn<Int32> {
                integerValues = try sumIntegers(typed.values, groups: groups, name: col.name)
            } else if let typed = col as? TypedColumn<Int> {
                integerValues = try sumIntegers(typed.values, groups: groups, name: col.name)
            } else if col.dtype == .int32 || col.dtype == .int64 {
                let values: [Int64?] = try groups.rowGroups.indices.map { row in
                    guard let value = col.value(at: row) else { return nil }
                    switch value {
                    case let value as Int64: return value
                    case let value as Int32: return Int64(value)
                    case let value as Int: return Int64(value)
                    default:
                        throw SwiftMLError.typeMismatch(column: col.name,
                            expected: "Int32, Int64 or Int", got: String(describing: type(of: value)))
                    }
                }
                integerValues = try sumIntegers(values, groups: groups, name: col.name)
            } else {
                resultColumns.append(TypedColumn<Double>(name: col.name,
                    values: aggregateNumeric(col: col, groups: groups, agg: .sum)))
                continue
            }
            resultColumns.append(TypedColumn<Int64>(name: col.name, values: integerValues))
        }
        return resultColumns.isEmpty ? DataFrame.empty : try DataFrame(columns: resultColumns)
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
        var resultColumns = groupKeyColumns(groups)

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
        if let groups = buildBoundedIntegerGroups(values) { return groups }
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

    private func buildBoundedIntegerGroups<T: FixedWidthInteger>(_ values: [T?]) -> GroupIndex? {
        // Bound the table to 65,536 entries and at most one entry per input row.
        let slotLimit = Swift.min(values.count, 65_536)
        var lower = T.max
        var upper = T.min
        for case let value? in values {
            if value >= lower && value <= upper { continue }
            lower = Swift.min(lower, value)
            upper = Swift.max(upper, value)
            let (span, overflow) = upper.subtractingReportingOverflow(lower)
            guard !overflow, let width = Int(exactly: span), width < slotLimit else { return nil }
        }

        var groups = GroupIndex(rowCapacity: values.count)
        if lower > upper {
            for _ in values { groups.append(0) }
            return groups
        }

        let slotCount = Int(upper - lower) + 1
        var lookup = [Int](repeating: -1, count: slotCount)
        var nullGroup = -1
        for value in values {
            let group: Int
            if let value {
                let slot = Int(value - lower)
                if lookup[slot] == -1 { lookup[slot] = groups.count }
                group = lookup[slot]
            } else {
                if nullGroup == -1 { nullGroup = groups.count }
                group = nullGroup
            }
            groups.append(group)
        }
        return groups
    }

    private func groupKeyColumns(_ groups: GroupIndex) -> [any AnyColumn] {
        groupColumns.compactMap { name in
            guard let column = dataFrame[column: name] else { return nil }
            let keys = groups.representatives.map { row in
                column.value(at: row).map { "\($0)" }
            }
            return TypedColumn<String>(name: name, values: keys)
        }
    }

    // A Swift array holds at most Int.max signed 64-bit values, so their total
    // fits in signed 128 bits. Two words keep this available on macOS 14.
    private struct IntegerSum {
        var low: UInt64 = 0
        var high: Int64 = 0
        var hasValue = false

        mutating func add(_ value: Int64) {
            let (next, carry) = low.addingReportingOverflow(UInt64(bitPattern: value))
            low = next
            high += (value < 0 ? -1 : 0) + (carry ? 1 : 0)
            hasValue = true
        }

        var int64: Int64? {
            let value = Int64(bitPattern: low)
            return high == (value < 0 ? -1 : 0) ? value : nil
        }
    }

    private func sumIntegers<T: FixedWidthInteger & SignedInteger>(
        _ values: [T?], groups: GroupIndex, name: String
    ) throws -> [Int64?] {
        var sums = [IntegerSum](repeating: IntegerSum(), count: groups.count)
        for row in groups.rowGroups.indices {
            if let value = values[row] {
                sums[groups.rowGroups[row]].add(Int64(value))
            }
        }
        return try sums.indices.map { group in
            guard sums[group].hasValue else { return nil }
            guard let value = sums[group].int64 else {
                throw SwiftMLError.integerOverflow(column: name, group: group)
            }
            return value
        }
    }

    private func aggregate(using agg: Aggregation) -> DataFrame {
        let groups = buildGroups()
        var resultColumns = groupKeyColumns(groups)

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
        if let typed = col as? TypedColumn<Int> {
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
            var corrections = [Double](repeating: 0, count: groups.count)
            var counts = [Int](repeating: 0, count: groups.count)
            for row in ids.indices {
                if let value = vals[row]?.doubleValue {
                    let group = ids[row]
                    let sum = sums[group]
                    let next = sum + value
                    if next.isFinite {
                        // Recover the low-order contribution lost by the larger operand.
                        corrections[group] += abs(sum) >= abs(value)
                            ? (sum - next) + value
                            : (value - next) + sum
                    } else {
                        // Keep IEEE infinity/NaN propagation without inf - inf in the correction.
                        corrections[group] = 0
                    }
                    sums[group] = next
                    counts[group] += 1
                }
            }
            return sums.indices.map { group in
                guard counts[group] > 0 else { return nil }
                let total = sums[group] + corrections[group]
                return agg == .mean ? total / Double(counts[group]) : total
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
        case let x as Int:    return Double(x)
        default: return nil
        }
    }
}
