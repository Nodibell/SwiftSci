import Foundation

/// A strongly-typed column backed by a contiguous Swift array.
/// `nil` elements represent null / missing values.
public struct TypedColumn<T: SupportedType>: AnyColumn {

    // MARK: – AnyColumn

    /// The name.
    public let name: String
    /// The dtype.
    public let dtype: ColumnDType
    /// The count.
    public var count: Int     { values.count }
    /// The null count.
    public var nullCount: Int { _nullCount }

    // MARK: – Storage

    /// Raw values — `nil` means null.
    public let values: [T?]

    private let _nullCount: Int

    // MARK: – Init

    /// Creates a new instance with optional values.
    /// - Parameters:
    ///   - name: The name.
    ///   - values: The values (optional elements).
    public init(name: String, values: [T?]) {
        var nullCount = 0
        for case nil in values {
            nullCount += 1
        }
        self.init(name: name, values: values, nullCount: nullCount)
    }

    private init(name: String, values: [T?], nullCount: Int) {
        self.name = name
        self.dtype = T.columnDType
        self.values = values
        self._nullCount = nullCount
    }

    /// Creates a new instance with non-optional values.
    /// - Parameters:
    ///   - name: The name.
    ///   - values: The values (non-optional elements).
    public init(name: String, values: [T]) {
        self.init(name: name, values: values.map { $0 as T? }, nullCount: 0)
    }


    // MARK: – Subscript

    /// Accesses the element at the given index.
    /// - Parameters:
    ///   - index: The index.
    public subscript(index: Int) -> T? { values[index] }

    // MARK: – AnyColumn conformance

    /// Returns a new column containing elements where the boolean mask is `true`.
    /// - Parameter mask: Array of boolean flags indicating which rows to keep.
    /// - Throws: `DataFrameError.columnLengthMismatch` if the mask length does not match column count.
    /// - Returns: A new `AnyColumn` filtered by the mask.
    public func filtered(by mask: [Bool]) throws -> any AnyColumn {
        guard mask.count == count else {
            throw SwiftMLError.columnLengthMismatch(
                expected: count, got: mask.count, column: name
            )
        }
        var kept = 0
        for flag in mask where flag { kept += 1 }
        var result: [T?] = []
        result.reserveCapacity(kept)
        var nullCount = 0
        for (val, keep) in zip(values, mask) where keep {
            result.append(val)
            if case nil = val { nullCount += 1 }
        }
        return TypedColumn<T>(name: name, values: result, nullCount: nullCount)
    }

    /// Returns a new column containing only unique elements (preserving order of first appearance).
    public var unique: any AnyColumn {
        return typedUnique
    }

    /// Strongly-typed property returning `TypedColumn<T>` containing only unique elements (preserving order of first appearance).
    public var typedUnique: TypedColumn<T> {
        var seen: Set<T> = []
        var seenNull = false
        var uniqueValues: [T?] = []
        uniqueValues.reserveCapacity(values.count)
        
        for val in values {
            if let v = val {
                if seen.insert(v).inserted {
                    uniqueValues.append(v)
                }
            } else {
                if !seenNull {
                    seenNull = true
                    uniqueValues.append(nil)
                }
            }
        }
        return TypedColumn<T>(name: name, values: uniqueValues, nullCount: seenNull ? 1 : 0)
    }

    /// Gathers elements at the specified row indices using typed indexed loops.
    /// - Parameter indices: Array of row indices to gather.
    /// - Returns: A new `AnyColumn` containing elements at the requested indices.
    public func gathered(at indices: [Int]) -> any AnyColumn {
        if let doubleCol = self as? TypedColumn<Double> {
            return doubleCol.vGather(at: indices)
        }
        if let int64Col = self as? TypedColumn<Int64> {
            return int64Col.gatheredNumeric(at: indices)
        }
        if let column = self as? TypedColumn<Int> {
            return column.gatheredNumeric(at: indices)
        }
        if let column = self as? TypedColumn<Int32> {
            return column.gatheredNumeric(at: indices)
        }
        if let column = self as? TypedColumn<Float> {
            return column.gatheredNumeric(at: indices)
        }
        let n = indices.count
        var result = Array<T?>(repeating: nil, count: n)
        var gatheredNullCount = 0
        values.withUnsafeBufferPointer { srcBuf in
            indices.withUnsafeBufferPointer { idxBuf in
                result.withUnsafeMutableBufferPointer { dstBuf in
                    guard let src = srcBuf.baseAddress,
                          let idx = idxBuf.baseAddress,
                          let dst = dstBuf.baseAddress else { return }
                    for k in 0..<n {
                        let value = src[idx[k]]
                        dst[k] = value
                        if _nullCount > 0, case nil = value { gatheredNullCount += 1 }
                    }
                }
            }
        }
        return TypedColumn<T>(name: name, values: result, nullCount: gatheredNullCount)
    }

    /// Evaluates supported conditions on typed column values and returns matching row indices.
    /// - Parameter condition: Filter condition comparison operator and threshold.
    /// - Returns: Array of row indices matching the condition, or `nil` if unsupported.
    public func filteredIndices(matching condition: FilterCondition) -> [Int]? {
        if let column = self as? TypedColumn<Double> {
            return filterIndicesFloating(values: column.values, condition: condition)
        }
        if let column = self as? TypedColumn<Int64> {
            return filterIndicesInteger(values: column.values, condition: condition)
        }
        if let column = self as? TypedColumn<Float> {
            return filterIndicesFloating(values: column.values, condition: condition)
        }
        if let column = self as? TypedColumn<Int32> {
            return filterIndicesInteger(values: column.values, condition: condition)
        }
        if let column = self as? TypedColumn<Int> {
            return filterIndicesInteger(values: column.values, condition: condition)
        }
        if let column = self as? TypedColumn<String> {
            return filterIndicesString(values: column.values, condition: condition)
        }
        return nil
    }

    /// Accesses raw value at the specified row index as `Any?`.
    /// - Parameter index: Row index to inspect.
    /// - Returns: Value at row index or `nil` if null or out of bounds.
    public func value(at index: Int) -> Any? {
        guard index >= 0 && index < count else { return nil }
        return values[index] as Any?
    }

    /// Converts numeric column values into an array of non-null `Double` values.
    /// - Returns: Array of `Double` values or `nil` if column is non-numeric.
    public func toDoubles() -> [Double]? {
        guard dtype.isNumeric else { return nil }
        return values.compactMap { $0?.doubleValue }
    }

    /// Converts column values into an array of string representations.
    /// - Returns: Array of string values with `"null"` for missing values.
    public func toStrings() -> [String] {
        values.map { v in
            guard let v else { return "null" }
            return "\(v)"
        }
    }

    /// Returns a new column instance with a renamed identifier.
    /// - Parameter newName: Target column name.
    /// - Returns: Renamed column instance.
    public func renamed(to newName: String) -> any AnyColumn {
        TypedColumn<T>(name: newName, values: values, nullCount: _nullCount)
    }

    /// Computes a stable row permutation sorted by column values, with nulls last.
    /// - Parameter ascending: Sort order direction (`true` for ascending, `false` for descending).
    /// - Returns: Array of row indices sorted by column values.
    public func sortedIndices(ascending: Bool) -> [Int] {
        var indices = Array(0..<values.count)
        let vals = values

        // Specialize common Comparable element types without constraining SupportedType
        // (Bool is Hashable but not Comparable).
        if let column = self as? TypedColumn<Double> {
            return sortIndicesPrimitiveFast(column.values, nullCount: column.nullCount, ascending: ascending)
        }
        if let column = self as? TypedColumn<Float> {
            return sortIndicesPrimitiveFast(column.values, nullCount: column.nullCount, ascending: ascending)
        }
        if let column = self as? TypedColumn<Int64> {
            return sortIndicesPrimitiveFast(column.values, nullCount: column.nullCount, ascending: ascending)
        }
        if let column = self as? TypedColumn<Int32> {
            return sortIndicesPrimitiveFast(column.values, nullCount: column.nullCount, ascending: ascending)
        }
        if let column = self as? TypedColumn<Int> {
            return sortIndicesPrimitiveFast(column.values, nullCount: column.nullCount, ascending: ascending)
        }
        if let strings = vals as? [String?] {
            sortIndices(&indices, ascending: ascending) { strings[$0] }
            return indices
        }
        if let dates = vals as? [Date?] {
            sortIndices(&indices, ascending: ascending) { dates[$0] }
            return indices
        }
        if let bools = vals as? [Bool?] {
            sortIndices(&indices, ascending: ascending) { i -> Int? in
                bools[i].map { $0 ? 1 : 0 }
            }
            return indices
        }

        // Fallback: numeric promotion then string
        indices.sort { i, j in
            switch (vals[i], vals[j]) {
            case (nil, nil): return false
            case (nil, _):   return false
            case (_, nil):   return true
            case let (l?, r?):
                if let ld = l.doubleValue, let rd = r.doubleValue {
                    return ascending ? ld < rd : ld > rd
                }
                let ls = "\(l)", rs = "\(r)"
                return ascending ? ls < rs : ls > rs
            }
        }
        return indices
    }

    // MARK: – TypedColumn-specific operations

    /// Returns a new column by applying a transform to every element.
    /// - Parameters:
    ///   - transform: Transformation closure or mapping function.
    /// - Returns: A strongly-typed column containing the computed values.
    public func map<U: SupportedType>(_ transform: (T?) -> U?) -> TypedColumn<U> {
        TypedColumn<U>(name: name, values: values.map(transform))
    }

    /// Applies transform only to non-null elements; nil inputs are passed through as nil.
    /// - Parameters:
    ///   - transform: Transformation closure or mapping function.
    /// - Returns: A strongly-typed column containing the computed values.
    public func compactMap<U: SupportedType>(_ transform: (T) -> U?) -> TypedColumn<U> {
        TypedColumn<U>(name: name, values: values.map { $0.flatMap(transform) })
    }
    /// Lagged.
    /// - Returns: A `any AnyColumn` result.
    /// - Parameters:
    ///   - offset: Byte or element offset within the buffer.
    public func lagged(by offset: Int) -> any AnyColumn {
        var newValues = [T?](repeating: nil, count: count)
        if offset > 0 {
            if offset < count {
                for i in offset..<count {
                    newValues[i] = values[i - offset]
                }
            }
        } else if offset < 0 {
            let absOffset = abs(offset)
            if absOffset < count {
                for i in 0..<(count - absOffset) {
                    newValues[i] = values[i + absOffset]
                }
            }
        } else {
            newValues = values
        }
        return TypedColumn<T>(name: name, values: newValues)
    }

    /// Returns a new column with all null values removed.
    /// - Returns: A strongly-typed column containing the computed values.
    public func dropNulls() -> TypedColumn<T> {
        TypedColumn<T>(name: name, values: values.compactMap { $0 }.map { Optional($0) })
    }

    /// Returns a new column where null values are replaced by `value`.
    /// - Parameters:
    ///   - value: Scalar value to assign, check, or replace.
    /// - Returns: A strongly-typed column containing the computed values.
    public func fillNull(with value: T) -> TypedColumn<T> {
        TypedColumn<T>(name: name, values: values.map { $0 ?? value })
    }

    /// Returns non-null values as a plain array.
    public var nonNullValues: [T] { values.compactMap { $0 } }
}

/// Sorts cached non-null keys once and appends missing rows in their original order.
private func sortIndicesPrimitiveFast<T: Comparable>(_ vals: [T?], nullCount: Int, ascending: Bool) -> [Int] {
    var tagged: [(index: Int, value: T)] = []
    var missing: [Int] = []
    tagged.reserveCapacity(vals.count - nullCount)
    missing.reserveCapacity(nullCount)
    for (index, value) in vals.enumerated() {
        if let value {
            // NaN does not define a strict ordering. Preserve the existing comparison path.
            if value != value {
                var indices = Array(vals.indices)
                sortIndices(&indices, ascending: ascending) { vals[$0] }
                return indices
            }
            tagged.append((index, value))
        } else {
            missing.append(index)
        }
    }
    // Keep direction inside one comparator for the Swift 6.4 Release workaround.
    tagged.sort { ascending ? $0.value < $1.value : $0.value > $1.value }
    var indices = tagged.map { $0.index }
    indices.append(contentsOf: missing)
    return indices
}

/// Nulls-last index sort over optional Comparable keys.
private func sortIndices<C: Comparable>(_ indices: inout [Int], ascending: Bool, key: (Int) -> C?) {
    // Swift 6.4 -O can miscompile separate ascending/descending generic closures.
    // Keep direction in one closure to avoid that optimization defect.
    indices.sort { i, j in
        switch (key(i), key(j)) {
        case (nil, nil): return false
        case (nil, _):   return false
        case (_, nil):   return true
        case let (l?, r?): return ascending ? l < r : l > r
        }
    }
}

import Foundation
import Accelerate

// MARK: – Double column filter & vDSP vectorised reductions fast path

extension TypedColumn where T == Double {
    /// Computes sample mean using Accelerate vDSP.
    /// - Returns: The computed arithmetic mean value.
    public func mean() -> Double {
        let nonNulls = nonNullValues
        guard !nonNulls.isEmpty else { return 0.0 }
        var result = 0.0
        vDSP_meanvD(nonNulls, 1, &result, vDSP_Length(nonNulls.count))
        return result
    }

    /// Computes sample variance using two-pass vDSP operations with Bessel's correction.
    /// - Returns: The calculated sample or population variance.
    public func variance() -> Double {
        let nonNulls = nonNullValues
        let count = nonNulls.count
        guard count > 1 else { return 0.0 }
        let length = vDSP_Length(count)

        var meanVal = 0.0
        var meanSquareVal = 0.0

        vDSP_meanvD(nonNulls, 1, &meanVal, length)
        vDSP_measqvD(nonNulls, 1, &meanSquareVal, length)

        let popVariance = meanSquareVal - (meanVal * meanVal)
        let besselCorrection = Double(count) / Double(count - 1)
        return max(0.0, popVariance * besselCorrection)
    }

    /// Computes sample standard deviation using Accelerate vDSP.
    /// - Returns: The computed standard deviation.
    public func stdDev() -> Double {
        sqrt(variance())
    }

    /// Indexed gather for Double columns, preserving missing values.
    /// - Parameters:
    ///   - indices: Array of row or column integer indices.
    /// - Returns: A strongly-typed column containing the computed values.
    public func vGather(at indices: [Int]) -> TypedColumn<Double> {
        let n = indices.count
        guard n > 0 else { return TypedColumn<Double>(name: name, values: []) }

        var result = [Double?](repeating: nil, count: n)
        var gatheredNullCount = 0
        values.withUnsafeBufferPointer { srcBuf in
            indices.withUnsafeBufferPointer { idxBuf in
                result.withUnsafeMutableBufferPointer { dstBuf in
                    guard let src = srcBuf.baseAddress,
                          let idx = idxBuf.baseAddress,
                          let dst = dstBuf.baseAddress else { return }
                    for i in 0..<n {
                        let value = src[idx[i]]
                        dst[i] = value
                        if _nullCount > 0, case nil = value { gatheredNullCount += 1 }
                    }
                }
            }
        }
        return TypedColumn<Double>(name: name, values: result, nullCount: gatheredNullCount)
    }

    /// Builds a row mask for common numeric `FilterCondition`s without type erasure.
    /// Returns `nil` when the condition is not a numeric comparison handled here.
    func mask(matching condition: FilterCondition) -> [Bool]? {
        switch condition {
        case .isNull:
            return values.map { $0 == nil }
        case .isNotNull:
            return values.map { $0 != nil }
        case .greaterThan(let rhs):
            guard let thr = Self.asDouble(rhs) else { return nil }
            return values.map { ($0 ?? .nan) > thr }
        case .lessThan(let rhs):
            guard let thr = Self.asDouble(rhs) else { return nil }
            return values.map { ($0 ?? .nan) < thr }
        case .greaterThanOrEqual(let rhs):
            guard let thr = Self.asDouble(rhs) else { return nil }
            return values.map { ($0 ?? .nan) >= thr }
        case .lessThanOrEqual(let rhs):
            guard let thr = Self.asDouble(rhs) else { return nil }
            return values.map { ($0 ?? .nan) <= thr }
        case .equals(let rhs):
            guard let thr = Self.asDouble(rhs) else { return nil }
            return values.map { $0 == Optional(thr) }
        case .notEquals(let rhs):
            guard let thr = Self.asDouble(rhs) else { return nil }
            return values.map { $0 != Optional(thr) }
        case .contains:
            return nil
        }
    }

    private static func asDouble(_ value: Any) -> Double? {
        switch value {
        case let x as Double: return x
        case let x as Float:  return Double(x)
        case let x as Int64:  return Double(x)
        case let x as Int32:  return Double(x)
        case let x as Int:    return Double(x)
        default: return nil
        }
    }
}

// MARK: – Typed filter indices

private func filterIndicesFloating<T: BinaryFloatingPoint>(values: [T?], condition: FilterCondition) -> [Int]? {
    switch condition {
    case .isNull: return values.indices.filter { values[$0] == nil }
    case .isNotNull: return selectedIndices(values) { _ in true }
    default: break
    }
    guard let comparison = numericComparison(condition), let threshold = toDouble(comparison.1) else { return nil }
    var operation = comparison.0
    if let integer = toInt64(comparison.1), Int64(exactly: threshold) != integer {
        // No representable Double lies between this rounded threshold and the integer.
        let roundedBelow = Int64(exactly: threshold).map { $0 < integer } ?? false
        switch operation {
        case .equal: return []
        case .notEqual: return selectedIndices(values) { _ in true }
        case .less, .lessEqual: operation = roundedBelow ? .lessEqual : .less
        case .greater, .greaterEqual: operation = roundedBelow ? .greater : .greaterEqual
        }
    }
    switch operation {
    case .equal: return selectedIndices(values) { Double($0) == threshold }
    case .notEqual: return selectedIndices(values) { Double($0) != threshold }
    case .less: return selectedIndices(values) { Double($0) < threshold }
    case .lessEqual: return selectedIndices(values) { Double($0) <= threshold }
    case .greater: return selectedIndices(values) { Double($0) > threshold }
    case .greaterEqual: return selectedIndices(values) { Double($0) >= threshold }
    }
}

private enum NumericComparison {
    case equal, notEqual, less, lessEqual, greater, greaterEqual
}

private func numericComparison(_ condition: FilterCondition) -> (NumericComparison, any Sendable)? {
    switch condition {
    case .equals(let value): return (.equal, value)
    case .notEquals(let value): return (.notEqual, value)
    case .lessThan(let value): return (.less, value)
    case .lessThanOrEqual(let value): return (.lessEqual, value)
    case .greaterThan(let value): return (.greater, value)
    case .greaterThanOrEqual(let value): return (.greaterEqual, value)
    default: return nil
    }
}

private func selectedIndices<T>(_ values: [T?], where predicate: (T) -> Bool) -> [Int] {
    var result: [Int] = []
    result.reserveCapacity(values.count / 2)
    for (index, value) in values.enumerated() {
        if let value, predicate(value) { result.append(index) }
    }
    return result
}

private func filterIndicesInteger<T: FixedWidthInteger & SignedInteger>(values: [T?], condition: FilterCondition) -> [Int]? {
    switch condition {
    case .isNull: return values.indices.filter { values[$0] == nil }
    case .isNotNull: return selectedIndices(values) { _ in true }
    default: break
    }
    guard let comparison = numericComparison(condition) else { return nil }
    var operation = comparison.0
    let rhs = comparison.1

    let threshold: Int64
    if let integer = toInt64(rhs) {
        threshold = integer
    } else if let floating = toDouble(rhs) {
        // Classify once. Never round a column integer through Double or trap on a cast.
        guard let floor = Int64(exactly: floating.rounded(.down)) else {
            let matches: Bool
            if floating.isNaN {
                matches = operation == .notEqual
            } else if floating > 0 {
                matches = operation == .less || operation == .lessEqual || operation == .notEqual
            } else {
                matches = operation == .greater || operation == .greaterEqual || operation == .notEqual
            }
            return matches ? selectedIndices(values) { _ in true } : []
        }
        threshold = floor
        if floating != Double(floor) {
            // For fractional r: x < r iff x <= floor(r), x >= r iff x > floor(r).
            switch operation {
            case .equal: return []
            case .notEqual: return selectedIndices(values) { _ in true }
            case .less, .lessEqual: operation = .lessEqual
            case .greater, .greaterEqual: operation = .greater
            }
        }
    } else {
        return nil
    }

    switch operation {
    case .equal: return selectedIndices(values) { Int64($0) == threshold }
    case .notEqual: return selectedIndices(values) { Int64($0) != threshold }
    case .less: return selectedIndices(values) { Int64($0) < threshold }
    case .lessEqual: return selectedIndices(values) { Int64($0) <= threshold }
    case .greater: return selectedIndices(values) { Int64($0) > threshold }
    case .greaterEqual: return selectedIndices(values) { Int64($0) >= threshold }
    }
}

private func filterIndicesString(values: [String?], condition: FilterCondition) -> [Int]? {
    var res = [Int]()
    res.reserveCapacity(values.count / 2)
    switch condition {
    case .isNull:
        for (i, v) in values.enumerated() where v == nil { res.append(i) }
    case .isNotNull:
        for (i, v) in values.enumerated() where v != nil { res.append(i) }
    case .equals(let rhs):
        guard let str = rhs as? String else { return nil }
        for (i, v) in values.enumerated() { if let x = v, x == str { res.append(i) } }
    case .notEquals(let rhs):
        guard let str = rhs as? String else { return nil }
        for (i, v) in values.enumerated() { if let x = v, x != str { res.append(i) } }
    case .contains(let substring):
        for (i, v) in values.enumerated() { if let x = v, x.contains(substring) { res.append(i) } }
    default:
        return nil
    }
    return res
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

private func toInt64(_ v: Any) -> Int64? {
    switch v {
    case let x as Int64:  return x
    case let x as Int:    return Int64(x)
    case let x as Int32:  return Int64(x)
    default: return nil
    }
}

// MARK: – Equatable (for testing)
extension TypedColumn: Equatable where T: Equatable {
    /// A public declaration.
    public static func == (lhs: TypedColumn<T>, rhs: TypedColumn<T>) -> Bool {
        lhs.name == rhs.name && lhs.values == rhs.values
    }
}


private extension TypedColumn {
    func gatheredNumeric(at indices: [Int]) -> TypedColumn<T> {
        if _nullCount == 0 {
            return TypedColumn<T>(name: name, values: indices.map { values[$0] }, nullCount: 0)
        }
        var gatheredNullCount = 0
        let result = indices.map { index in
            let value = values[index]
            if case nil = value { gatheredNullCount += 1 }
            return value
        }
        return TypedColumn<T>(name: name, values: result, nullCount: gatheredNullCount)
    }
}
