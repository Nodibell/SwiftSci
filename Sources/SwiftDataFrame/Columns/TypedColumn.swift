import Foundation

/// A strongly-typed column backed by a contiguous Swift array.
/// `nil` elements represent null / missing values.
public struct TypedColumn<T: SupportedType>: AnyColumn {

    // MARK: – AnyColumn

    public let name: String
    public let dtype: ColumnDType
    public var count: Int     { values.count }
    public var nullCount: Int { _nullCount }

    // MARK: – Storage

    /// Raw values — `nil` means null.
    public let values: [T?]

    private let _nullCount: Int

    // MARK: – Init

    public init(name: String, values: [T?]) {
        self.name       = name
        self.dtype      = T.columnDType
        self.values     = values
        self._nullCount = values.reduce(0) { $0 + ($1 == nil ? 1 : 0) }
    }

    // MARK: – Subscript

    public subscript(index: Int) -> T? { values[index] }

    // MARK: – AnyColumn conformance

    public func filtered(by mask: [Bool]) throws -> any AnyColumn {
        guard mask.count == count else {
            throw DataFrameError.columnLengthMismatch(
                expected: count, got: mask.count, column: name
            )
        }
        var kept = 0
        for flag in mask where flag { kept += 1 }
        var result: [T?] = []
        result.reserveCapacity(kept)
        for (val, keep) in zip(values, mask) where keep {
            result.append(val)
        }
        return TypedColumn<T>(name: name, values: result)
    }

    public func gathered(at indices: [Int]) -> any AnyColumn {
        if T.self == Double.self {
            let doubleCol = self as! TypedColumn<Double>
            return doubleCol.vGather(at: indices)
        }
        var result = Array<T?>(repeating: nil, count: indices.count)
        let vals = values
        for k in 0..<indices.count {
            result[k] = vals[indices[k]]
        }
        return TypedColumn<T>(name: name, values: result)
    }

    public func filteredIndices(matching condition: FilterCondition) -> [Int]? {
        if T.self == Double.self {
            let doubles = values as! [Double?]
            return filterIndicesDouble(values: doubles, condition: condition)
        }
        if T.self == Int64.self {
            let ints = values as! [Int64?]
            return filterIndicesInt64(values: ints, condition: condition)
        }
        if T.self == String.self {
            let strings = values as! [String?]
            return filterIndicesString(values: strings, condition: condition)
        }
        return nil
    }

    public func value(at index: Int) -> Any? {
        guard index >= 0 && index < count else { return nil }
        return values[index] as Any?
    }

    public func toDoubles() -> [Double]? {
        guard dtype.isNumeric else { return nil }
        return values.compactMap { $0?.doubleValue }
    }

    public func toStrings() -> [String] {
        values.map { v in
            guard let v else { return "null" }
            return "\(v)"
        }
    }

    public func renamed(to newName: String) -> any AnyColumn {
        TypedColumn<T>(name: newName, values: values)
    }

    public func sortedIndices(ascending: Bool) -> [Int] {
        var indices = Array(0..<values.count)
        let vals = values

        // Specialize common Comparable element types without constraining SupportedType
        // (Bool is Hashable but not Comparable).
        if T.self == Double.self {
            let doubles = vals as! [Double?]
            sortIndices(&indices, ascending: ascending) { doubles[$0] }
            return indices
        }
        if T.self == Float.self {
            let floats = vals as! [Float?]
            sortIndices(&indices, ascending: ascending) { floats[$0] }
            return indices
        }
        if T.self == Int64.self {
            let ints = vals as! [Int64?]
            sortIndices(&indices, ascending: ascending) { ints[$0] }
            return indices
        }
        if T.self == Int32.self {
            let ints = vals as! [Int32?]
            sortIndices(&indices, ascending: ascending) { ints[$0] }
            return indices
        }
        if T.self == String.self {
            let strings = vals as! [String?]
            sortIndices(&indices, ascending: ascending) { strings[$0] }
            return indices
        }
        if T.self == Date.self {
            let dates = vals as! [Date?]
            sortIndices(&indices, ascending: ascending) { dates[$0] }
            return indices
        }
        if T.self == Bool.self {
            let bools = vals as! [Bool?]
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
    public func map<U: SupportedType>(_ transform: (T?) -> U?) -> TypedColumn<U> {
        TypedColumn<U>(name: name, values: values.map(transform))
    }

    /// Applies transform only to non-null elements; nil inputs are passed through as nil.
    public func compactMap<U: SupportedType>(_ transform: (T) -> U?) -> TypedColumn<U> {
        TypedColumn<U>(name: name, values: values.map { $0.flatMap(transform) })
    }
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
    public func dropNulls() -> TypedColumn<T> {
        TypedColumn<T>(name: name, values: values.compactMap { $0 }.map { Optional($0) })
    }

    /// Returns a new column where null values are replaced by `value`.
    public func fillNull(with value: T) -> TypedColumn<T> {
        TypedColumn<T>(name: name, values: values.map { $0 ?? value })
    }

    /// Returns non-null values as a plain array.
    public var nonNullValues: [T] { values.compactMap { $0 } }
}

/// Nulls-last index sort over optional Comparable keys.
private func sortIndices<C: Comparable>(_ indices: inout [Int], ascending: Bool, key: (Int) -> C?) {
    if ascending {
        indices.sort { i, j in
            switch (key(i), key(j)) {
            case (nil, nil): return false
            case (nil, _):   return false
            case (_, nil):   return true
            case let (l?, r?): return l < r
            }
        }
    } else {
        indices.sort { i, j in
            switch (key(i), key(j)) {
            case (nil, nil): return false
            case (nil, _):   return false
            case (_, nil):   return true
            case let (l?, r?): return l > r
            }
        }
    }
}

import Foundation
import Accelerate

// MARK: – Double column filter & vDSP vectorised reductions fast path

extension TypedColumn where T == Double {
    /// Computes sample mean using Accelerate vDSP.
    public func mean() -> Double {
        let nonNulls = nonNullValues
        guard !nonNulls.isEmpty else { return 0.0 }
        var result = 0.0
        vDSP_meanvD(nonNulls, 1, &result, vDSP_Length(nonNulls.count))
        return result
    }

    /// Computes sample variance using two-pass vDSP operations with Bessel's correction.
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
    public func stdDev() -> Double {
        sqrt(variance())
    }

    /// vDSP-accelerated indexed gather for Double columns.
    public func vGather(at indices: [Int]) -> TypedColumn<Double> {
        let n = indices.count
        guard n > 0 else { return TypedColumn<Double>(name: name, values: []) }

        var result = [Double?](repeating: nil, count: n)
        let vals = values
        for i in 0..<n {
            result[i] = vals[indices[i]]
        }
        return TypedColumn<Double>(name: name, values: result)
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

// MARK: – Filter Indices Helpers (Bitmap-Free Fast Paths)

private func filterIndicesDouble(values: [Double?], condition: FilterCondition) -> [Int]? {
    var res = [Int]()
    res.reserveCapacity(values.count / 2)
    switch condition {
    case .isNull:
        for (i, v) in values.enumerated() where v == nil { res.append(i) }
    case .isNotNull:
        for (i, v) in values.enumerated() where v != nil { res.append(i) }
    case .greaterThan(let rhs):
        guard let thr = toDouble(rhs) else { return nil }
        for (i, v) in values.enumerated() { if let x = v, x > thr { res.append(i) } }
    case .lessThan(let rhs):
        guard let thr = toDouble(rhs) else { return nil }
        for (i, v) in values.enumerated() { if let x = v, x < thr { res.append(i) } }
    case .greaterThanOrEqual(let rhs):
        guard let thr = toDouble(rhs) else { return nil }
        for (i, v) in values.enumerated() { if let x = v, x >= thr { res.append(i) } }
    case .lessThanOrEqual(let rhs):
        guard let thr = toDouble(rhs) else { return nil }
        for (i, v) in values.enumerated() { if let x = v, x <= thr { res.append(i) } }
    case .equals(let rhs):
        guard let thr = toDouble(rhs) else { return nil }
        for (i, v) in values.enumerated() { if let x = v, x == thr { res.append(i) } }
    case .notEquals(let rhs):
        guard let thr = toDouble(rhs) else { return nil }
        for (i, v) in values.enumerated() { if let x = v, x != thr { res.append(i) } }
    case .contains:
        return nil
    }
    return res
}

private func filterIndicesInt64(values: [Int64?], condition: FilterCondition) -> [Int]? {
    var res = [Int]()
    res.reserveCapacity(values.count / 2)
    switch condition {
    case .isNull:
        for (i, v) in values.enumerated() where v == nil { res.append(i) }
    case .isNotNull:
        for (i, v) in values.enumerated() where v != nil { res.append(i) }
    case .greaterThan(let rhs):
        guard let thr = toInt64(rhs) else { return nil }
        for (i, v) in values.enumerated() { if let x = v, x > thr { res.append(i) } }
    case .lessThan(let rhs):
        guard let thr = toInt64(rhs) else { return nil }
        for (i, v) in values.enumerated() { if let x = v, x < thr { res.append(i) } }
    case .greaterThanOrEqual(let rhs):
        guard let thr = toInt64(rhs) else { return nil }
        for (i, v) in values.enumerated() { if let x = v, x >= thr { res.append(i) } }
    case .lessThanOrEqual(let rhs):
        guard let thr = toInt64(rhs) else { return nil }
        for (i, v) in values.enumerated() { if let x = v, x <= thr { res.append(i) } }
    case .equals(let rhs):
        guard let thr = toInt64(rhs) else { return nil }
        for (i, v) in values.enumerated() { if let x = v, x == thr { res.append(i) } }
    case .notEquals(let rhs):
        guard let thr = toInt64(rhs) else { return nil }
        for (i, v) in values.enumerated() { if let x = v, x != thr { res.append(i) } }
    case .contains:
        return nil
    }
    return res
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
    public static func == (lhs: TypedColumn<T>, rhs: TypedColumn<T>) -> Bool {
        lhs.name == rhs.name && lhs.values == rhs.values
    }
}

