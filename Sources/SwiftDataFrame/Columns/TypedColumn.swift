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
        let allIndices = Array(0..<values.count)
        return allIndices.sorted(by: { i, j in
            switch (values[i], values[j]) {
            case (nil, nil): return false
            case (nil, _):   return false   // nulls last
            case (_, nil):   return true    // nulls last
            case let (l?, r?):
                // Numeric fast path
                if let ld = l.doubleValue, let rd = r.doubleValue {
                    return ascending ? ld < rd : rd < ld
                }
                // Lexicographic fallback (String, Date, Bool)
                let ls = "\(l)", rs = "\(r)"
                return ascending ? ls < rs : rs < ls
            }
        })
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

// MARK: – Double column filter fast path

extension TypedColumn where T == Double {
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

// MARK: – Equatable (for testing)
extension TypedColumn: Equatable where T: Equatable {
    public static func == (lhs: TypedColumn<T>, rhs: TypedColumn<T>) -> Bool {
        lhs.name == rhs.name && lhs.values == rhs.values
    }
}
