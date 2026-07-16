import Foundation

/// Marker protocol for types that can be stored in a TypedColumn.
/// Conforming types must be Sendable and Hashable.
public protocol SupportedType: Sendable, Hashable {
    /// The corresponding ColumnDType for this Swift type.
    static var columnDType: ColumnDType { get }

    /// Try to parse this type from a raw CSV string.
    static func parse(from string: String) -> Self?

    /// Convert to Double for numeric operations. Returns nil for non-numeric types.
    var doubleValue: Double? { get }
}

// MARK: – Conformances

extension Int32: SupportedType {
    public static var columnDType: ColumnDType { .int32 }
    public static func parse(from string: String) -> Int32? { Int32(string.trimmingCharacters(in: .whitespaces)) }
    public var doubleValue: Double? { Double(self) }
}

extension Int64: SupportedType {
    public static var columnDType: ColumnDType { .int64 }
    public static func parse(from string: String) -> Int64? { Int64(string.trimmingCharacters(in: .whitespaces)) }
    public var doubleValue: Double? { Double(self) }
}

extension Float: SupportedType {
    public static var columnDType: ColumnDType { .float32 }
    public static func parse(from string: String) -> Float? { Float(string.trimmingCharacters(in: .whitespaces)) }
    public var doubleValue: Double? { Double(self) }
}

extension Double: SupportedType {
    public static var columnDType: ColumnDType { .float64 }
    public static func parse(from string: String) -> Double? { Double(string.trimmingCharacters(in: .whitespaces)) }
    public var doubleValue: Double? { self }
}

extension Bool: SupportedType {
    public static var columnDType: ColumnDType { .boolean }
    public static func parse(from string: String) -> Bool? {
        switch string.trimmingCharacters(in: .whitespaces).lowercased() {
        case "true", "yes", "1": return true
        case "false", "no", "0": return false
        default: return nil
        }
    }
    public var doubleValue: Double? { nil }
}

extension String: SupportedType {
    public static var columnDType: ColumnDType { .utf8 }
    public static func parse(from string: String) -> String? { string }
    public var doubleValue: Double? { Double(self) }
}

extension Date: SupportedType {
    public static var columnDType: ColumnDType { .date32 }
    public static func parse(from string: String) -> Date? {
        let s = string.trimmingCharacters(in: .whitespaces)
        // ISO 8601 date only: YYYY-MM-DD
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withFullDate, .withDashSeparatorInDate]
        return formatter.date(from: s)
    }
    public var doubleValue: Double? { nil }
}
