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
    /// The column d type.
    public static var columnDType: ColumnDType { .int32 }
    /// Parse.
    /// - Returns: A `Int32?` result.
    public static func parse(from string: String) -> Int32? { Int32(string.trimmingCharacters(in: .whitespaces)) }
    /// The double value.
    public var doubleValue: Double? { Double(self) }
}

extension Int: SupportedType {
    /// The column d type.
    public static var columnDType: ColumnDType { .int64 }
    /// Parse.
    /// - Returns: A `Int?` result.
    public static func parse(from string: String) -> Int? { Int(string.trimmingCharacters(in: .whitespaces)) }
    /// The double value.
    public var doubleValue: Double? { Double(self) }
}

extension Int64: SupportedType {
    /// The column d type.
    public static var columnDType: ColumnDType { .int64 }
    /// Parse.
    /// - Returns: A `Int64?` result.
    public static func parse(from string: String) -> Int64? { Int64(string.trimmingCharacters(in: .whitespaces)) }
    /// The double value.
    public var doubleValue: Double? { Double(self) }
}

extension Float: SupportedType {
    /// The column d type.
    public static var columnDType: ColumnDType { .float32 }
    /// Parse.
    /// - Returns: A `Float?` result.
    public static func parse(from string: String) -> Float? { Float(string.trimmingCharacters(in: .whitespaces)) }
    /// The double value.
    public var doubleValue: Double? { Double(self) }
}

extension Double: SupportedType {
    /// The column d type.
    public static var columnDType: ColumnDType { .float64 }
    /// Parse.
    /// - Returns: A `Double?` result.
    public static func parse(from string: String) -> Double? { Double(string.trimmingCharacters(in: .whitespaces)) }
    /// The double value.
    public var doubleValue: Double? { self }
}

extension Bool: SupportedType {
    /// The column d type.
    public static var columnDType: ColumnDType { .boolean }
    /// Parse.
    /// - Returns: A `Bool?` result.
    public static func parse(from string: String) -> Bool? {
        switch string.trimmingCharacters(in: .whitespaces).lowercased() {
        case "true", "yes", "1": return true
        case "false", "no", "0": return false
        default: return nil
        }
    }
    /// The double value.
    public var doubleValue: Double? { nil }
}

extension String: SupportedType {
    /// The column d type.
    public static var columnDType: ColumnDType { .utf8 }
    /// Parse.
    /// - Returns: A `String?` result.
    public static func parse(from string: String) -> String? { string }
    /// The double value.
    public var doubleValue: Double? { Double(self) }
}

extension Date: SupportedType {
    /// The column d type.
    public static var columnDType: ColumnDType { .date32 }
    /// Parse.
    /// - Returns: A `Date?` result.
    public static func parse(from string: String) -> Date? {
        let s = string.trimmingCharacters(in: .whitespaces)
        // ISO 8601 date only: YYYY-MM-DD
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withFullDate, .withDashSeparatorInDate]
        return formatter.date(from: s)
    }
    /// The double value.
    public var doubleValue: Double? { nil }
}
