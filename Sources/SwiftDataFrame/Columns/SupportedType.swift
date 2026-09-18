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
    /// - Parameters:
    ///   - string: Input string content.
    public static func parse(from string: String) -> Int32? { Int32(string.trimmingCharacters(in: .whitespaces)) }
    /// The double value.
    public var doubleValue: Double? { Double(self) }
}

extension Int: SupportedType {
    /// The column d type.
    public static var columnDType: ColumnDType { .int64 }
    /// Parse.
    /// - Returns: A `Int?` result.
    /// - Parameters:
    ///   - string: Input string content.
    public static func parse(from string: String) -> Int? { Int(string.trimmingCharacters(in: .whitespaces)) }
    /// The double value.
    public var doubleValue: Double? { Double(self) }
}

extension Int64: SupportedType {
    /// The column d type.
    public static var columnDType: ColumnDType { .int64 }
    /// Parse.
    /// - Returns: A `Int64?` result.
    /// - Parameters:
    ///   - string: Input string content.
    public static func parse(from string: String) -> Int64? { Int64(string.trimmingCharacters(in: .whitespaces)) }
    /// The double value.
    public var doubleValue: Double? { Double(self) }
}

extension Float: SupportedType {
    /// The column d type.
    public static var columnDType: ColumnDType { .float32 }
    /// Parse.
    /// - Returns: A `Float?` result.
    /// - Parameters:
    ///   - string: Input string content.
    public static func parse(from string: String) -> Float? { Float(string.trimmingCharacters(in: .whitespaces)) }
    /// The double value.
    public var doubleValue: Double? { Double(self) }
}

extension Double: SupportedType {
    /// The column d type.
    public static var columnDType: ColumnDType { .float64 }
    /// Parse.
    /// - Returns: A `Double?` result.
    /// - Parameters:
    ///   - string: Input string content.
    public static func parse(from string: String) -> Double? { Double(string.trimmingCharacters(in: .whitespaces)) }
    /// The double value.
    public var doubleValue: Double? { self }
}

extension Bool: SupportedType {
    /// The column d type.
    public static var columnDType: ColumnDType { .boolean }
    /// Parse.
    /// - Returns: A `Bool?` result.
    /// - Parameters:
    ///   - string: Input string content.
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
    /// - Parameters:
    ///   - string: Input string content.
    public static func parse(from string: String) -> String? { string }
    /// The double value.
    public var doubleValue: Double? { Double(self) }
}

extension Date: SupportedType {
    /// The column d type.
    public static var columnDType: ColumnDType { .date32 }
    /// Parse.
    /// - Returns: A `Date?` result.
    /// - Parameters:
    ///   - string: Input string content.
    public static func parse(from string: String) -> Date? {
        var s = string.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("\"") && s.hasSuffix("\"") && s.count >= 2 {
            s.removeFirst()
            s.removeLast()
            s = s.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if s.isEmpty { return nil }

        let isoFormatter = ISO8601DateFormatter()
        isoFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let d = isoFormatter.date(from: s) { return d }

        isoFormatter.formatOptions = [.withInternetDateTime]
        if let d = isoFormatter.date(from: s) { return d }

        isoFormatter.formatOptions = [.withFullDate, .withDashSeparatorInDate]
        if let d = isoFormatter.date(from: s) { return d }

        let formats = [
            "yyyy-MM-dd HH:mm:ss",
            "yyyy-MM-dd HH:mm",
            "yyyy/MM/dd HH:mm:ss",
            "yyyy/MM/dd HH:mm",
            "dd/MM/yyyy HH:mm:ss",
            "dd/MM/yyyy HH:mm",
            "MM/dd/yyyy HH:mm:ss",
            "MM/dd/yyyy HH:mm",
            "yyyy-MM-dd",
            "yyyy/MM/dd",
            "dd/MM/yyyy",
            "MM/dd/yyyy",
            "dd-MM-yyyy"
        ]
        let df = DateFormatter()
        df.locale = Locale(identifier: "en_US_POSIX")
        df.timeZone = TimeZone(secondsFromGMT: 0)
        for fmt in formats {
            df.dateFormat = fmt
            if let d = df.date(from: s) {
                return d
            }
        }
        return nil
    }
    /// The double value.
    public var doubleValue: Double? { nil }
}
