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

    private static let _posixLocale = Locale(identifier: "en_US_POSIX")
    private static let _gmtTimeZone = TimeZone(secondsFromGMT: 0) ?? .gmt
    private static let _gregorianCalendar = Calendar(identifier: .gregorian)

    private static func createFormatters(_ fmts: [String]) -> [DateFormatter] {
        return fmts.map { fmt in
            let df = DateFormatter()
            df.locale = _posixLocale
            df.timeZone = _gmtTimeZone
            df.dateFormat = fmt
            return df
        }
    }

    // 4-digit year at start (e.g. 2020-01-21, 2020/01/21, ISO formats)
    private static let yearFirst4Formatters: [DateFormatter] = createFormatters([
        "yyyy-MM-dd HH:mm:ss.SSSSSS",
        "yyyy-MM-dd HH:mm:ss.SSS",
        "yyyy-MM-dd HH:mm:ss",
        "yyyy-MM-dd HH:mm",
        "yyyy-MM-dd",
        "yyyy/MM/dd HH:mm:ss.SSSSSS",
        "yyyy/MM/dd HH:mm:ss.SSS",
        "yyyy/MM/dd HH:mm:ss",
        "yyyy/MM/dd HH:mm",
        "yyyy/MM/dd",
        "yyyy.MM.dd HH:mm:ss.SSSSSS",
        "yyyy.MM.dd HH:mm:ss.SSS",
        "yyyy.MM.dd HH:mm:ss",
        "yyyy.MM.dd HH:mm",
        "yyyy.MM.dd",
        "yyyy-MM-dd'T'HH:mm:ss.SSSSSSZ",
        "yyyy-MM-dd'T'HH:mm:ss.SSSZ",
        "yyyy-MM-dd'T'HH:mm:ssZ",
        "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
        "yyyy-MM-dd'T'HH:mm:ss.SSS",
        "yyyy-MM-dd'T'HH:mm:ss",
        "yyyy-MM-dd HH:mm:ss Z",
        "yyyy-MM-dd HH:mm:ss.SSS Z",
        "yyyy-MM"
    ])

    // 4-digit year at end (e.g. 12/30/2020, 31/12/2020)
    private static let yearLast4Formatters: [DateFormatter] = createFormatters([
        "MM/dd/yyyy HH:mm:ss.SSSSSS",
        "MM/dd/yyyy HH:mm:ss.SSS",
        "MM/dd/yyyy HH:mm:ss",
        "MM/dd/yyyy HH:mm",
        "MM/dd/yyyy",
        "dd/MM/yyyy HH:mm:ss.SSSSSS",
        "dd/MM/yyyy HH:mm:ss.SSS",
        "dd/MM/yyyy HH:mm:ss",
        "dd/MM/yyyy HH:mm",
        "dd/MM/yyyy",
        "M/d/yyyy HH:mm:ss",
        "M/d/yyyy HH:mm",
        "M/d/yyyy",
        "d/M/yyyy HH:mm:ss",
        "d/M/yyyy HH:mm",
        "d/M/yyyy",
        "dd-MM-yyyy HH:mm:ss.SSS",
        "dd-MM-yyyy HH:mm:ss",
        "dd-MM-yyyy HH:mm",
        "dd-MM-yyyy",
        "dd.MM.yyyy HH:mm:ss.SSS",
        "dd.MM.yyyy HH:mm:ss",
        "dd.MM.yyyy HH:mm",
        "dd.MM.yyyy"
    ])

    // 2-digit year at end, month first (e.g. 12/30/20, 1/21/20, 1/1/21, 5/31/21)
    private static let yearLast2MonthFirstFormatters: [DateFormatter] = createFormatters([
        "M/d/yy HH:mm:ss",
        "M/d/yy HH:mm",
        "M/d/yy",
        "MM/dd/yy HH:mm:ss",
        "MM/dd/yy HH:mm",
        "MM/dd/yy"
    ])

    // 2-digit year at end, day first (e.g. 30/12/20, 31-12-20)
    private static let yearLast2DayFirstFormatters: [DateFormatter] = createFormatters([
        "d/M/yy HH:mm:ss",
        "d/M/yy HH:mm",
        "d/M/yy",
        "dd/MM/yy HH:mm:ss",
        "dd/MM/yy HH:mm",
        "dd/MM/yy",
        "dd-MM-yy HH:mm:ss",
        "dd-MM-yy HH:mm",
        "dd-MM-yy",
        "dd.MM.yy HH:mm:ss",
        "dd.MM.yy HH:mm",
        "dd.MM.yy"
    ])

    // 2-digit year at start (e.g. 20-12-30, 20/12/30)
    private static let yearFirst2Formatters: [DateFormatter] = createFormatters([
        "yy-MM-dd HH:mm:ss",
        "yy-MM-dd HH:mm",
        "yy-MM-dd",
        "yy/MM/dd HH:mm:ss",
        "yy/MM/dd HH:mm",
        "yy/MM/dd"
    ])

    nonisolated(unsafe) private static let isoFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    nonisolated(unsafe) private static let isoStandardFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f
    }()

    nonisolated(unsafe) private static let isoFullDateFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withFullDate, .withDashSeparatorInDate]
        return f
    }()

    /// Trims whitespace, newlines, and enclosing quotes.
    public static func cleanDateString(_ string: String) -> String {
        var s = string.trimmingCharacters(in: .whitespacesAndNewlines)
        if (s.hasPrefix("\"") && s.hasSuffix("\"")) || (s.hasPrefix("'") && s.hasSuffix("'")) {
            s = String(s.dropFirst().dropLast()).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return s
    }

    /// Try to parse this type from a raw string.
    public static func parse(from string: String) -> Date? {
        let cleaned = cleanDateString(string)
        guard !cleaned.isEmpty && cleaned != "—" && cleaned != "null" && cleaned != "NA" && cleaned != "NaN" else {
            return nil
        }

        // Prevent arbitrary integers/floats (e.g. prices 525000 or IDs 123456) from matching date formats
        if let intVal = Int(cleaned) {
            if cleaned.count == 4 && (1800...2100).contains(intVal) {
                var comp = DateComponents()
                comp.year = intVal
                comp.month = 1
                comp.day = 1
                return Calendar(identifier: .gregorian).date(from: comp)
            }
            return nil
        }
        if Double(cleaned) != nil {
            return nil
        }

        // Try ISO8601 internet datetime first if ISO "T" is present
        if cleaned.contains("T") {
            if let d = isoFormatter.date(from: cleaned) ?? isoStandardFormatter.date(from: cleaned) {
                return normalizeYearIfNeeded(d)
            }
        }

        // Fast path for strict date-only (yyyy-MM-dd)
        if cleaned.count == 10 && !cleaned.contains(" ") && !cleaned.contains(":") {
            if let d = isoFullDateFormatter.date(from: cleaned) {
                return normalizeYearIfNeeded(d)
            }
        }

        // Fast path: select formatters by token lengths
        let datePart = cleaned.split(separator: " ").first.flatMap { $0.split(separator: "T").first }.map(String.init) ?? cleaned
        let sepTokens = datePart.components(separatedBy: CharacterSet(charactersIn: "-/."))

        var formattersToTry: [DateFormatter] = []
        if sepTokens.count == 3 {
            let t0Len = sepTokens[0].count
            let t2Len = sepTokens[2].count

            if t0Len == 4 {
                formattersToTry = yearFirst4Formatters + yearLast4Formatters + yearLast2MonthFirstFormatters + yearLast2DayFirstFormatters + yearFirst2Formatters
            } else if t2Len == 4 {
                formattersToTry = yearLast4Formatters + yearFirst4Formatters + yearLast2MonthFirstFormatters + yearLast2DayFirstFormatters + yearFirst2Formatters
            } else if t2Len == 2 || t2Len == 1 {
                if let v0 = Int(sepTokens[0]), v0 > 12 {
                    formattersToTry = yearLast2DayFirstFormatters + yearLast2MonthFirstFormatters + yearFirst2Formatters + yearLast4Formatters
                } else if let v1 = Int(sepTokens[1]), v1 > 12 {
                    formattersToTry = yearLast2MonthFirstFormatters + yearLast2DayFirstFormatters + yearFirst2Formatters + yearLast4Formatters
                } else {
                    formattersToTry = yearLast2MonthFirstFormatters + yearLast2DayFirstFormatters + yearFirst2Formatters + yearLast4Formatters
                }
            } else if t0Len == 2 {
                formattersToTry = yearFirst2Formatters + yearLast2MonthFirstFormatters + yearLast2DayFirstFormatters + yearLast4Formatters + yearFirst4Formatters
            } else {
                formattersToTry = yearLast2MonthFirstFormatters + yearLast2DayFirstFormatters + yearLast4Formatters + yearFirst4Formatters + yearFirst2Formatters
            }
        } else {
            formattersToTry = yearFirst4Formatters + yearLast4Formatters + yearLast2MonthFirstFormatters + yearLast2DayFirstFormatters + yearFirst2Formatters
        }

        for df in formattersToTry {
            if let d = df.date(from: cleaned) {
                return normalizeYearIfNeeded(d)
            }
        }
        return nil
    }


    /// Checks if a string can be parsed as a Date.
    public static func isDateString(_ string: String) -> Bool {
        parse(from: string) != nil
    }

    /// Checks whether the string representation contains an hour/minute indicator.
    public static func stringHasTime(_ string: String) -> Bool {
        let cleaned = cleanDateString(string)
        if cleaned.contains(":") { return true }
        if cleaned.contains("T") && cleaned.count > 10 { return true }
        if cleaned.count > 10 && cleaned.contains(" ") { return true }
        return false
    }

    /// Normalizes 2-digit years (< 70 -> 2000s, >= 70 -> 1900s).
    private static func normalizeYearIfNeeded(_ date: Date) -> Date {
        let year = _gregorianCalendar.component(.year, from: date)
        if year < 100 {
            let fullYear = year < 70 ? 2000 + year : 1900 + year
            var comp = _gregorianCalendar.dateComponents([.year, .month, .day, .hour, .minute, .second], from: date)
            comp.year = fullYear
            if let adjusted = _gregorianCalendar.date(from: comp) {
                return adjusted
            }
        }
        return date
    }

    /// Checks if this Date has a non-zero time component in GMT/UTC.
    public var hasTimeComponent: Bool {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(secondsFromGMT: 0) ?? .gmt
        let comps = cal.dateComponents([.hour, .minute, .second, .nanosecond], from: self)
        return (comps.hour ?? 0) != 0 || (comps.minute ?? 0) != 0 || (comps.second ?? 0) != 0 || (comps.nanosecond ?? 0) != 0
    }

    /// Formats as "yyyy-MM-dd HH:mm:ss" if time is present, otherwise "yyyy-MM-dd".
    public var formattedDateOrDateTimeString: String {
        let fmt = DateFormatter()
        fmt.locale = Self._posixLocale
        fmt.timeZone = Self._gmtTimeZone
        if hasTimeComponent {
            fmt.dateFormat = "yyyy-MM-dd HH:mm:ss"
        } else {
            fmt.dateFormat = "yyyy-MM-dd"
        }
        return fmt.string(from: self)
    }

    /// Formats strictly as "yyyy-MM-dd".
    public var formattedDateString: String {
        let fmt = DateFormatter()
        fmt.locale = Self._posixLocale
        fmt.timeZone = Self._gmtTimeZone
        fmt.dateFormat = "yyyy-MM-dd"
        return fmt.string(from: self)
    }

    /// Formats strictly as "yyyy-MM-dd HH:mm:ss".
    public var formattedDateTimeString: String {
        let fmt = DateFormatter()
        fmt.locale = Self._posixLocale
        fmt.timeZone = Self._gmtTimeZone
        fmt.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return fmt.string(from: self)
    }

    /// Convert to Double for numeric operations. Returns nil for non-numeric types.
    public var doubleValue: Double? { nil }
}
