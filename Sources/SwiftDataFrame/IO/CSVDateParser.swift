import Foundation

/// Prepared conversion for one date column. Each column worker owns its parser.
internal final class CSVDateParser {
    private let format: String?
    private let numeric: CSVNumericDateFormat?
    private lazy var formatter: DateFormatter? = format.map { pattern in
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = pattern
        formatter.isLenient = false
        return formatter
    }

    init(format: String?) {
        self.format = format
        numeric = format.flatMap(CSVNumericDateFormat.init)
    }

    func parse(_ text: String) -> Date? {
        guard format != nil else { return Date.parse(from: text) }
        let cleaned = Date.cleanDateString(text)
        if let numeric,
           let date = cleaned.utf8.withContiguousStorageIfAvailable({
               numeric.parse(buffer: $0, range: $0.indices)
           }) ?? nil {
            return date
        }
        return formatter?.date(from: cleaned)
    }

    func parse(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> Date? {
        if let numeric, !offset.escapedQuotesPresent {
            var start = offset.startOffset
            var end = start + offset.length
            if end - start >= 2, buffer[start] == 34, buffer[end - 1] == 34 {
                start += 1
                end -= 1
            }
            while start < end, Self.isWhitespace(buffer[start]) { start += 1 }
            while start < end, Self.isWhitespace(buffer[end - 1]) { end -= 1 }
            if let date = numeric.parse(buffer: buffer, range: start..<end) { return date }
        }
        let text = VectorizedByteParsers.parseString(buffer: buffer, offset: offset)
        guard format != nil else { return Date.parse(from: text) }
        return formatter?.date(from: Date.cleanDateString(text))
    }

    private static func isWhitespace(_ byte: UInt8) -> Bool {
        byte == 32 || byte == 9 || byte == 13 || byte == 10
    }
}

/// A conservative numeric subset of Foundation date patterns, with Gregorian UTC output.
internal struct CSVNumericDateFormat {
    private enum Component: UInt8 {
        case year = 121, month = 77, day = 100
    }
    private let components: [Component]
    private let separators: [UInt8]
    private static let monthStarts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

    init?(_ pattern: String) {
        let bytes = Array(pattern.utf8)
        var components = [Component]()
        var separators = [UInt8]()
        var index = 0
        while index < bytes.count {
            guard let component = Component(rawValue: bytes[index]),
                  !components.contains(component) else { return nil }
            let start = index
            while index < bytes.count, bytes[index] == component.rawValue { index += 1 }
            let width = index - start
            guard component == .year ? width == 4 : (1...2).contains(width) else { return nil }
            components.append(component)
            if index < bytes.count {
                let separator = bytes[index]
                guard separator == 45 || separator == 47 || separator == 46 else { return nil }
                separators.append(separator)
                index += 1
            }
        }
        guard components.count == 3, separators.count == 2 else { return nil }
        self.components = components
        self.separators = separators
    }

    func parse(buffer: UnsafeBufferPointer<UInt8>, range: Range<Int>) -> Date? {
        var index = range.lowerBound
        var year = 0
        var month = 0
        var day = 0
        for field in 0..<3 {
            let component = components[field]
            let start = index
            var value = 0
            let maxDigits = component == .year ? 4 : 2
            while index < range.upperBound, index - start < maxDigits {
                let byte = buffer[index]
                guard byte >= 48, byte <= 57 else { break }
                value = value * 10 + Int(byte - 48)
                index += 1
            }
            let digits = index - start
            guard component == .year ? digits == 4 : digits > 0 else { return nil }
            switch component {
            case .year: year = value
            case .month: month = value
            case .day: day = value
            }
            if field < 2 {
                guard index < range.upperBound, buffer[index] == separators[field] else { return nil }
                index += 1
            }
        }
        // Earlier years use Foundation's calendar behavior, including the Gregorian cutover.
        guard index == range.upperBound, year >= 1600, year <= 9999,
              month >= 1, month <= 12, day >= 1 else { return nil }
        let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
        let daysInMonth: Int
        switch month {
        case 2: daysInMonth = leap ? 29 : 28
        case 4, 6, 9, 11: daysInMonth = 30
        default: daysInMonth = 31
        }
        guard day <= daysInMonth else { return nil }
        let priorYear = year - 1
        let priorDays = 365 * priorYear + priorYear / 4 - priorYear / 100 + priorYear / 400
        let leapDay = month > 2 && leap ? 1 : 0
        let days = priorDays + Self.monthStarts[month - 1] + leapDay + day - 1 - 719_162
        return Date(timeIntervalSince1970: Double(days) * 86_400)
    }
}
