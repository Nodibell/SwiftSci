import Foundation

/// Inlined byte-level ASCII parsers for zero-allocation numeric and string extraction.
public enum VectorizedByteParsers {

    @usableFromInline
    internal static let exactDecimalPowers: [Double] = [
        1, 10, 100, 1_000, 10_000, 100_000, 1_000_000,
        1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
        1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22
    ]

    /// Parses a decimal `Double` from CSV bytes, with one rounding operation for short decimals.
    /// Long significands and scientific notation use Swift's correctly rounded conversion.
    @inlinable
    public static func parseDouble(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> Double? {
        var start = offset.startOffset
        var end = start + offset.length
        if end - start >= 2 && buffer[start] == 34 && buffer[end - 1] == 34 {
            start += 1
            end -= 1
        }
        while start < end && (buffer[start] == 32 || buffer[start] == 9) {
            start += 1
        }
        while start < end && (buffer[end - 1] == 32 || buffer[end - 1] == 9 || buffer[end - 1] == 13 || buffer[end - 1] == 10) {
            end -= 1
        }
        guard start < end else { return nil }

        var index = start
        let negative = buffer[index] == 45
        if negative || buffer[index] == 43 { index += 1 }

        // Fifteen decimal digits fit exactly in binary64 and cannot overflow UInt64.
        // The length check removes overflow and decimal-state branches from each digit.
        guard end - index <= 15 else {
            return parseLongDouble(buffer: buffer, start: start, end: end)
        }
        let digitsStart = index
        var significand: UInt64 = 0
        while index < end {
            let digit = buffer[index] &- 48
            guard digit <= 9 else { break }
            significand = significand &* 10 &+ UInt64(digit)
            index += 1
        }
        if index == end {
            guard index > digitsStart else { return nil }
            let value = Double(significand)
            return negative ? -value : value
        }
        if buffer[index] == 46 {
            let integerDigits = index - digitsStart
            index += 1
            let fractionStart = index
            while index < end {
                let digit = buffer[index] &- 48
                guard digit <= 9 else { break }
                significand = significand &* 10 &+ UInt64(digit)
                index += 1
            }
            if index == end {
                let fractionalDigits = index - fractionStart
                guard integerDigits > 0 || fractionalDigits > 0 else { return nil }
                // Both operands are exact binary64 values, so division rounds the complete decimal once.
                let value = Double(significand) / exactDecimalPowers[fractionalDigits]
                return negative ? -value : value
            }
        }
        return parseLongDouble(buffer: buffer, start: start, end: end)
    }

    @usableFromInline
    @inline(never)
    internal static func parseLongDouble(buffer: UnsafeBufferPointer<UInt8>, start: Int, end: Int) -> Double? {
        var index = start
        if index < end && (buffer[index] == 43 || buffer[index] == 45) { index += 1 }
        let integerStart = index
        while index < end && buffer[index] >= 48 && buffer[index] <= 57 { index += 1 }
        let integerDigits = index - integerStart
        var fractionalDigits = 0
        if index < end && buffer[index] == 46 {
            index += 1
            let fractionStart = index
            while index < end && buffer[index] >= 48 && buffer[index] <= 57 { index += 1 }
            fractionalDigits = index - fractionStart
        }
        guard integerDigits > 0 || fractionalDigits > 0 else { return nil }
        if index < end && (buffer[index] == 101 || buffer[index] == 69) {
            index += 1
            if index < end && (buffer[index] == 43 || buffer[index] == 45) { index += 1 }
            let exponentStart = index
            while index < end && buffer[index] >= 48 && buffer[index] <= 57 { index += 1 }
            guard index > exponentStart else { return nil }
        }
        guard index == end else { return nil }
        return Double(String(decoding: buffer[start..<end], as: UTF8.self))
    }

    /// Parses an `Int` directly from ASCII bytes in an un-copied buffer.
    /// - Parameters:
    ///   - buffer: Underlying byte buffer or contiguous memory storage.
    ///   - offset: Byte or element offset within the buffer.
    /// - Returns: Calculated integer value, or `nil` if undefined.
    @inlinable
    public static func parseInt(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> Int? {
        var start = offset.startOffset
        var length = offset.length

        if length >= 2 && buffer[start] == 34 && buffer[start + length - 1] == 34 {
            start += 1
            length -= 2
        }

        if length <= 0 { return nil }

        var value = 0
        var sign = 1
        var idx = start
        let endIdx = start + length

        while idx < endIdx && (buffer[idx] == 32 || buffer[idx] == 9) {
            idx += 1
        }

        if idx < endIdx {
            if buffer[idx] == 45 {
                sign = -1
                idx += 1
            } else if buffer[idx] == 43 {
                idx += 1
            }
        }

        var digitsFound = false
        while idx < endIdx {
            let byte = buffer[idx]
            if byte >= 48 && byte <= 57 {
                value = value * 10 + Int(byte - 48)
                digitsFound = true
                idx += 1
            } else {
                break
            }
        }

        while idx < endIdx && (buffer[idx] == 32 || buffer[idx] == 9 || buffer[idx] == 13 || buffer[idx] == 10) {
            idx += 1
        }
        guard digitsFound && idx == endIdx else { return nil }
        return value * sign
    }

    /// Constructs a `String` from raw buffer coordinates, unescaping double quote sequences (`""`).
    /// - Parameters:
    ///   - buffer: Underlying byte buffer or contiguous memory storage.
    ///   - offset: Byte or element offset within the buffer.
    /// - Returns: Generated or formatted text string.
    public static func parseString(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> String {
        var start = offset.startOffset
        var length = offset.length

        if length >= 2 && buffer[start] == 34 && buffer[start + length - 1] == 34 {
            start += 1
            length -= 2
        }

        // Trim leading and trailing whitespace / control characters at byte level
        while length > 0 && (buffer[start] == 32 || buffer[start] == 9 || buffer[start] == 13 || buffer[start] == 10) {
            start += 1
            length -= 1
        }
        while length > 0 && (buffer[start + length - 1] == 32 || buffer[start + length - 1] == 9 || buffer[start + length - 1] == 13 || buffer[start + length - 1] == 10) {
            length -= 1
        }

        if length <= 0 { return "" }

        if offset.escapedQuotesPresent {
            var rawBytes = [UInt8]()
            rawBytes.reserveCapacity(length)
            var i = 0
            while i < length {
                let byte = buffer[start + i]
                if byte == 34 && i + 1 < length && buffer[start + i + 1] == 34 {
                    rawBytes.append(34)
                    i += 2
                } else {
                    rawBytes.append(byte)
                    i += 1
                }
            }
            return String(bytes: rawBytes, encoding: .utf8) ?? ""
        } else {
            guard let basePtr = buffer.baseAddress?.advanced(by: start) else { return "" }
            return String(decoding: UnsafeBufferPointer(start: basePtr, count: length), as: UTF8.self)
        }
    }
}
