import Foundation

/// Inlined byte-level ASCII parsers for zero-allocation numeric and string extraction.
public enum VectorizedByteParsers {

    /// Parses a `Double` directly from ASCII bytes in an un-copied buffer.
    @inlinable
    public static func parseDouble(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> Double? {
        var start = offset.startOffset
        var length = offset.length

        if length >= 2 && buffer[start] == 34 && buffer[start + length - 1] == 34 {
            start += 1
            length -= 2
        }

        if length <= 0 { return nil }

        var value = 0.0
        var sign = 1.0
        var idx = start
        let endIdx = start + length

        // Skip leading whitespace / tabs
        while idx < endIdx && (buffer[idx] == 32 || buffer[idx] == 9) {
            idx += 1
        }

        if idx < endIdx {
            if buffer[idx] == 45 { // ASCII '-'
                sign = -1.0
                idx += 1
            } else if buffer[idx] == 43 { // ASCII '+'
                idx += 1
            }
        }

        var digitsFound = false
        while idx < endIdx {
            let byte = buffer[idx]
            if byte >= 48 && byte <= 57 { // ASCII '0'...'9'
                value = value * 10.0 + Double(byte - 48)
                digitsFound = true
                idx += 1
            } else {
                break
            }
        }

        if idx < endIdx && buffer[idx] == 46 { // ASCII '.'
            idx += 1
            var fractionAcc = 0.0
            var divisor = 10.0
            while idx < endIdx {
                let byte = buffer[idx]
                if byte >= 48 && byte <= 57 {
                    fractionAcc += Double(byte - 48) / divisor
                    divisor *= 10.0
                    digitsFound = true
                    idx += 1
                } else {
                    break
                }
            }
            value += fractionAcc
        }

        return digitsFound ? (value * sign) : nil
    }

    /// Parses an `Int` directly from ASCII bytes in an un-copied buffer.
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

        return digitsFound ? (value * sign) : nil
    }

    /// Constructs a `String` from raw buffer coordinates, unescaping double quote sequences (`""`).
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
