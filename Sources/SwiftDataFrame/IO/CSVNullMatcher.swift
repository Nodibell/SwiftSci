import Foundation

/// Matches the String decoder's null semantics without decoding ordinary ASCII fields.
internal struct CSVNullMatcher {
    private let nullValues: Set<String>
    private let patterns: [[UInt8]]
    private let requiresString: Bool

    init(_ nullValues: Set<String>) {
        self.nullValues = nullValues
        patterns = nullValues.map { Array($0.utf8) }
        requiresString = patterns.contains { $0.contains { $0 >= 128 } }
    }

    func contains(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> Bool {
        if requiresString || offset.escapedQuotesPresent {
            return nullValues.contains(VectorizedByteParsers.parseString(buffer: buffer, offset: offset))
        }
        var start = offset.startOffset
        var end = start + offset.length
        if end - start >= 2 && buffer[start] == 34 && buffer[end - 1] == 34 {
            start += 1
            end -= 1
        }
        while start < end && isWhitespace(buffer[start]) { start += 1 }
        while start < end && isWhitespace(buffer[end - 1]) { end -= 1 }
        for index in start..<end where buffer[index] >= 128 {
            // Unicode normalization can equate non-ASCII input with an ASCII token.
            return nullValues.contains(VectorizedByteParsers.parseString(buffer: buffer, offset: offset))
        }
        let length = end - start
        for pattern in patterns where pattern.count == length {
            var matches = true
            for index in pattern.indices where buffer[start + index] != pattern[index] {
                matches = false
                break
            }
            if matches { return true }
        }
        return false
    }

    private func isWhitespace(_ byte: UInt8) -> Bool {
        byte == 32 || byte == 9 || byte == 13 || byte == 10
    }
}
