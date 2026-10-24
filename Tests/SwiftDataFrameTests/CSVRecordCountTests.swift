import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV count-only grammar parity")
struct CSVRecordCountTests {
    @Test("Count matches record discovery for quotes and permissive EOF",
          arguments: ["", "a,b", "a,b,", "a,b\n", "\n\n", "\"a\nb\",c\r\nlast,", "\"unterminated,tail", "a,\"b\"\"c\"\n"])
    func records(_ text: String) throws {
        try compare(Array(text.utf8), delimiter: 44, quote: 34)
    }

    @Test("Count shares custom delimiter and quote precedence", arguments: [UInt8(44), 59, 10, 34, 0, 255])
    func generated(_ delimiter: UInt8) throws {
        for quote: UInt8 in [34, 39, 10, 44, 0] {
            var seed: UInt64 = 81_471
            let alphabet: [UInt8] = [65, 10, 13, 34, delimiter, quote]
            for length in 0..<256 {
                var bytes = [UInt8]()
                for _ in 0..<(length % 37) {
                    seed = seed &* 6_364_136_223_846_793_005 &+ 1
                    bytes.append(alphabet[Int(seed >> 32) % alphabet.count])
                }
                try compare(bytes, delimiter: delimiter, quote: quote)
            }
        }
    }

    @Test("Count crosses search windows and preserves row limits", arguments: [65_535, 65_536, 65_537])
    func longField(_ length: Int) throws {
        let text = "\"" + String(repeating: "x", count: length) + "\ninside\",tail\r\nnext,row\n"
        try compare(Array(text.utf8), delimiter: 44, quote: 34)
    }

    @Test("Count preserves trailing fields after skipped delimiters across search windows",
          arguments: ["tail,", "\"tail,", "tail,\"\"", "tail, ", "\"tail,\"", "tail,\"\"\""])
    func trailingField(_ suffix: String) throws {
        let text = String(repeating: "x", count: 65_536) + ",1\n" + suffix
        try compare(Array(text.utf8), delimiter: 44, quote: 34)
    }

    @Test("Record counting checks cancellation")
    func cancellation() async throws {
        let bytes = Array("a,b\n1,2\n".utf8)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try bytes.withUnsafeBufferPointer {
                try SystemsCSVParser().countRecords(buffer: $0)
            }
        }
        do {
            _ = try await task.value
            Issue.record("Cancelled counting completed")
        } catch is CancellationError {} catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    private func compare(_ bytes: [UInt8], delimiter: UInt8, quote: UInt8) throws {
        try bytes.withUnsafeBufferPointer { buffer in
            let parser = SystemsCSVParser(delimiterByte: delimiter, quoteByte: quote)
            let expected = parser.parseIndex(buffer: buffer, minimumParallelBytes: .max).count
            let actual = try parser.countRecords(buffer: buffer)
            #expect(actual == expected)
            for limit in [0, 1, 2, 100] {
                let limited = try parser.countRecords(buffer: buffer, maximumRows: limit)
                #expect(limited == min(expected, limit))
            }
        }
    }
}
