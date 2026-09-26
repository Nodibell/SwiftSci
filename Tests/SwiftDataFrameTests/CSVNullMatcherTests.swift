import Testing
import Foundation
@testable import SwiftDataFrame

@Suite("CSV byte null matching")
struct CSVNullMatcherTests {
    @Test func matchesDecodedStrings() {
        let tokens: [Set<String>] = [CSVReadOptions.default.nullValues, [], ["NA"], ["", "3.5", "-42"], ["K"], ["K"], ["é"], ["e\u{301}"], ["a\"b"], ["�"]]
        var inputs = ["", " ", "\tNA\r\n", "NA", "na", "Na", "123", "3.5", "-42", "\" NA \"", "\"a\"\"b\"", "é", "e\u{301}", "K", "K", "\"\"", " \"NA\" "].map { Array($0.utf8) }
        inputs += [[0xff], [34, 0xff, 34], [34, 0xff, 34, 34, 34]]
        for nulls in tokens {
            let matcher = CSVNullMatcher(nulls)
            for bytes in inputs {
                bytes.withUnsafeBufferPointer { buffer in
                    for escaped in [false, true] {
                        let offset = CSVFieldOffset(startOffset: 0, length: bytes.count, escapedQuotesPresent: escaped)
                        let expected = nulls.contains(VectorizedByteParsers.parseString(buffer: buffer, offset: offset))
                        #expect(matcher.contains(buffer: buffer, offset: offset) == expected)
                    }
                }
            }
        }
    }
    @Test func numericFileColumnsHonorConfiguredTokens() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("csv-nulls-\(UUID().uuidString).csv")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("integer,decimal\n10,1.25\n\" -42 \",\" 3.5 \"\n30,5e1\n".utf8).write(to: url)
        for overrides in [false, true] {
            var options = CSVReadOptions()
            options.nullValues = ["-42", "3.5"]
            if overrides { options.columnTypeOverrides = ["integer": .int64, "decimal": .float64] }
            let frame = try await DataFrame(csv: url, options: options)
            #expect(frame[column: "integer", as: Int64.self]?.values == [10, nil, 30])
            #expect(frame[column: "decimal", as: Double.self]?.values == [1.25, nil, 50])
        }
    }

}
