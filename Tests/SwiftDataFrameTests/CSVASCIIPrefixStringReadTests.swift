import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV prepared String read parity")
struct CSVASCIIPrefixStringReadTests {
    @Test("Eager and streamed chunks agree across decoder-selection boundaries")
    func publicReads() async throws {
        let fixture: [String] = ["value", "NA", "", "K", "K", "é", "e\u{301}", "a\"b", "汉字"]
        let values: [String] = (0..<64).map { (index: Int) -> String in fixture[index % fixture.count] }
        let lines: [String] = values.enumerated().map { pair -> String in
            let escaped: String = pair.element.replacingOccurrences(of: "\"", with: "\"\"")
            return "\(pair.offset),\" \t\(escaped) \r\n\"\n"
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("csv-prepared-nulls-\(UUID().uuidString).csv")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(("id,text\n" + lines.joined()).utf8).write(to: url)
        let tokens: Set<String> = ["", "K", "NA"]
        let expected: [String?] = values.map { (value: String) -> String? in tokens.contains(value) ? nil : value }
        for mode in [CSVColumnConversionMode.parallel, .serial] {
            var options = CSVReadOptions()
            options.nullValues = tokens
            options.columnConversionMode = mode
            options.columnTypeOverrides = ["id": .int64, "text": .utf8]
            let frame = try await DataFrame(csv: url, options: options)
            #expect(frame[column: "text", as: String.self]?.values == expected)
            for size in [3, 63] {
                var actual: [String?] = []
                for try await chunk in DataFrame.readCSVStream(contentsOf: url, chunkSize: size, options: options) {
                    let column = try #require(chunk[column: "text", as: String.self])
                    actual.append(contentsOf: column.values)
                }
                #expect(actual == expected)
            }
        }
    }
}
