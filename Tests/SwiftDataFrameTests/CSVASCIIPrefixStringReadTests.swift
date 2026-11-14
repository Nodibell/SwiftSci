import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV prepared String read parity")
struct CSVASCIIPrefixStringReadTests {
    @Test("Eager and streamed chunks agree across decoder-selection boundaries")
    func publicReads() async throws {
        let fixture: [String] = ["value", "NA", "", "K", "K", "é", "e\u{301}", "a\"b", "汉字"]
        let values: [String] = (0..<64).map { (index: Int) -> String in fixture[index % fixture.count] }
        let quotedLines: [String] = values.enumerated().map { pair -> String in
            let escaped: String = pair.element.replacingOccurrences(of: "\"", with: "\"\"")
            return "\(pair.offset),\" \t\(escaped) \r\n\"\n"
        }
        let unquotedLines: [String] = values.enumerated().map { pair -> String in
            if pair.element.contains("\"") {
                let escaped = pair.element.replacingOccurrences(of: "\"", with: "\"\"")
                return "\(pair.offset),\"\(escaped)\"\n"
            }
            return pair.element.isEmpty ? "\(pair.offset),\n" : "\(pair.offset), \t\(pair.element) \n"
        }
        let quotedURL = FileManager.default.temporaryDirectory.appendingPathComponent("csv-prepared-quoted-\(UUID().uuidString).csv")
        let unquotedURL = FileManager.default.temporaryDirectory.appendingPathComponent("csv-prepared-nulls-\(UUID().uuidString).csv")
        defer {
            try? FileManager.default.removeItem(at: quotedURL)
            try? FileManager.default.removeItem(at: unquotedURL)
        }
        try Data(("id,text\n" + quotedLines.joined()).utf8).write(to: quotedURL)
        try Data(("id,text\n" + unquotedLines.joined()).utf8).write(to: unquotedURL)
        let tokens: Set<String> = ["", "K", "NA"]
        let expectedQuoted: [String?] = values.map(Optional.some)
        let expectedUnquoted: [String?] = values.map { (value: String) -> String? in
            (!value.contains("\"") && tokens.contains(value)) ? nil : value
        }
        for mode in [CSVColumnConversionMode.parallel, .serial] {
            var options = CSVReadOptions()
            options.nullValues = tokens
            options.columnConversionMode = mode
            options.columnTypeOverrides = ["id": .int64, "text": .utf8]

            let quotedFrame = try await DataFrame(csv: quotedURL, options: options)
            #expect(quotedFrame[column: "text", as: String.self]?.values == expectedQuoted)
            let unquotedFrame = try await DataFrame(csv: unquotedURL, options: options)
            #expect(unquotedFrame[column: "text", as: String.self]?.values == expectedUnquoted)

            for size in [3, 63] {
                var actualQuoted: [String?] = []
                for try await chunk in DataFrame.readCSVStream(contentsOf: quotedURL, chunkSize: size, options: options) {
                    let column = try #require(chunk[column: "text", as: String.self])
                    actualQuoted.append(contentsOf: column.values)
                }
                #expect(actualQuoted == expectedQuoted)

                var actualUnquoted: [String?] = []
                for try await chunk in DataFrame.readCSVStream(contentsOf: unquotedURL, chunkSize: size, options: options) {
                    let column = try #require(chunk[column: "text", as: String.self])
                    actualUnquoted.append(contentsOf: column.values)
                }
                #expect(actualUnquoted == expectedUnquoted)
            }
        }
    }
}
