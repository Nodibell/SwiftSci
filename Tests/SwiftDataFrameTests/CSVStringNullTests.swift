import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV String null membership")
struct CSVStringNullTests {
    private var tokenSets: [Set<String>] {
        let mediumValues: [String] = (0..<17).map { (index: Int) -> String in "missing-\(index)" }
        let largeValues: [String] = (0..<2_048).map { (index: Int) -> String in "missing-\(index)" }
        let medium = Set<String>(mediumValues)
        let large = Set<String>(largeValues)
        return [[], ["NA"], ["", "NA"], ["é", "K"], ["é", "K", "NA"], CSVReadOptions.default.nullValues, medium, large]
    }

    @Test("String decoding retains Set equality across null-set sizes")
    func decodedMembership() {
        let strings: [String] = ["", " \tNA\r\n", "na", "é", "e\u{301}", "K", "K", "missing-16", "missing-2047", "\"a\"\"b\""]
        var inputs: [[UInt8]] = strings.map { (value: String) -> [UInt8] in Array(value.utf8) }
        inputs.append([0xff])
        inputs.append([34, 0xff, 34])
        for tokens in tokenSets + [["a\"b", "�"]] {
            let decoder = CSVStringDecoder(nullValues: tokens)
            for bytes in inputs {
                bytes.withUnsafeBufferPointer { buffer in
                    let field = CSVFieldOffset(startOffset: 0, length: bytes.count, escapedQuotesPresent: bytes.contains(34))
                    let text = VectorizedByteParsers.parseString(buffer: buffer, offset: field)
                    let isUnquotedNull = !VectorizedByteParsers.isQuoted(buffer: buffer, offset: field) && tokens.contains(text)
                    switch decoder.decode(buffer: buffer, offset: field) {
                    case .missing: #expect(isUnquotedNull)
                    case .value(let value):
                        #expect(!isUnquotedNull)
                        #expect(value == text)
                    case .invalid: Issue.record("String conversion unexpectedly rejected a field")
                    }
                }
            }
        }
    }

    @Test("Eager and streamed String columns distinguish unquoted nulls from quoted literals")
    func publicReads() async throws {
        let values: [String] = ["value", "NA", "", "K", "K", "é", "e\u{301}", "a\"b", "汉字", "NaN", "missing-16", "missing-2047"]
        let quotedLines: [String] = values.enumerated().map { pair -> String in
            let index: Int = pair.offset
            let value: String = pair.element
            let escaped = value.replacingOccurrences(of: "\"", with: "\"\"")
            return "\(index),\" \t\(escaped) \r\n\"\n"
        }
        let unquotedLines: [String] = values.enumerated().map { pair -> String in
            let index: Int = pair.offset
            let value: String = pair.element
            if value.contains("\"") {
                let escaped = value.replacingOccurrences(of: "\"", with: "\"\"")
                return "\(index),\"\(escaped)\"\n"
            }
            return value.isEmpty ? "\(index),\n" : "\(index), \t\(value) \n"
        }
        let quotedURL = FileManager.default.temporaryDirectory.appendingPathComponent("csv-string-quoted-\(UUID().uuidString).csv")
        let unquotedURL = FileManager.default.temporaryDirectory.appendingPathComponent("csv-string-nulls-\(UUID().uuidString).csv")
        defer {
            try? FileManager.default.removeItem(at: quotedURL)
            try? FileManager.default.removeItem(at: unquotedURL)
        }
        try Data(("id,text\n" + quotedLines.joined()).utf8).write(to: quotedURL)
        try Data(("id,text\n" + unquotedLines.joined()).utf8).write(to: unquotedURL)
        for tokens in tokenSets {
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
                var actualQuoted: [String?] = []
                for try await chunk in DataFrame.readCSVStream(contentsOf: quotedURL, chunkSize: 3, options: options) {
                    let column = try #require(chunk[column: "text", as: String.self])
                    actualQuoted.append(contentsOf: column.values)
                }
                #expect(actualQuoted == expectedQuoted)

                let unquotedFrame = try await DataFrame(csv: unquotedURL, options: options)
                #expect(unquotedFrame[column: "text", as: String.self]?.values == expectedUnquoted)
                var actualUnquoted: [String?] = []
                for try await chunk in DataFrame.readCSVStream(contentsOf: unquotedURL, chunkSize: 3, options: options) {
                    let column = try #require(chunk[column: "text", as: String.self])
                    actualUnquoted.append(contentsOf: column.values)
                }
                #expect(actualUnquoted == expectedUnquoted)
            }
        }
    }
}
