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
                    switch decoder.decode(buffer: buffer, offset: field) {
                    case .missing: #expect(tokens.contains(text))
                    case .value(let value):
                        #expect(!tokens.contains(text))
                        #expect(value == text)
                    case .invalid: Issue.record("String conversion unexpectedly rejected a field")
                    }
                }
            }
        }
    }

    @Test("Eager and streamed String columns preserve configured nulls")
    func publicReads() async throws {
        let values: [String] = ["value", "NA", "", "K", "K", "é", "e\u{301}", "a\"b", "汉字", "NaN", "missing-16", "missing-2047"]
        let lines: [String] = values.enumerated().map { pair -> String in
            let index: Int = pair.offset
            let value: String = pair.element
            let escaped = value.replacingOccurrences(of: "\"", with: "\"\"")
            return "\(index),\" \t\(escaped) \r\n\"\n"
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("csv-string-nulls-\(UUID().uuidString).csv")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(("id,text\n" + lines.joined()).utf8).write(to: url)
        for tokens in tokenSets {
            let expected: [String?] = values.map { (value: String) -> String? in tokens.contains(value) ? nil : value }
            for mode in [CSVColumnConversionMode.parallel, .serial] {
                var options = CSVReadOptions()
                options.nullValues = tokens
                options.columnConversionMode = mode
                options.columnTypeOverrides = ["id": .int64, "text": .utf8]
                let frame = try await DataFrame(csv: url, options: options)
                #expect(frame[column: "text", as: String.self]?.values == expected)
                var actual: [String?] = []
                for try await chunk in DataFrame.readCSVStream(contentsOf: url, chunkSize: 3, options: options) {
                    let column = try #require(chunk[column: "text", as: String.self])
                    actual.append(contentsOf: column.values)
                }
                #expect(actual == expected)
            }
        }
    }
}
