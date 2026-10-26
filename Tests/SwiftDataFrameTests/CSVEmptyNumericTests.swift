import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("Empty numeric CSV fields")
struct CSVEmptyNumericTests {
    @Test("Numeric inference distinguishes normalized empties from missing fields",
          arguments: ["", "\"\"", " \t", "\" \t\""], [false, true])
    func classification(_ text: String, _ declaredNull: Bool) {
        let bytes = Array(text.utf8)
        let field = CSVFieldOffset(startOffset: 0, length: bytes.count, escapedQuotesPresent: false)
        let nulls = CSVNullMatcher(declaredNull ? [""] : ["?"])
        bytes.withUnsafeBufferPointer { buffer in
            let integer = CSVIntegerDecoder(nulls: nulls)
            let double = CSVDoubleDecoder(nulls: nulls)
            #expect(integer.decodeValue(buffer: buffer, offset: field) == nil)
            #expect(double.decodeValue(buffer: buffer, offset: field) == nil)
            switch integer.decode(buffer: buffer, offset: field) {
            case .missing: #expect(declaredNull)
            case .invalid: #expect(!declaredNull)
            case .value: Issue.record("Empty integer field became a value")
            }
            switch double.decode(buffer: buffer, offset: field) {
            case .missing: #expect(declaredNull)
            case .invalid: #expect(!declaredNull)
            case .value: Issue.record("Empty Double field became a value")
            }
        }
    }

    @Test("Explicit numeric schemas retain nil for empty, quoted-empty and whitespace fields",
          arguments: [CSVDoubleStorage.optional, .compact])
    func explicitSchema(_ storage: CSVDoubleStorage) async throws {
        let csv = "i,x,label\n,,raw\n\"\",\"\",quoted\n ,\t,space\n7,1.25,value\n"
        let url = try write(csv)
        defer { try? FileManager.default.removeItem(at: url) }
        var options = CSVReadOptions()
        options.nullValues = ["?"]
        options.doubleStorage = storage
        options.columnTypeOverrides = ["i": .int64, "x": .float64]
        let frame = try await DataFrame(csv: url, options: options)
        let integers = try #require(frame[column: "i", as: Int64.self])
        let doubles = try #require(frame[column: "x", as: Double.self])
        #expect(integers.values == [nil, nil, nil, 7])
        #expect(doubles.values == [nil, nil, nil, 1.25])
        #expect(integers.nullCount == 3)
        #expect(doubles.nullCount == 3)
    }

    @Test("An undeclared normalized empty after the inference sample preserves text",
          arguments: ["", "\"\"", " \t", "\" \t\""])
    func lateEmpty(_ empty: String) async throws {
        let csv = "i,x,label\n" + String(repeating: "0042,1.25,value\n", count: 1001) + "\(empty),\(empty),empty\n"
        let url = try write(csv)
        defer { try? FileManager.default.removeItem(at: url) }
        var options = CSVReadOptions()
        options.nullValues = ["?"]
        let frame = try await DataFrame(csv: url, options: options)
        let integers = try #require(frame[column: "i", as: String.self])
        let doubles = try #require(frame[column: "x", as: String.self])
        #expect(integers.count == 1002)
        #expect(integers.values.first! == "0042")
        #expect(doubles.values.first! == "1.25")
        #expect(integers.values.last! == "")
        #expect(doubles.values.last! == "")
        #expect(integers.nullCount == 0)
        #expect(doubles.nullCount == 0)
    }

    private func write(_ csv: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("csv-empty-\(UUID().uuidString).csv")
        try Data(csv.utf8).write(to: url)
        return url
    }
}
