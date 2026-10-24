import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV sampled inference validation")
struct CSVInferenceValidationTests {
    @Test("Late fractions promote sampled integers without losing values", arguments: [false, true])
    func lateFraction(_ quoted: Bool) async throws {
        let integer = quoted ? "\"42\"" : "42"
        let fraction = quoted ? "\"0.125\"" : "0.125"
        let csv = "value\n" + String(repeating: integer + "\n", count: 1000) + "NA\n" + fraction + "\n"
        let url = try write(csv)
        defer { try? FileManager.default.removeItem(at: url) }
        let frame = try await DataFrame(csv: url)
        checkFraction(frame)
        var chunks = 0
        for try await chunk in DataFrame.readCSVStream(contentsOf: url, chunkSize: 2000) {
            checkFraction(chunk)
            chunks += 1
        }
        #expect(chunks == 1)
    }

    @Test("Late text falls back to original strings for every sampled type",
          arguments: ["0042", "1.25", "true", "2020-01-02"])
    func lateText(_ prefix: String) async throws {
        let csv = "value\n" + String(repeating: prefix + "\n", count: 1000) + "NA\nsensor offline\n"
        let url = try write(csv)
        defer { try? FileManager.default.removeItem(at: url) }
        let frame = try await DataFrame(csv: url)
        let values = try #require(frame[column: "value", as: String.self])
        #expect(values.count == 1002)
        #expect(values.values.prefix(1000).allSatisfy { $0 == prefix })
        #expect(values.values[1000] == nil)
        #expect(values.values[1001] == "sensor offline")
    }

    @Test("Explicit integer overrides retain their invalid-value null contract")
    func explicitType() async throws {
        let url = try write("value\n42\n0.125\nNA\n")
        defer { try? FileManager.default.removeItem(at: url) }
        var options = CSVReadOptions()
        options.columnTypeOverrides = ["value": .int64]
        let frame = try await DataFrame(csv: url, options: options)
        #expect(frame[column: "value", as: Int64.self]?.values == [42, nil, nil])
    }

    private func checkFraction(_ frame: DataFrame) {
        let values = frame[column: "value", as: Double.self]
        #expect(values?.count == 1002)
        #expect(values?.values.prefix(1000).allSatisfy { $0 == 42 } == true)
        #expect(values?.values[1000] == nil)
        #expect(values?.values[1001] == 0.125)
        #expect(values?.nullCount == 1)
    }

    private func write(_ csv: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("csv-inference-\(UUID().uuidString).csv")
        try Data(csv.utf8).write(to: url)
        return url
    }
}
