import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("Eager CSV batch inference")
struct CSVEagerBatchTests {
    @Test("Late Boolean remains eligible after numeric candidates fail", arguments: [1_000, 70_000])
    func lateBoolean(_ rows: Int) async throws {
        let csv = "value,other\n" + String(repeating: "1,7\n", count: rows) + "true,8\n"
        let frame = try await read(csv)
        let values = try #require(frame[column: "value", as: Bool.self])
        #expect(values.count == rows + 1)
        #expect(values.values.allSatisfy { $0 == true })
        let other = try #require(frame[column: "other", as: Int64.self])
        #expect(other.values.last == 8)
    }

    @Test("Late fallback replays exact source tokens", arguments: ["009007199254740993", "0.10000000000000001"])
    func originalText(_ token: String) async throws {
        let csv = "value\n" + String(repeating: token + "\n", count: 70_000) + "offline\n"
        let frame = try await read(csv)
        let values = try #require(frame[column: "value", as: String.self])
        #expect(values.values.prefix(70_000).allSatisfy { $0 == token })
        #expect(values.values.last == "offline")
    }

    @Test("All-null sample keeps later numeric tokens as text")
    func nullSample() async throws {
        let csv = "value\n" + String(repeating: "NA\n", count: 1_000) + "0042\n"
        let frame = try await read(csv)
        let values = try #require(frame[column: "value", as: String.self])
        #expect(values.nullCount == 1_000)
        #expect(values.values.last == "0042")
    }

    @Test("Empty sample fields do not become null when empty is not configured")
    func nonNullEmptyFields() async throws {
        var options = CSVReadOptions()
        options.nullValues = []
        let csv = "value,other\n" + String(repeating: ",1\n", count: 1_000) + "42,2\n"
        let frame = try await read(csv, options: options)
        let values = try #require(frame[column: "value", as: String.self])
        #expect(values.nullCount == 0)
        #expect(values.values.first == "")
        #expect(values.values.last == "42")
    }

    @Test("A row limit excludes a later type failure across internal batches")
    func rowLimit() async throws {
        var options = CSVReadOptions()
        options.maxRows = 70_000
        let csv = "value\n" + String(repeating: "9007199254740993\n", count: 70_000) + "offline\n"
        let frame = try await read(csv, options: options)
        let values = try #require(frame[column: "value", as: Int64.self])
        #expect(values.count == 70_000)
        #expect(values.values.allSatisfy { $0 == 9_007_199_254_740_993 })
    }

    private func read(_ text: String, options: CSVReadOptions = .default) async throws -> DataFrame {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".csv")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(text.utf8).write(to: url)
        return try await DataFrame(csv: url, options: options)
    }
}
