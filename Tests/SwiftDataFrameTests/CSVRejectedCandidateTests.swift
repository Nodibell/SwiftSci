import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV rejected candidate evidence")
struct CSVRejectedCandidateTests {
    @Test("A rejected token cannot admit a type excluded by the sample", arguments: ["1", "2", "001"])
    func eligibleCandidates(_ prefix: String) async throws {
        let csv = "flag,keep\n" + String(repeating: prefix + ",7\n", count: 17_000) + "NA,8\ntrue,9\n"
        try await check(csv) { frame in
            #expect(frame.rowCount == 17_002)
            if prefix == "1" {
                let values = try #require(frame[column: "flag", as: Bool.self])
                #expect(values.values.prefix(17_000).allSatisfy { $0 == true })
                #expect(values.values[17_000] == nil)
                #expect(values.values.last == true)
            } else {
                let values = try #require(frame[column: "flag", as: String.self])
                #expect(values.values.prefix(17_000).allSatisfy { $0 == prefix })
                #expect(values.values[17_000] == nil)
                #expect(values.values.last == "true")
            }
            #expect(frame[column: "keep", as: Int64.self]?.values.last == 9)
        }
    }

    @Test("A surviving candidate still validates the earlier rows")
    func earlierContradiction() async throws {
        let csv = "flag\n" + String(repeating: "1\n", count: 17_000) + "002\ntrue\n"
        try await check(csv) { frame in
            let values = try #require(frame[column: "flag", as: String.self])
            #expect(values.values.prefix(17_000).allSatisfy { $0 == "1" })
            #expect(values.values.suffix(2) == ["002", "true"])
        }
    }

    private func check(_ csv: String, body: (DataFrame) throws -> Void) async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".csv")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(csv.utf8).write(to: url)
        let eager = try await DataFrame(csv: url)
        try body(eager)
        var chunks = 0
        for try await frame in DataFrame.readCSVStream(contentsOf: url, chunkSize: 20_000) {
            try body(frame)
            chunks += 1
        }
        #expect(chunks == 1)
    }
}
