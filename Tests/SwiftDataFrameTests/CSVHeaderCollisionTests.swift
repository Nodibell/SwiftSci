import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV generated header names")
struct CSVHeaderCollisionTests {
    @Test("Generated suffixes remain unique when a source header already uses that suffix")
    func generatedSuffixes() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".csv")
        try Data("id,id,id_1\n1,2,3\n".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let frame = try await DataFrame(csv: url)
        check(frame)
        for try await chunk in DataFrame.readCSVStream(contentsOf: url, chunkSize: 1) {
            check(chunk)
        }
    }

    private func check(_ frame: DataFrame) {
        #expect(frame.columnNames == ["id", "id_1", "id_1_1"])
        #expect(frame[column: "id", as: Int64.self]?.values == [1])
        #expect(frame[column: "id_1", as: Int64.self]?.values == [2])
        #expect(frame[column: "id_1_1", as: Int64.self]?.values == [3])
    }
}
