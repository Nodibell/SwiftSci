import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("Scoped CSV public readers")
struct CSVScopedReaderTests {
    @Test("Large unquoted file and stream reads own all returned values")
    func parallelUnquotedReaders() async throws {
        let rows = 40_000
        let line = "17,0.25,station-with-a-long-name\n"
        let data = Data(("id,value,station\n" + String(repeating: line, count: rows)).utf8)
        #expect(data.count > 1_048_576)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".csv")
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)
        let frame = try await DataFrame(csv: url)
        var chunks = [DataFrame]()
        for try await chunk in DataFrame.readCSVStream(contentsOf: url, chunkSize: 4_093) {
            chunks.append(chunk)
        }
        try FileManager.default.removeItem(at: url)
        #expect(frame.rowCount == rows)
        #expect(chunks.reduce(0) { $0 + $1.rowCount } == rows)
        checkValues(frame)
        for chunk in chunks { checkValues(chunk) }
    }

    private func checkValues(_ frame: DataFrame) {
        #expect(frame.columnNames == ["id", "value", "station"])
        let ids = frame[column: "id", as: Int64.self]
        let values = frame[column: "value", as: Double.self]
        let stations = frame[column: "station", as: String.self]
        #expect(ids?.values.allSatisfy { $0 == 17 } == true)
        #expect(values?.values.allSatisfy { $0 == 0.25 } == true)
        #expect(stations?.values.allSatisfy { $0 == "station-with-a-long-name" } == true)
    }
}
