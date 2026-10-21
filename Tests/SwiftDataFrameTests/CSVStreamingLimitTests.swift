import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV streaming row limits")
struct CSVStreamingLimitTests {
    @Test("A stream honors the same row limit as an eager read", arguments: [0, 1, 3, 20])
    func rowLimit(_ limit: Int) async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".csv")
        try Data("value\n1\n2\n3\n4\n5\n".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        var options = CSVReadOptions()
        options.maxRows = limit
        options.columnTypeOverrides = ["value": .int64]
        let eager = try await DataFrame(csv: url, options: options)
        var values = [Int64?]()
        var sizes = [Int]()
        for try await frame in DataFrame.readCSVStream(contentsOf: url, chunkSize: 2, options: options) {
            sizes.append(frame.rowCount)
            values.append(contentsOf: try #require(frame[column: "value", as: Int64.self]).values)
        }
        #expect(values == eager[column: "value", as: Int64.self]?.values)
        #expect(sizes.allSatisfy { $0 > 0 && $0 <= 2 })
    }
}
