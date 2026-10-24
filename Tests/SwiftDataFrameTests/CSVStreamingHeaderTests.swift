import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV streaming headers")
struct CSVStreamingHeaderTests {
    @Test("Streamed frames use the eager reader's deduplicated headers")
    func duplicateHeaders() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".csv")
        try Data("id,id,value\n1,2,3\n4,5,6\n".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let eager = try await DataFrame(csv: url)
        var chunks = [DataFrame]()
        for try await chunk in DataFrame.readCSVStream(contentsOf: url, chunkSize: 1) {
            chunks.append(chunk)
        }
        #expect(chunks.count == 2)
        for chunk in chunks { #expect(chunk.columnNames == eager.columnNames) }
        for name in eager.columnNames {
            let values: [Int64?] = chunks.flatMap { $0[column: name, as: Int64.self]!.values }
            #expect(values == eager[column: name, as: Int64.self]?.values)
        }
    }
}
