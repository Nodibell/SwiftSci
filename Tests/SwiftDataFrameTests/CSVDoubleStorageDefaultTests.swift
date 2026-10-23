import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV optional storage compatibility")
struct CSVDoubleStorageDefaultTests {
    @Test("Default reads retain optional values so repeated exports share storage")
    func defaultExports() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("csv-default-storage-\(UUID().uuidString).csv")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("x\n1.5\nNULL\n-0.0\n".utf8).write(to: url)
        let frame = try await DataFrame(csv: url)
        let column = try #require(frame[column: "x", as: Double.self])
        #expect(column.compactNumericStorage == nil)
        let first = column.values
        let second = column.values
        #expect(first.withUnsafeBufferPointer { lhs in
            second.withUnsafeBufferPointer { rhs in lhs.baseAddress == rhs.baseAddress }
        })
        #expect(column.nullCount == 1)
        #expect(column[2]?.bitPattern == (-0.0).bitPattern)
    }
}
