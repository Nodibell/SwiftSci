import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV conversion boundaries")
struct CSVConversionBoundaryTests {
    @Test("Wide record indices preserve inferred Doubles and absent fields")
    func wideConversion() throws {
        let bytes = Array("1,1.5\n2\n3,2.5\n".utf8)
        try bytes.withUnsafeBufferPointer { buffer in
            let records = SystemsCSVParser().scanIndex(
                buffer: buffer, range: buffer.indices, fieldType: PackedCSVField.self)
            let plan = CSVColumnPlan(buffer: buffer, records: records, startRow: 0,
                rowCount: records.count, column: 1, name: "x", options: .default)
            #expect(plan.candidates.first == .float64)
            let conversion = plan.conversion(type: .float64, selectedRowCount: records.count)
            conversion.appendIndexed(buffer: buffer, records: records, startRow: 0,
                rowCount: records.count, column: 1)
            let column = try #require(conversion.column(named: "x") as? TypedColumn<Double>)
            #expect(column.values == [1.5, nil, 2.5])
            #expect(column.nullCount == 1)
            #expect(!conversion.rejected)
        }
    }

    @Test("Eager reading preserves the existing unfinished-row EOF contract")
    func unfinishedRecord() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".csv")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("id,value\nrow,".utf8).write(to: url)
        let frame = try await DataFrame(csv: url)
        #expect(frame.rowCount == 0)
        #expect(frame.shape.columns == 0)
    }

    @Test("A negative streaming row budget finishes a retained cursor")
    func exhaustedBudget() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".csv")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("x\n1\n2\n".utf8).write(to: url)
        var options = CSVReadOptions()
        options.maxRows = -1
        let cursor = CSVReadCursor(url: url, options: options, chunkSize: 1)
        #expect(try cursor.next() == nil)
        #expect(try cursor.next() == nil)
    }
}
