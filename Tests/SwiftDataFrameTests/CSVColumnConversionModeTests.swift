import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV column conversion modes")
struct CSVColumnConversionModeTests {
    @Test("Serial and parallel reads preserve native types, nulls, and streamed output",
          arguments: [CSVColumnConversionMode.parallel, .serial])
    func publicReads(_ mode: CSVColumnConversionMode) async throws {
        var options = CSVReadOptions()
        options.columnConversionMode = mode
        options.columnTypeOverrides = ["id": .int64, "value": .float64, "flag": .boolean, "date": .date32]
        options.columnDateFormats = ["date": "d/M/yyyy"]
        let csv = "id,value,flag,date,label\n1,2.5,true,2/1/2020,\"alpha\nline\"\nNA,invalid,maybe,bad,tail\n"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".csv")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(csv.utf8).write(to: url)
        let eager = try await DataFrame(csv: url, options: options)
        let integers: [Int64?] = [1, nil]
        let doubles: [Double?] = [2.5, nil]
        let flags: [Bool?] = [true, nil]
        let labels: [String?] = ["alpha\nline", "tail"]
        #expect(eager[column: "id", as: Int64.self]?.values == integers)
        #expect(eager[column: "value", as: Double.self]?.values == doubles)
        #expect(eager[column: "flag", as: Bool.self]?.values == flags)
        #expect(eager[column: "label", as: String.self]?.values == labels)
        let dates = try #require(eager[column: "date", as: Date.self])
        #expect(dates[0]?.timeIntervalSince1970 == 1_577_923_200)
        #expect(dates[1] == nil)
        var offset = 0
        for try await chunk in DataFrame.readCSVStream(contentsOf: url, chunkSize: 1, options: options) {
            #expect(chunk.rowCount == 1)
            #expect(chunk.dtypes == eager.dtypes)
            #expect(chunk[column: "id", as: Int64.self]?[0] == integers[offset])
            #expect(chunk[column: "value", as: Double.self]?[0] == doubles[offset])
            #expect(chunk[column: "flag", as: Bool.self]?[0] == flags[offset])
            #expect(chunk[column: "date", as: Date.self]?[0] == dates[offset])
            #expect(chunk[column: "label", as: String.self]?[0] == labels[offset])
            offset += 1
        }
        #expect(offset == 2)
    }
}
