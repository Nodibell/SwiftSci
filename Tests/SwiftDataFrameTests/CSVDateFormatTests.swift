import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV explicit date formats")
struct CSVDateFormatTests {
    @Test("Each column uses its declared date convention", arguments: [false, true])
    func dateConventions(_ quoted: Bool) async throws {
        let cell = quoted ? "\"1/2/2007\"" : "1/2/2007"
        let csv = "day,month,default\n\(cell),\(cell),\(cell)\nNA,NA,NA\n31/2/2007,31/2/2007,NA\n"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("csv-date-\(UUID().uuidString).csv")
        try Data(csv.utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        var options = CSVReadOptions()
        options.columnTypeOverrides = ["day": .date32, "month": .date32, "default": .date32]
        options.columnDateFormats = ["day": "d/M/yyyy", "month": "M/d/yyyy"]
        let frame = try await DataFrame(csv: url, options: options)
        check(frame)
        let parsed = try CSVReader.parse(csv, options: options)
        check(parsed)
        options.inferTypes = false
        var dates: [Date?] = []
        for try await chunk in DataFrame.readCSVStream(contentsOf: url, chunkSize: 1, options: options) {
            let column = try #require(chunk[column: "day", as: Date.self])
            dates.append(contentsOf: column.values)
        }
        #expect(dates == [Date(timeIntervalSince1970: 1170288000), nil, nil])
    }

    @Test("Date format entries do not override inference or text columns")
    func untypedFormats() throws {
        var options = CSVReadOptions()
        options.columnDateFormats = ["date": "d/M/yyyy"]
        let inferred = CSVReader.inferColumn(name: "date", cells: ["1/2/2007"], options: options)
        let dates = try #require(inferred as? TypedColumn<Date>)
        #expect(dates.values == [Date(timeIntervalSince1970: 1167696000)])
        options.columnTypeOverrides = ["date": .utf8]
        let text = CSVReader.inferColumn(name: "date", cells: ["1/2/2007"], options: options)
        #expect((text as? TypedColumn<String>)?.values == ["1/2/2007"])
    }

    private func check(_ frame: DataFrame) {
        let february = Date(timeIntervalSince1970: 1170288000)
        let january = Date(timeIntervalSince1970: 1167696000)
        #expect(frame[column: "day", as: Date.self]?.values == [february, nil, nil])
        #expect(frame[column: "month", as: Date.self]?.values == [january, nil, nil])
        #expect(frame[column: "default", as: Date.self]?.values == [january, nil, nil])
    }
}
