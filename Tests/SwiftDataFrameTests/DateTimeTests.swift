import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("Date and Datetime Handling in DataFrame")
struct DateTimeTests {
    @Test("Parse various date and datetime strings including coffee-sales formats")
    func testDateParsing() {
        // Coffee sales formats: "2024-03-01" and "2024-03-01 10:15:40.500" or "2024-03-01 10:15:40"
        let dateOnly = Date.parse(from: "2024-03-01")
        #expect(dateOnly != nil)
        #expect(dateOnly?.hasTimeComponent == false)
        #expect(dateOnly?.formattedDateString == "2024-03-01")
        #expect(dateOnly?.formattedDateOrDateTimeString == "2024-03-01")

        let dateTimeMilli = Date.parse(from: "2024-03-01 10:15:40.500")
        #expect(dateTimeMilli != nil)
        #expect(dateTimeMilli?.hasTimeComponent == true)
        #expect(dateTimeMilli?.formattedDateTimeString == "2024-03-01 10:15:40")
        #expect(dateTimeMilli?.formattedDateOrDateTimeString == "2024-03-01 10:15:40")

        let dateTimeSec = Date.parse(from: "2024-03-01 10:15:40")
        #expect(dateTimeSec != nil)
        #expect(dateTimeSec?.hasTimeComponent == true)
        #expect(dateTimeSec?.formattedDateTimeString == "2024-03-01 10:15:40")

        // ISO8601
        let isoDate = Date.parse(from: "2024-03-01T10:15:40Z")
        #expect(isoDate != nil)
        #expect(isoDate?.hasTimeComponent == true)

        // Quoted strings
        let quoted = Date.parse(from: "\"2024-03-01 10:15:40\"")
        #expect(quoted != nil)

        // Non-dates
        #expect(Date.parse(from: "123456") == nil)
        #expect(Date.parse(from: "525000.50") == nil)
        #expect(Date.parse(from: "Latte") == nil)
        #expect(Date.parse(from: "null") == nil)
        #expect(Date.parse(from: "") == nil)
    }

    @Test("stringHasTime detection")
    func testStringHasTime() {
        #expect(Date.stringHasTime("2024-03-01 10:15:40.500"))
        #expect(Date.stringHasTime("2024-03-01 10:15:40"))
        #expect(Date.stringHasTime("2024-03-01T10:15:40Z"))
        #expect(!Date.stringHasTime("2024-03-01"))
        #expect(!Date.stringHasTime("01/03/2024"))
    }

    @Test("TypedColumn<Date> toStrings preserves time for datetime columns")
    func testTypedColumnDateToStrings() {
        let d1 = Date.parse(from: "2024-03-01 10:15:40")!
        let d2 = Date.parse(from: "2024-03-02 00:00:00")! // midnight in a datetime column
        let col = TypedColumn<Date>(name: "datetime", values: [d1, d2, nil])
        let strings = col.toStrings()
        #expect(strings == ["2024-03-01 10:15:40", "2024-03-02 00:00:00", "null"])

        let dDateOnly1 = Date.parse(from: "2024-03-01")!
        let dDateOnly2 = Date.parse(from: "2024-03-02")!
        let colDate = TypedColumn<Date>(name: "date", values: [dDateOnly1, dDateOnly2])
        let dateStrings = colDate.toStrings()
        #expect(dateStrings == ["2024-03-01", "2024-03-02"])
    }

    @Test("CSV Reader and Writer handle coffee-sales datetime columns correctly")
    func testCSVDateTimeRoundTrip() async throws {
        let csvContent = """
date,datetime,cash_type,money,coffee_name
2024-03-01,2024-03-01 10:15:40,card,38.70,Latte
2024-03-02,2024-03-02 11:20:15,cash,42.00,Americano
"""
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("coffee_test_\(UUID().uuidString).csv")
        try csvContent.write(to: tempURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tempURL) }
        var options = CSVReadOptions()
        options.inferTypes = true
        let df = try await DataFrame(csv: tempURL, options: options)
        #expect(df.shape.rows == 2)
        #expect(df.shape.columns == 5)

        guard let dtCol = df[column: "datetime", as: Date.self] else {
            Issue.record("Expected datetime column to be inferenced as Date")
            return
        }
        #expect(dtCol.values.compactMap { $0 }.count == 2)
        #expect(dtCol.values[0]?.hasTimeComponent == true)
        #expect(dtCol.toStrings()[0] == "2024-03-01 10:15:40")

        guard let dateCol = df[column: "date", as: Date.self] else {
            Issue.record("Expected date column to be inferenced as Date")
            return
        }
        #expect(dateCol.values[0]?.hasTimeComponent == false)
        #expect(dateCol.toStrings()[0] == "2024-03-01")

        let exportURL = FileManager.default.temporaryDirectory.appendingPathComponent("coffee_export_\(UUID().uuidString).csv")
        try await df.writeCSV(to: exportURL)
        defer { try? FileManager.default.removeItem(at: exportURL) }
        let exportedCSV = try String(contentsOf: exportURL, encoding: .utf8)
        #expect(exportedCSV.contains("2024-03-01 10:15:40"))
        #expect(exportedCSV.contains("2024-03-01,"))
    }
}
