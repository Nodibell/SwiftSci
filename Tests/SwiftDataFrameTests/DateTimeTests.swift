import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("Date and Datetime Handling in DataFrame")
struct DateTimeTests {

    @Test("Parse various date and datetime strings including coffee-sales formats and sub-second precision")
    func testDateParsing() {
        // Coffee sales formats: "2024-03-01" and "2024-03-01 10:15:40.500" or "2024-03-01 10:15:40"
        let dateOnly = Date.parse(from: "2024-03-01")
        #expect(dateOnly != nil)
        #expect(dateOnly?.hasTimeComponent == false)
        #expect(dateOnly?.formattedDateString == "2024-03-01")
        #expect(dateOnly?.formattedDateOrDateTimeString == "2024-03-01")

        // Sub-second precision (milliseconds .SSS)
        let dateTimeMilli = Date.parse(from: "2024-03-01 10:15:40.500")
        #expect(dateTimeMilli != nil)
        #expect(dateTimeMilli?.hasTimeComponent == true)
        #expect(dateTimeMilli?.formattedDateTimeString == "2024-03-01 10:15:40.500")
        #expect(dateTimeMilli?.formattedDateOrDateTimeString == "2024-03-01 10:15:40.500")

        // Sub-second precision (microseconds .SSSSSS)
        let dateTimeMicro = Date.parse(from: "2024-03-01 10:15:40.123456")
        #expect(dateTimeMicro != nil)
        #expect(dateTimeMicro?.hasTimeComponent == true)
        #expect(dateTimeMicro?.formattedDateTimeString == "2024-03-01 10:15:40.123456")
        #expect(dateTimeMicro?.formattedDateOrDateTimeString == "2024-03-01 10:15:40.123456")

        // Second-precision datetime
        let dateTimeSec = Date.parse(from: "2024-03-01 10:15:40")
        #expect(dateTimeSec != nil)
        #expect(dateTimeSec?.hasTimeComponent == true)
        #expect(dateTimeSec?.formattedDateTimeString == "2024-03-01 10:15:40")
        #expect(dateTimeSec?.formattedDateOrDateTimeString == "2024-03-01 10:15:40")

        // ISO8601 variations
        let isoDate = Date.parse(from: "2024-03-01T10:15:40Z")
        #expect(isoDate != nil)
        #expect(isoDate?.hasTimeComponent == true)

        let isoDateOffset = Date.parse(from: "2024-03-01T10:15:40+02:00")
        #expect(isoDateOffset != nil)
        #expect(isoDateOffset?.hasTimeComponent == true)

        let isoSpaceZ = Date.parse(from: "2024-03-01 10:15:40 Z")
        #expect(isoSpaceZ != nil)

        // Quoted strings
        let doubleQuoted = Date.parse(from: "\"2024-03-01 10:15:40\"")
        #expect(doubleQuoted != nil)

        let singleQuoted = Date.parse(from: "'2024-03-01 10:15:40'")
        #expect(singleQuoted != nil)

        // Slash and dot separated dates
        let slashDate = Date.parse(from: "2024/03/01 10:15:40")
        #expect(slashDate != nil)

        let dotDate = Date.parse(from: "2024.03.01 10:15:40.500")
        #expect(dotDate != nil)
        #expect(dotDate?.hasTimeComponent == true)

        let dotDateOnly = Date.parse(from: "2024.03.01")
        #expect(dotDateOnly != nil)

        // 4-digit year standalone
        let yearOnly = Date.parse(from: "2024")
        #expect(yearOnly != nil)
        #expect(yearOnly?.formattedDateString == "2024-01-01")

        // 2-digit years
        let twoDigitYear1 = Date.parse(from: "12/30/20")
        #expect(twoDigitYear1 != nil)
        #expect(twoDigitYear1?.formattedDateString == "2020-12-30")

        let twoDigitYear2 = Date.parse(from: "30/12/20")
        #expect(twoDigitYear2 != nil)
        #expect(twoDigitYear2?.formattedDateString == "2020-12-30")

        let twoDigitYear3 = Date.parse(from: "20-12-30")
        #expect(twoDigitYear3 != nil)
        #expect(twoDigitYear3?.formattedDateString == "2020-12-30")

        let twoDigitYearOld = Date.parse(from: "85-05-12")
        #expect(twoDigitYearOld != nil)
        #expect(twoDigitYearOld?.formattedDateString == "1985-05-12")

        let twoDigitYearShort = Date.parse(from: "1/21/20")
        #expect(twoDigitYearShort != nil)
        #expect(twoDigitYearShort?.formattedDateString == "2020-01-21")

        let twoDigitYearAmbiguous = Date.parse(from: "01/03/20")
        #expect(twoDigitYearAmbiguous != nil)
        #expect(twoDigitYearAmbiguous?.formattedDateString == "2020-01-03")

        // Non-dates and invalid inputs
        #expect(Date.parse(from: "123456") == nil)
        #expect(Date.parse(from: "525000.50") == nil)
        #expect(Date.parse(from: "Latte") == nil)
        #expect(Date.parse(from: "null") == nil)
        #expect(Date.parse(from: "NA") == nil)
        #expect(Date.parse(from: "NaN") == nil)
        #expect(Date.parse(from: "—") == nil)
        #expect(Date.parse(from: "") == nil)
        #expect(Date.parse(from: "2024-13-45") == nil)
        #expect(Date.parse(from: "99/99/9999") == nil)

        // Helpers and properties
        #expect(Date.isDateString("2024-03-01"))
        #expect(!Date.isDateString("Latte"))
        #expect(dateOnly?.doubleValue == nil)
    }

    @Test("Ambiguous date formats resolve to MM/dd/yyyy by default")
    func testAmbiguousDates() {
        // 01/03/2024 has month <= 12 and day <= 12.
        // Under standard default precedence (MM/dd/yyyy), this resolves to January 3, 2024.
        let ambiguousDate = Date.parse(from: "01/03/2024")
        #expect(ambiguousDate != nil)
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(secondsFromGMT: 0) ?? .gmt
        #expect(cal.component(.month, from: ambiguousDate!) == 1)
        #expect(cal.component(.day, from: ambiguousDate!) == 3)
        #expect(cal.component(.year, from: ambiguousDate!) == 2024)

        // Day > 12: 25/03/2024 is unambiguously dd/MM/yyyy -> March 25, 2024
        let dayFirstDate = Date.parse(from: "25/03/2024")
        #expect(dayFirstDate != nil)
        #expect(cal.component(.month, from: dayFirstDate!) == 3)
        #expect(cal.component(.day, from: dayFirstDate!) == 25)
        #expect(cal.component(.year, from: dayFirstDate!) == 2024)

        // Month-first unambiguous: 03/25/2024 -> March 25, 2024
        let monthFirstDate = Date.parse(from: "03/25/2024")
        #expect(monthFirstDate != nil)
        #expect(cal.component(.month, from: monthFirstDate!) == 3)
        #expect(cal.component(.day, from: monthFirstDate!) == 25)
    }

    @Test("stringHasTime detection")
    func testStringHasTime() {
        #expect(Date.stringHasTime("2024-03-01 10:15:40.500"))
        #expect(Date.stringHasTime("2024-03-01 10:15:40"))
        #expect(Date.stringHasTime("2024-03-01T10:15:40Z"))
        #expect(!Date.stringHasTime("2024-03-01"))
        #expect(!Date.stringHasTime("01/03/2024"))
    }

    @Test("TypedColumn<Date> toStrings preserves time and sub-seconds for datetime columns")
    func testTypedColumnDateToStrings() {
        let d1 = Date.parse(from: "2024-03-01 10:15:40.500")!
        let d2 = Date.parse(from: "2024-03-01 10:15:40.123456")!
        let d3 = Date.parse(from: "2024-03-02 00:00:00")! // midnight in a datetime column
        let col = TypedColumn<Date>(name: "datetime", values: [d1, d2, d3, nil])
        let strings = col.toStrings()
        #expect(strings == [
            "2024-03-01 10:15:40.500",
            "2024-03-01 10:15:40.123456",
            "2024-03-02 00:00:00",
            "null"
        ])

        let dDateOnly1 = Date.parse(from: "2024-03-01")!
        let dDateOnly2 = Date.parse(from: "2024-03-02")!
        let colDate = TypedColumn<Date>(name: "date", values: [dDateOnly1, dDateOnly2])
        let dateStrings = colDate.toStrings()
        #expect(dateStrings == ["2024-03-01", "2024-03-02"])
    }

    @Test("CSV Reader and Writer handle sub-second datetime columns with complete round-trip preservation")
    func testCSVDateTimeRoundTrip() async throws {
        let csvContent = """
date,datetime,cash_type,money,coffee_name
2024-03-01,2024-03-01 10:15:40.500,card,38.70,Latte
2024-03-02,2024-03-02 11:20:15.123456,cash,42.00,Americano
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
        #expect(dtCol.toStrings()[0] == "2024-03-01 10:15:40.500")
        #expect(dtCol.toStrings()[1] == "2024-03-02 11:20:15.123456")

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
        #expect(exportedCSV.contains("2024-03-01 10:15:40.500"))
        #expect(exportedCSV.contains("2024-03-02 11:20:15.123456"))
        #expect(exportedCSV.contains("2024-03-01,"))

        // Read exported CSV back into DataFrame to verify full round-trip
        let roundTripDF = try await DataFrame(csv: exportURL, options: options)
        guard let roundTripDtCol = roundTripDF[column: "datetime", as: Date.self] else {
            Issue.record("Expected round-trip datetime column to be inferenced as Date")
            return
        }
        #expect(roundTripDtCol.toStrings()[0] == "2024-03-01 10:15:40.500")
        #expect(roundTripDtCol.toStrings()[1] == "2024-03-02 11:20:15.123456")
    }
}
