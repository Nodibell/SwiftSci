import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV date conversion compatibility")
struct CSVDateParityTests {
    @Test("Declared numeric formats retain Foundation calendar results",
          arguments: ["d/M/yyyy", "M/d/yyyy", "yyyy-MM-dd", "d.M.yyyy"])
    func calendarParity(_ pattern: String) async throws {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = pattern
        formatter.isLenient = false
        var fields = [String]()
        for year in [7, 1582, 1600, 1900, 2000, 2024, 9999] {
            for month in [2, 10] {
                for day in [1, 28, 29, 31] {
                    switch pattern {
                    case "d/M/yyyy": fields.append("\(day)/\(month)/\(year)")
                    case "M/d/yyyy": fields.append("\(month)/\(day)/\(year)")
                    case "yyyy-MM-dd": fields.append("\(year)-\(month)-\(day)")
                    default: fields.append("\(day).\(month).\(year)")
                    }
                }
            }
        }
        fields += ["NA", "1-2-2007", "1/2/07", "1/2/2007junk", "'1/2/2007'", "١/٢/٢٠٠٧"]
        let expected: [Date?] = fields.map { $0 == "NA" ? nil : formatter.date(from: Date.cleanDateString($0)) }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".csv")
        try Data(("date\n" + fields.joined(separator: "\n") + "\n").utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        var options = CSVReadOptions()
        options.columnTypeOverrides = ["date": .date32]
        options.columnDateFormats = ["date": pattern]
        let frame = try await DataFrame(csv: url, options: options)
        #expect(frame[column: "date", as: Date.self]?.values == expected)
        var streamed = [Date?]()
        for try await chunk in DataFrame.readCSVStream(contentsOf: url, chunkSize: 7, options: options) {
            streamed.append(contentsOf: try #require(chunk[column: "date", as: Date.self]).values)
        }
        #expect(streamed == expected)
    }
}
