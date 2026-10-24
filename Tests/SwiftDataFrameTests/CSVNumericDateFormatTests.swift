import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("Prepared CSV numeric dates")
struct CSVNumericDateFormatTests {
    @Test("Every supported numeric order converts directly", arguments: ["d/M/yyyy", "M/d/yyyy", "yyyy-MM-dd", "yyyy-d-M"])
    func directConversion(_ pattern: String) throws {
        let format = try #require(CSVNumericDateFormat(pattern))
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = pattern
        formatter.isLenient = false
        for seconds in [-11_676_096_000.0, -86_400, 0, 951_782_400, 1_709_164_800, 253_402_214_400] {
            let expected = Date(timeIntervalSince1970: seconds)
            let text = formatter.string(from: expected)
            let bytes = Array(text.utf8)
            let date = bytes.withUnsafeBufferPointer { format.parse(buffer: $0, range: $0.indices) }
            #expect(date == expected)
        }
    }

    @Test("Unsupported formats keep the Foundation conversion")
    func formatterFallback() {
        #expect(CSVNumericDateFormat("yyyy-MM-dd HH:mm:ss.SSS ZZZZZ") == nil)
        let parser = CSVDateParser(format: "yyyy-MM-dd HH:mm:ss.SSS ZZZZZ")
        let date = parser.parse("1970-01-01 01:00:00.125 +01:00")
        #expect(date?.timeIntervalSince1970 == 0.125)
    }

    @Test("Byte conversion handles quotes and nonzero offsets", arguments: [false, true])
    func byteFields(_ escaped: Bool) {
        let parser = CSVDateParser(format: "d/M/yyyy")
        let value = escaped ? "\"1/2/2007\"" : " 1/2/2007 "
        let text = "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\""
        let bytes = Array(("prefix" + text + "suffix").utf8)
        let field = CSVFieldOffset(startOffset: 6, length: text.utf8.count, escapedQuotesPresent: escaped)
        let date = bytes.withUnsafeBufferPointer { parser.parse(buffer: $0, offset: field) }
        #expect(date == Date(timeIntervalSince1970: 1_170_288_000))
    }
    @Test("Configured date-shaped null tokens take priority over direct conversion")
    func dateNullToken() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".csv")
        try Data("date\n1/2/2007\n2/2/2007\n".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        var options = CSVReadOptions()
        options.columnTypeOverrides = ["date": .date32]
        options.columnDateFormats = ["date": "d/M/yyyy"]
        options.nullValues = ["1/2/2007"]
        let frame = try await DataFrame(csv: url, options: options)
        #expect(frame[column: "date", as: Date.self]?.values == [nil, Date(timeIntervalSince1970: 1_170_374_400)])
    }

}
