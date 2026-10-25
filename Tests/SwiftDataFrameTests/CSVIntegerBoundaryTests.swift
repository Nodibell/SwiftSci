import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV integer boundaries")
struct CSVIntegerBoundaryTests {
    @Test("The byte parser accepts both signed integer limits", arguments: [Int.min, Int.max])
    func signedLimits(_ expected: Int) {
        #expect(parse(String(expected)) == expected)
    }

    @Test("Out-of-range decimal integers return nil")
    func overflow() {
        let aboveMax = String(UInt(Int.max) + 1)
        let belowMin = "-" + String(UInt(Int.max) + 2)
        for text in [aboveMax, belowMin, String(repeating: "9", count: 100)] {
            #expect(parse(text) == nil)
        }
    }

    @Test("Signs, quotes, padding and whitespace preserve integer values",
          arguments: [Int.min, Int.max, -1, 0, 1])
    func formatting(_ value: Int) {
        let text = String(value)
        let padded = value < 0 ? "-000" + text.dropFirst() : "+000" + text
        for field in [text, "\"" + text + "\"", " \t" + text + "\r\n", padded] {
            #expect(parse(field) == value)
        }
    }

    @Test("Malformed integer tokens remain invalid",
          arguments: ["", " \t", "+", "-", "1x", "1.0", "1e3", "--1", "+-1", "1 2", "\n1", "\"1"])
    func malformed(_ text: String) {
        #expect(parse(text) == nil)
    }

    @Test("Leading zeroes do not cause magnitude overflow", arguments: ["", "+", "-"])
    func leadingZeroes(_ sign: String) {
        #expect(parse(sign + String(repeating: "0", count: 100) + "1") == (sign == "-" ? -1 : 1))
    }

    @Test("Explicit integer CSV schemas accept limits and coerce invalid values to nil",
          arguments: [CSVColumnConversionMode.serial, .parallel])
    func explicitSchema(_ mode: CSVColumnConversionMode) async throws {
        let csv = "x,label\n\(Int64.min),min\n\(Int64.max),max\n9223372036854775808,above\n-9223372036854775809,below\n?,null\n"
        let url = try write(csv)
        defer { try? FileManager.default.removeItem(at: url) }
        var options = CSVReadOptions()
        options.columnConversionMode = mode
        options.columnTypeOverrides = ["x": .int64]
        options.nullValues = ["?"]
        let frame = try await DataFrame(csv: url, options: options)
        let column = try #require(frame[column: "x", as: Int64.self])
        #expect(column.values == [Int64.min, Int64.max, nil, nil, nil])
        #expect(column.nullCount == 3)
    }

    @Test("Inference retains exact signed limits as integers")
    func inferredLimits() async throws {
        let url = try write("x\n\(Int64.min)\n\(Int64.max)\n")
        defer { try? FileManager.default.removeItem(at: url) }
        let frame = try await DataFrame(csv: url)
        let column = try #require(frame[column: "x", as: Int64.self])
        #expect(column.values == [Int64.min, Int64.max])
    }

    private func write(_ csv: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("csv-integer-\(UUID().uuidString).csv")
        try Data(csv.utf8).write(to: url)
        return url
    }

    private func parse(_ text: String) -> Int? {
        let bytes = Array(("prefix" + text + "suffix").utf8)
        let field = CSVFieldOffset(startOffset: 6, length: text.utf8.count, escapedQuotesPresent: false)
        return bytes.withUnsafeBufferPointer { buffer in
            VectorizedByteParsers.parseInt(buffer: buffer, offset: field)
        }
    }
}
