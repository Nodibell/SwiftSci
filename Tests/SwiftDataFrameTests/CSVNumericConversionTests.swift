import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV numeric conversion")
struct CSVNumericConversionTests {
    private func parse(_ text: String) -> Double? {
        Array(text.utf8).withUnsafeBufferPointer { buffer in
            VectorizedByteParsers.parseDouble(
                buffer: buffer,
                offset: CSVFieldOffset(startOffset: 0, length: buffer.count, escapedQuotesPresent: false)
            )
        }
    }

    @Test("Decimal conversion matches Swift Double rounding", arguments: [
        "0.29", "1.015", "123.456789", "-123.456", "999999999.999999",
        "9007199254740991", "9007199254740992", "9007199254740993",
        "9007199254740.992", "9007199254740.993", "000000000000001.23456",
        "0.0000000000000000000001", "0.00000000000000000000001",
        "0.123456789012345678901234567890123456789",
        "184467440737095516160000000000000000000000.125",
        "1.7976931348623157e308", "1e309", "5e-324", "1e-9999", "-1e-9999",
        "1.2345678901234567e-120", "1E+12", ".5", "1.", "+12.5", "-0.0"
    ])
    func decimalRounding(_ literal: String) throws {
        let expected = try #require(Double(literal))
        let actual = try #require(parse(literal))
        #expect(actual.bitPattern == expected.bitPattern)
    }

    @Test("Short decimal corpus has bit-exact rounding")
    func shortDecimalCorpus() throws {
        var state: UInt64 = 0x243F6A8885A308D3
        for _ in 0..<10_000 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let whole = state % 1_000_000
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let fraction = state % 1_000_000
            let literal = "\(whole).\(String(repeating: "0", count: 6 - String(fraction).count))\(fraction)"
            let expected = try #require(Double(literal))
            let actual = try #require(parse(literal))
            #expect(actual.bitPattern == expected.bitPattern, "Incorrect rounding for \(literal)")
        }
    }

    @Test("CSV quoting and whitespace remain supported")
    func csvSyntax() {
        #expect(parse("\" \t-123.456\r\n\"") == -123.456)
        #expect(parse(" \t+0.29 \r\n") == Double("0.29"))
        #expect(parse("-0").map(\.bitPattern) == (-Double.zero).bitPattern)
    }

    @Test("CSV Double overrides preserve rounding, fallbacks, nulls, and integer columns")
    func csvDoubleOverride() async throws {
        let literals = [
            "0.29", "0.00000000000000000000001",
            "0.123456789012345678901234567890123456789",
            "184467440737095516160000000000000000000000.125",
            "1.2345678901234567e-120", "5e-324", "1e309", "-0.0", "-1e-9999",
            "invalid", "NA"
        ]
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("csv-decimals-\(UUID().uuidString).csv")
        defer { try? FileManager.default.removeItem(at: url) }
        let csv = "id,amount\n" + literals.map { "9007199254740993,\($0)\n" }.joined()
        try Data(csv.utf8).write(to: url)
        var options = CSVReadOptions()
        options.columnTypeOverrides = ["id": .int64, "amount": .float64]
        let frame = try await DataFrame(csv: url, options: options)
        let amounts = try #require(frame[column: "amount", as: Double.self])
        #expect(amounts.count == literals.count)
        for (index, literal) in literals.enumerated() {
            #expect(amounts.values[index]?.bitPattern == Double(literal)?.bitPattern,
                    "Incorrect CSV conversion for \(literal)")
        }
        #expect(frame[column: "id", as: Int64.self]?.values ==
                Array<Int64?>(repeating: 9_007_199_254_740_993, count: literals.count))
    }

    @Test("Invalid decimal fields are rejected", arguments: [
        "", " ", "+", "-", ".", "1.2.3", "1e", "1e+", "1e-", "1e2x", "1 2",
        "0x1p0", "12_345", "184467440737095516160000000000x"
    ])
    func invalidFields(_ literal: String) {
        #expect(parse(literal) == nil)
    }
}
