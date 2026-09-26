import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV numeric null fast path")
struct CSVNumericNullFastPathTests {
    @Test("Only digit-free ASCII null tokens permit numeric parsing first")
    func eligibility() {
        #expect(CSVNullMatcher(CSVReadOptions.default.nullValues).canSkipNumericMatch)
        #expect(CSVNullMatcher([]).canSkipNumericMatch)
        #expect(CSVNullMatcher(["", "NA", "nan", "+", "-", "."]).canSkipNumericMatch)
        for token in ["0", "1", "-0.0", "1e2", "001", "NA1", "é", "K"] {
            #expect(!CSVNullMatcher([token]).canSkipNumericMatch)
        }
    }

    @Test("Numeric custom nulls take priority over successful numeric conversion",
          arguments: ["0", "1", "-0.0", "1e2", "001"])
    func customNumericNulls(_ token: String) async throws {
        var options = CSVReadOptions()
        options.nullValues.insert(token)
        options.columnTypeOverrides = ["integer": .int64, "floating": .float64]
        let csv = "integer,floating\n\(token),\(token)\n\" \(token) \t\",\" \(token) \t\"\n42,2.5\n"
        let frame = try await read(csv, options: options)
        let integers = try #require(frame[column: "integer", as: Int64.self])
        let doubles = try #require(frame[column: "floating", as: Double.self])
        #expect(integers.values == [nil, nil, 42])
        #expect(doubles.values == [nil, nil, 2.5])
        #expect(integers.nullCount == 2)
        #expect(doubles.nullCount == 2)
    }

    @Test("Default null tokens, quoted whitespace and ragged rows preserve conversion")
    func defaultTokensAndMissingFields() async throws {
        var options = CSVReadOptions()
        options.columnTypeOverrides = ["integer": .int64, "floating": .float64]
        let csv = "label,integer,floating\nvalid,\" 9007199254740993 \t\",\" 1.25 \t\"\nmissing,NA,nan\nquoted,\" NA \",\" NaN \"\nragged\nempty,,\nzero,0,-0.0\n"
        let frame = try await read(csv, options: options)
        let integers = try #require(frame[column: "integer", as: Int64.self])
        let doubles = try #require(frame[column: "floating", as: Double.self])
        #expect(integers.values == [9_007_199_254_740_993, nil, nil, nil, nil, 0])
        #expect(doubles.values == [1.25, nil, nil, nil, nil, -0.0])
        #expect(doubles.values.last!!.sign == .minus)
        #expect(integers.nullCount == 4)
        #expect(doubles.nullCount == 4)
    }

    @Test("Unicode null tokens retain canonical String matching")
    func unicodeNulls() async throws {
        var options = CSVReadOptions()
        options.nullValues.insert("é")
        options.columnTypeOverrides = ["integer": .int64, "floating": .float64]
        let csv = "integer,floating\ne\u{301},e\u{301}\n\" é \",\" é \"\n42,2.5\n"
        let frame = try await read(csv, options: options)
        #expect(frame[column: "integer", as: Int64.self]?.values == [nil, nil, 42])
        #expect(frame[column: "floating", as: Double.self]?.values == [nil, nil, 2.5])
    }

    @Test("Special floating null tokens are checked before String fallback")
    func specialFloatingNulls() async throws {
        var options = CSVReadOptions()
        options.nullValues.formUnion(["inf", "-inf", "infinity"])
        options.columnTypeOverrides = ["value": .float64]
        let frame = try await read("value\ninf\n-inf\ninfinity\nnan\n2.5\n", options: options)
        #expect(frame[column: "value", as: Double.self]?.values == [nil, nil, nil, nil, 2.5])
    }

    @Test("Inferred numeric columns preserve exact integers and custom nulls")
    func inferredColumns() async throws {
        var options = CSVReadOptions()
        options.nullValues.insert("001")
        let csv = "integer,floating\n9007199254740993,1.25\n001,001\n42,2.5\n"
        let frame = try await read(csv, options: options)
        #expect(frame[column: "integer", as: Int64.self]?.values == [9_007_199_254_740_993, nil, 42])
        #expect(frame[column: "floating", as: Double.self]?.values == [1.25, nil, 2.5])
    }

    private func read(_ csv: String, options: CSVReadOptions) async throws -> DataFrame {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("csv-numeric-null-\(UUID().uuidString).csv")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(csv.utf8).write(to: url)
        return try await DataFrame(csv: url, options: options)
    }
}
