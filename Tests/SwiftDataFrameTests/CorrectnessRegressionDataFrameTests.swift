import Foundation
import Testing
@testable import SwiftDataFrame

struct CorrectnessRegressionDataFrameTests {
    @Test func uniqueRetainsDistinctRows() throws {
        let df = try DataFrame(columns: [
            TypedColumn<String>(name: "a", values: ["a|b", "a", nil, "nil", "a|b"]),
            TypedColumn<String>(name: "b", values: ["c", "b|c", "x", "x", "c"])
        ])
        #expect(df.unique.shape.rows == 4)
    }
    @Test func stableVariance() {
        let col = TypedColumn<Double>(name: "x", values: [1e12, nil, 1e12 + 1, 1e12 + 2])
        #expect(abs(col.variance() - 1) < 1e-12)
    }
    @Test func csvNullTokenRoundTrip() async throws {
        let values: [String?] = ["text", "NA", "null", "NaN", "", nil, "None"]
        let df = try DataFrame(columns: [TypedColumn<String>(name: "x", values: values)])
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".csv")
        defer { try? FileManager.default.removeItem(at: url) }
        try await df.writeCSV(to: url)
        let actual = try await DataFrame(csv: url)
        #expect(actual[column: "x", as: String.self]?.values == values)
        #expect(actual[column: "x", as: String.self]?.nullCount == 1)

        var streamed: [String?] = []
        for try await chunk in DataFrame.readCSVStream(contentsOf: url, chunkSize: 2) {
            streamed.append(contentsOf: try #require(chunk[column: "x", as: String.self]).values)
        }
        #expect(streamed == values)

        let raw = try String(contentsOf: url, encoding: .utf8)
        let parsed = try CSVReader.parse(raw, options: .default)
        #expect(parsed[column: "x", as: String.self]?.values == values)
    }

    @Test func csvQuotedVsUnquotedExternalNullsPositionsAndEscaping() async throws {
        let externalCSV = [
            "first,middle,last",
            ",\"NA\",",
            "\"\",\"\",null",
            "NaN,\"line1\nline2\",\"escaped \"\"NA\"\" and, delim\"",
            "None,NA,\"None\"",
            "\"null\",\"NaN\",ordinary"
        ].joined(separator: "\n")

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("csv-external-nulls-\(UUID().uuidString).csv")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(externalCSV.utf8).write(to: url)

        let expectedFirst: [String?] = [nil, "", nil, nil, "null"]
        let expectedMiddle: [String?] = ["NA", "", "line1\nline2", nil, "NaN"]
        let expectedLast: [String?] = [nil, nil, "escaped \"NA\" and, delim", "None", "ordinary"]

        for mode in [CSVColumnConversionMode.parallel, .serial] {
            var options = CSVReadOptions()
            options.columnConversionMode = mode
            let eager = try await DataFrame(csv: url, options: options)
            #expect(eager[column: "first", as: String.self]?.values == expectedFirst)
            #expect(eager[column: "middle", as: String.self]?.values == expectedMiddle)
            #expect(eager[column: "last", as: String.self]?.values == expectedLast)

            var streamFirst: [String?] = []
            var streamMiddle: [String?] = []
            var streamLast: [String?] = []
            for try await chunk in DataFrame.readCSVStream(contentsOf: url, chunkSize: 2, options: options) {
                streamFirst.append(contentsOf: try #require(chunk[column: "first", as: String.self]).values)
                streamMiddle.append(contentsOf: try #require(chunk[column: "middle", as: String.self]).values)
                streamLast.append(contentsOf: try #require(chunk[column: "last", as: String.self]).values)
            }
            #expect(streamFirst == expectedFirst)
            #expect(streamMiddle == expectedMiddle)
            #expect(streamLast == expectedLast)
        }

        let parsed = try CSVReader.parse(externalCSV, options: .default)
        #expect(parsed[column: "first", as: String.self]?.values == expectedFirst)
        #expect(parsed[column: "middle", as: String.self]?.values == expectedMiddle)
        #expect(parsed[column: "last", as: String.self]?.values == expectedLast)

        let roundTripURL = FileManager.default.temporaryDirectory.appendingPathComponent("csv-roundtrip-multi-\(UUID().uuidString).csv")
        defer { try? FileManager.default.removeItem(at: roundTripURL) }
        try await parsed.writeCSV(to: roundTripURL)
        let roundTripped = try await DataFrame(csv: roundTripURL)
        #expect(roundTripped[column: "first", as: String.self]?.values == expectedFirst)
        #expect(roundTripped[column: "middle", as: String.self]?.values == expectedMiddle)
        #expect(roundTripped[column: "last", as: String.self]?.values == expectedLast)
    }

    @Test func csvCustomNullValuesAndDecoderPathParity() async throws {
        let decoderNullSets: [Set<String>] = [
            ["MISSING"],                        // CSVSmallStringDecoder (1 token)
            ["", "MISSING"],                    // CSVSmallStringDecoder (2 tokens)
            ["", "MISSING", "UNSET", "VOID"],   // CSVASCIIPrefixStringDecoder (3+ ASCII tokens)
            ["", "MISSING", "Н/Д"]              // CSVStringDecoder (non-ASCII fallback)
        ]

        for customNulls in decoderNullSets {
            let token = customNulls.first { !$0.isEmpty } ?? ""
            let values: [String?] = ["alpha", token, "", nil, "quoted \"quote\"", "normal"]
            let df = try DataFrame(columns: [TypedColumn<String>(name: "v", values: values)])
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("csv-custom-\(UUID().uuidString).csv")
            defer { try? FileManager.default.removeItem(at: url) }
            try await CSVWriter.write(df, to: url, nullValues: customNulls)

            // Also append unquoted rows for non-empty custom tokens to cover external CSV reads.
            let externalLines: [String] = (0..<20).map { row in
                switch row % 4 {
                case 0: return "\(row),value-\(row)\n"
                case 1: return "\(row),\(token)\n"
                case 2: return "\(row),\"\(token)\"\n"
                default: return customNulls.contains("") ? "\(row),\"\"\n" : "\(row),\n"
                }
            }
            let externalURL = FileManager.default.temporaryDirectory.appendingPathComponent("csv-ext-custom-\(UUID().uuidString).csv")
            defer { try? FileManager.default.removeItem(at: externalURL) }
            try Data(("id,v\n" + externalLines.joined()).utf8).write(to: externalURL)

            let expectedExternal: [String?] = (0..<20).map { row in
                switch row % 4 {
                case 0: return "value-\(row)"
                case 1: return nil
                case 2: return token
                default: return ""
                }
            }

            for mode in [CSVColumnConversionMode.parallel, .serial] {
                var options = CSVReadOptions()
                options.nullValues = customNulls
                options.columnConversionMode = mode

                let readBack = try await DataFrame(csv: url, options: options)
                let expectedRoundTrip: [String?] = customNulls.contains("") ? values : values.map { $0 ?? "" }
                #expect(readBack[column: "v", as: String.self]?.values == expectedRoundTrip)

                let extFrame = try await DataFrame(csv: externalURL, options: options)
                #expect(extFrame[column: "v", as: String.self]?.values == expectedExternal)

                var streamed: [String?] = []
                for try await chunk in DataFrame.readCSVStream(contentsOf: externalURL, chunkSize: 7, options: options) {
                    streamed.append(contentsOf: try #require(chunk[column: "v", as: String.self]).values)
                }
                #expect(streamed == expectedExternal)
            }
        }
    }

    @Test func csvCompactAndWideRecordIndexParity() throws {
        let csv = [
            "a,b,c",
            "1,NA,\"NA\"",
            "2,,\"\"",
            "3,\"null\",null",
            "4,NaN,\"NaN\"",
            "5,None,\"None\""
        ].joined(separator: "\n")
        let bytes = Array(csv.utf8)
        bytes.withUnsafeBufferPointer { buffer in
            let parser = SystemsCSVParser()
            let compact = parser.scanIndex(buffer: buffer, range: 0..<buffer.count, fieldType: CompactCSVField.self)
            let wide = parser.scanIndex(buffer: buffer, range: 0..<buffer.count, fieldType: PackedCSVField.self)
            let headers = compact.row(at: 0).map { VectorizedByteParsers.parseString(buffer: buffer, offset: $0) }
            for mode in [CSVColumnConversionMode.parallel, .serial] {
                for storage in [CSVDoubleStorage.optional, .compact] {
                    var options = CSVReadOptions()
                    options.columnConversionMode = mode
                    options.doubleStorage = storage
                    let compactFrame = try! CSVReader.buildFrame(
                        buffer: buffer, records: compact, startRow: 1,
                        rowCount: compact.count - 1, headers: headers, options: options
                    )
                    let wideFrame = try! CSVReader.buildFrame(
                        buffer: buffer, records: wide, startRow: 1,
                        rowCount: wide.count - 1, headers: headers, options: options
                    )
                    #expect(compactFrame[column: "a", as: Int64.self]?.values == [1, 2, 3, 4, 5])
                    #expect(compactFrame[column: "b", as: String.self]?.values == [nil, nil, "null", nil, nil])
                    #expect(compactFrame[column: "c", as: String.self]?.values == ["NA", "", nil, "NaN", "None"])
                    #expect(wideFrame[column: "a", as: Int64.self]?.values == compactFrame[column: "a", as: Int64.self]?.values)
                    #expect(wideFrame[column: "b", as: String.self]?.values == compactFrame[column: "b", as: String.self]?.values)
                    #expect(wideFrame[column: "c", as: String.self]?.values == compactFrame[column: "c", as: String.self]?.values)
                }
            }
        }
    }

    @Test func csvNumericInferenceAndExplicitOverridesWithQuotedNulls() async throws {
        // 1. Unquoted NA/NaN infers numeric with nil, whereas quoted "NA"/"NaN"/"" infers String.
        let mixedCSV = [
            "inferred_int_unquoted,inferred_int_quoted,inferred_double_unquoted,inferred_double_quoted,late_quoted",
            "10,10,1.5,1.5,100",
            "NA,\"NA\",NaN,\"NaN\",200",
            "20,20,2.5,2.5,\"NA\""
        ].joined(separator: "\n")

        // Ensure late rejection beyond the 1,000-row sample still falls back to String on quoted null.
        let largeRows = Array(repeating: "1,2.5", count: 1_001).joined(separator: "\n")
        let lateCSV = "late_int,late_double\n" + largeRows + "\n\"NA\",\"NaN\"\n"

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("csv-numeric-quoted-\(UUID().uuidString).csv")
        let lateURL = FileManager.default.temporaryDirectory.appendingPathComponent("csv-late-quoted-\(UUID().uuidString).csv")
        defer {
            try? FileManager.default.removeItem(at: url)
            try? FileManager.default.removeItem(at: lateURL)
        }
        try Data(mixedCSV.utf8).write(to: url)
        try Data(lateCSV.utf8).write(to: lateURL)

        let inferred = try await DataFrame(csv: url)
        #expect(inferred[column: "inferred_int_unquoted", as: Int64.self]?.values == [10, nil, 20])
        #expect(inferred[column: "inferred_int_quoted", as: String.self]?.values == ["10", "NA", "20"])
        #expect(inferred[column: "inferred_double_unquoted", as: Double.self]?.values == [1.5, nil, 2.5])
        #expect(inferred[column: "inferred_double_quoted", as: String.self]?.values == ["1.5", "NaN", "2.5"])
        #expect(inferred[column: "late_quoted", as: String.self]?.values == ["100", "200", "NA"])

        let inferredParsed = try CSVReader.parse(mixedCSV, options: .default)
        #expect(inferredParsed[column: "inferred_int_unquoted", as: Int64.self]?.values == [10, nil, 20])
        #expect(inferredParsed[column: "inferred_int_quoted", as: String.self]?.values == ["10", "NA", "20"])
        #expect(inferredParsed[column: "inferred_double_unquoted", as: Double.self]?.values == [1.5, nil, 2.5])
        #expect(inferredParsed[column: "inferred_double_quoted", as: String.self]?.values == ["1.5", "NaN", "2.5"])

        let lateFrame = try await DataFrame(csv: lateURL)
        let lateInts = try #require(lateFrame[column: "late_int", as: String.self])
        let lateDoubles = try #require(lateFrame[column: "late_double", as: String.self])
        #expect(lateInts.values.last == "NA")
        #expect(lateDoubles.values.last == "NaN")

        // 2. Explicit numeric overrides coerce quoted null tokens (including "NaN" when in nullValues) to nil.
        for storage in [CSVDoubleStorage.optional, .compact] {
            var options = CSVReadOptions()
            options.doubleStorage = storage
            options.columnTypeOverrides = [
                "inferred_int_quoted": .int64,
                "inferred_double_quoted": .float64
            ]
            let overridden = try await DataFrame(csv: url, options: options)
            #expect(overridden[column: "inferred_int_quoted", as: Int64.self]?.values == [10, nil, 20])
            #expect(overridden[column: "inferred_double_quoted", as: Double.self]?.values == [1.5, nil, 2.5])

            let overriddenParsed = try CSVReader.parse(mixedCSV, options: options)
            #expect(overriddenParsed[column: "inferred_int_quoted", as: Int64.self]?.values == [10, nil, 20])
            #expect(overriddenParsed[column: "inferred_double_quoted", as: Double.self]?.values == [1.5, nil, 2.5])
        }
    }
    @Test func uniqueDatesKeepSubseconds() throws {
        let values = [Date(timeIntervalSince1970: 0.1), Date(timeIntervalSince1970: 0.2)]
        let df = try DataFrame(columns: [TypedColumn<Date>(name: "time", values: values)])
        #expect(df.unique.shape.rows == 2)
    }
    @Test func varianceAvoidsIntermediateOverflow() {
        let col = TypedColumn<Double>(name: "x", values: [-1e154, 1e154, -1e154, 1e154])
        #expect(abs(col.variance() / (1e308 / 3 * 4) - 1) < 1e-14)
        let constant = TypedColumn<Double>(name: "x", values: [1e308, 1e308])
        #expect(constant.variance() == 0)
    }

    @Test func uniqueTypedKeysAndNaNAndZeroPolicies() throws {
        // Large Int64, Double, Float, String, Bool, subsecond Date, nil vs "nil", and storage variants
        let nanA = Double(bitPattern: 0x7ff8_0000_0000_0001)
        let nanB = Double(bitPattern: 0x7ff8_0000_0000_0002)
        let floatNanA = Float(bitPattern: 0x7fc0_0001)
        let floatNanB = Float(bitPattern: 0x7fc0_0002)

        // NaN payloads deduplicate together while remaining distinct from nil; -0.0 == +0.0
        let dfMixed = try DataFrame(columns: [
            TypedColumn<Double>(name: "d", values: [nanA, nanB, nil, -0.0, 0.0, nanA]),
            TypedColumn<Float>(name: "f", values: [floatNanA, floatNanB, nil, -0.0, 0.0, floatNanA]),
            TypedColumn<Int64>(name: "i", values: [Int64.max, Int64.max, Int64.max, Int64.max - 1, Int64.max - 1, Int64.max]),
            TypedColumn<String>(name: "s", values: ["nil", "nil", "nil", "z", "z", "other"])
        ])
        let uniqueDf = dfMixed.unique
        // Distinct rows:
        // row 0: (NaN, floatNaN, max, "nil") [row 1 deduplicates NaN payloads!]
        // row 2: (nil, nil, max, "nil") [nil != NaN!]
        // row 3: (0.0, 0.0, max-1, "z") [row 4 collapses -0.0 == +0.0!]
        // row 5: (NaN, floatNaN, max, "other")
        #expect(uniqueDf.shape.rows == 4)

        // Check compact (.dense) vs nullable (.optional) storage equivalence
        let denseInts = TypedColumn<Int64>(name: "big", values: [Int64.max, Int64.max - 1, Int64.max])
        let nullableInts = TypedColumn<Int64>(name: "big", values: [Optional(Int64.max), Optional(Int64.max - 1), Optional(Int64.max)])
        #expect(denseInts.typedUnique.values == [Int64.max, Int64.max - 1])
        #expect(nullableInts.typedUnique.values == [Int64.max, Int64.max - 1])

        // TypedColumn.typedUnique NaN & -0.0 policy on Double & Float (with valid NaN + nil)
        let doubleCol = TypedColumn<Double>(name: "x", values: [nanA, nanB, nil, -0.0, 0.0, nil, 1.5])
        let uniqueDoubles = doubleCol.typedUnique
        #expect(uniqueDoubles.count == 4)
        #expect(uniqueDoubles.nullCount == 1)
        #expect(uniqueDoubles[0]?.isNaN == true)
        #expect(uniqueDoubles[1] == nil)
        #expect(uniqueDoubles[2] == 0.0)
        #expect(uniqueDoubles[3] == 1.5)

        let floatCol = TypedColumn<Float>(name: "fx", values: [floatNanA, floatNanB, -0.0, 0.0, nil])
        let uniqueFloats = floatCol.typedUnique
        #expect(uniqueFloats.count == 3)
        #expect(uniqueFloats[0]?.isNaN == true)
        #expect(uniqueFloats[1] == 0.0)
        #expect(uniqueFloats[2] == nil)
    }

    @Test func varianceAndStdDevEdgeCases() {
        // Empty and single-element columns
        #expect(TypedColumn<Double>(name: "empty", values: []).variance() == 0.0)
        #expect(TypedColumn<Double>(name: "empty", values: []).stdDev() == 0.0)
        #expect(TypedColumn<Double>(name: "single", values: [42.0]).variance() == 0.0)
        #expect(TypedColumn<Double>(name: "single", values: [42.0]).stdDev() == 0.0)
        #expect(TypedColumn<Double>(name: "oneValid", values: [nil, 5.0, nil]).variance() == 0.0)

        // Ordinary small sample [2, 4, 4, 4, 5, 5, 7, 9] -> sample var = 32/7, sd = sqrt(32/7)
        let small = TypedColumn<Double>(name: "s", values: [2, 4, 4, 4, 5, 5, 7, 9])
        #expect(abs(small.variance() - (32.0 / 7.0)) < 1e-12)
        #expect(abs(small.stdDev() - sqrt(32.0 / 7.0)) < 1e-12)

        // Constant column with nulls
        let constWithNulls = TypedColumn<Double>(name: "c", values: [1e12, nil, 1e12, 1e12])
        #expect(constWithNulls.variance() == 0.0)
        #expect(constWithNulls.stdDev() == 0.0)

        // Valid NaN (distinct from missing nil) & Infinity
        let validNaN = TypedColumn<Double>(name: "nan", values: [1.0, nil, Double.nan])
        #expect(validNaN.variance().isNaN && validNaN.stdDev().isNaN)

        let withInf = TypedColumn<Double>(name: "inf", values: [1.0, .infinity])
        #expect(withInf.variance().isNaN && withInf.stdDev().isNaN)

        // Large-magnitude values whose variance overflows Double while stdDev remains representable
        let midLarge = TypedColumn<Double>(name: "ml", values: [-1e155, 1e155])
        #expect(midLarge.variance() == .infinity)
        #expect(midLarge.stdDev().isFinite)
        #expect(abs(midLarge.stdDev() / (sqrt(2.0) * 1e155) - 1) < 1e-12)

        let extremeSpan = TypedColumn<Double>(name: "ex", values: [-1e308, 1e308])
        #expect(extremeSpan.variance() == .infinity)
        #expect(extremeSpan.stdDev().isFinite)
        #expect(abs(extremeSpan.stdDev() / (sqrt(2.0) * 1e308) - 1) < 1e-12)
    }
}
