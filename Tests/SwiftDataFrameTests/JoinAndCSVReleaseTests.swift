import Foundation
import Testing
@testable import SwiftDataFrame

struct JoinAndCSVReleaseTests {
    @Test func innerJoinPreservesDuplicateOrderAndPayloadNames() throws {
        let date = Date(timeIntervalSince1970: 0)
        let keyPairs: [(any AnyColumn, any AnyColumn)] = [
            (TypedColumn<Int32>(name: "id", values: [2, nil, 2, 3]),
             TypedColumn<Int32>(name: "id", values: [3, 2, nil, 2])),
            (TypedColumn<Date>(name: "id", values: [date, nil, date, date + 1]),
             TypedColumn<Date>(name: "id", values: [date + 1, date, nil, date])),
            (TypedColumn<Double>(name: "id", values: [2, nil, 2, 3]),
             TypedColumn<Double>(name: "id", values: [3, 2, nil, 2])),
            (TypedColumn<Int>(name: "id", values: [2, nil, 2, 3]),
             TypedColumn<Int>(name: "id", values: [3, 2, nil, 2]))
        ]
        for (leftKey, rightKey) in keyPairs {
            let left = try DataFrame(columns: [
                leftKey, TypedColumn(name: "value", values: ["A", "B", "C", "D"])
            ])
            let right = try DataFrame(columns: [
                rightKey, TypedColumn(name: "value", values: ["W", "X", "Y", "Z"])
            ])
            let joined = try left.joinSIMD(right, on: "id")
            #expect(joined.columnNames == ["id", "value_x", "value_y"])
            #expect(joined[column: "value_x", as: String.self]?.values == ["A", "A", "C", "C", "D"])
            #expect(joined[column: "value_y", as: String.self]?.values == ["X", "Z", "X", "Z", "W"])
            #expect(joined[column: "id"]?.dtype == leftKey.dtype)
            #expect(joined[column: "id"]?.nullCount == 0)
        }
    }

    @Test func genericColumnJoinPreservesIdentityAndReorderedMatches() throws {
        typealias GenericColumn = DataFrameParquetCoverageTests.CustomGenericColumn
        let keys = (0..<24).map(Int64.init)
        let values = keys.map(String.init)
        let left = try DataFrame(columns: [
            GenericColumn(name: "id", dtype: .int64, rawValues: keys.map { $0 as Any }),
            TypedColumn(name: "payload", values: values)
        ])
        let identical = try left.joinSIMD(left, on: "id")
        #expect(identical[column: "payload_x", as: String.self]?.values == values.map(Optional.some))
        #expect(identical[column: "payload_y", as: String.self]?.values == values.map(Optional.some))

        let rightKeys: [Int64?] = [23, 0, nil, 0] + keys.dropFirst().dropLast().reversed().map(Optional.some)
        let right = try DataFrame(columns: [
            GenericColumn(name: "id", dtype: .int64, rawValues: rightKeys.map { $0.map { $0 as Any } }),
            TypedColumn(name: "payload", values: rightKeys.indices.map { "right-\($0)" })
        ])
        let joined = try left.joinSIMD(right, on: "id")
        let expectedKeys = [Int64(0)] + keys
        let keyColumn = try #require(joined[column: "id"])
        #expect((0..<joined.rowCount).map { keyColumn.value(at: $0) as? Int64 } == expectedKeys.map(Optional.some))
        #expect(joined[column: "payload_y", as: String.self]?.values ==
                ([1, 3] + Array((4..<rightKeys.count).reversed()) + [0]).map { "right-\($0)" })
    }

    @Test func csvWritesOptionalAndCompactColumnsWithoutLosingValues() async throws {
        let date = Date(timeIntervalSince1970: 0)
        let compact = CompactNumericColumn(count: 3) { [1.25, nil, -0.0][$0] }
        let frame = try DataFrame(columns: [
            TypedColumn<Int>(name: "int", values: [1, nil, 2]),
            TypedColumn<Int32>(name: "int32", values: [3, nil, 4]),
            TypedColumn<Int64>(name: "int64", values: [9_007_199_254_740_993, nil, 6]),
            TypedColumn<Float>(name: "float", values: [1.5, nil, -2.5]),
            TypedColumn<Bool>(name: "flag", values: [true, nil, false]),
            TypedColumn<Date>(name: "day", values: [date, nil, date]),
            TypedColumn<Double>(name: "compact", compactColumn: compact),
            TypedColumn<String>(name: "note", values: [" NA ", nil, "a,\"b\""])
        ])
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID()).csv")
        defer { try? FileManager.default.removeItem(at: url) }
        try await frame.writeCSV(to: url)
        let raw = try String(contentsOf: url, encoding: .utf8)
        let rows = raw.split(separator: "\n").map { CSVReader.parseRowCells(String($0), delimiter: ",") }
        try #require(rows.count == 4)
        #expect(rows[1].map(\.text) == ["1", "3", "9007199254740993", "1.5", "true",
                                      date.formattedDateOrDateTimeString, "1.25", " NA "])
        #expect(rows[2].count == frame.columnNames.count)
        #expect(rows[2].allSatisfy { $0.text.isEmpty && !$0.wasQuoted })
        #expect(rows[3].map(\.text) == ["2", "4", "6", "-2.5", "false",
                                      date.formattedDateOrDateTimeString, "-0.0", "a,\"b\""])
        #expect(rows[1].last?.wasQuoted == true)
        #expect(rows[3].last?.wasQuoted == true)
    }

    @Test func csvEscapesGenericFieldsAndScalarValuesWithCustomDelimiters() async throws {
        typealias GenericColumn = DataFrameParquetCoverageTests.CustomGenericColumn
        let date = Date(timeIntervalSince1970: 0)
        let frame = try DataFrame(columns: [
            GenericColumn(name: "date", dtype: .date32, rawValues: [date, nil]),
            GenericColumn(name: "text", dtype: .utf8, rawValues: ["NA", "x•y"]),
            GenericColumn(name: "number", dtype: .int64, rawValues: [7, 17]),
            TypedColumn<Int32>(name: "count", values: [71, 17])
        ])
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID()).csv")
        defer { try? FileManager.default.removeItem(at: url) }

        for delimiter: Character in ["7", "•"] {
            try await CSVWriter.write(frame, to: url, delimiter: delimiter)
            let raw = try String(contentsOf: url, encoding: .utf8)
            let rows = raw.split(separator: "\n").map { CSVReader.parseRowCells(String($0), delimiter: delimiter) }
            try #require(rows.count == 3)
            #expect(rows[1].map(\.text) == [date.formattedDateOrDateTimeString, "NA", "7", "71"])
            #expect(rows[2].map(\.text) == ["", "x•y", "17", "17"])
            #expect(rows[1][1].wasQuoted)
            #expect(rows[1][3].wasQuoted == (delimiter == "7"))
            #expect(rows[2][1].wasQuoted == (delimiter == "•"))
        }
    }
}
