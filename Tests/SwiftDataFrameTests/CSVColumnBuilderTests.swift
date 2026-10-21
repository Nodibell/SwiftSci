import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV column null counts")
struct CSVColumnBuilderTests {
    @Test("Builder caches the exact count for empty, present, mixed and missing values")
    func builderNullCounts() {
        let fixtures: [[Int64?]] = [[], [1, 2], [nil, 1, nil], [nil, nil]]
        for expected in fixtures {
            var builder = CSVColumnBuilder<Int64>(capacity: expected.count)
            for value in expected { builder.append(value) }
            let column = builder.column(named: "value")
            #expect(column.values == expected)
            #expect(column.nullCount == expected.filter { $0 == nil }.count)
        }
    }

    @Test("Chunked finalization preserves typed values and releases its chunks")
    func chunkedBuilder() {
        var builder = CSVColumnBuilder<String>(capacity: 2, chunked: true)
        let expected: [String?] = ["a", nil, "c", "d", nil]
        for value in expected { builder.append(value) }
        let column = builder.column(named: "value")
        #expect(column.values == expected)
        #expect(column.nullCount == 2)
        builder.append("new")
        let next = builder.column(named: "next")
        #expect(next.values == ["new"])
        #expect(next.nullCount == 0)
        #expect(column.values == expected)
    }

    @Test("CSV null counts include explicit nulls, conversion failures and absent fields")
    func parsedNullCounts() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("csv-null-count-\(UUID().uuidString).csv")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("name,int,double,bool\nvalid,10,1.5,true\nexplicit,NA,NA,NA\ninvalid,not-a-value,not-a-value,not-a-value\nmissing\n".utf8).write(to: url)
        var options = CSVReadOptions()
        options.columnTypeOverrides = ["int": .int64, "double": .float64, "bool": .boolean]
        let frame = try await DataFrame(csv: url, options: options)
        let integers = try #require(frame[column: "int", as: Int64.self])
        let doubles = try #require(frame[column: "double", as: Double.self])
        let booleans = try #require(frame[column: "bool", as: Bool.self])
        let strings = try #require(frame[column: "name", as: String.self])
        #expect(integers.values == [10, nil, nil, nil])
        #expect(doubles.values == [1.5, nil, nil, nil])
        #expect(booleans.values == [true, nil, nil, nil])
        #expect(integers.nullCount == 3)
        #expect(doubles.nullCount == 3)
        #expect(booleans.nullCount == 3)
        #expect(strings.nullCount == 0)
    }

    @Test("Inference preserves incompatible values after the sample as text")
    func lateInvalidValue() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("csv-late-null-\(UUID().uuidString).csv")
        defer { try? FileManager.default.removeItem(at: url) }
        let csv = "value\n" + String(repeating: "42\n", count: 1_001) + "invalid\n"
        try Data(csv.utf8).write(to: url)
        let frame = try await DataFrame(csv: url)
        let column = try #require(frame[column: "value", as: String.self])
        #expect(column.count == 1_002)
        #expect(column.values.prefix(1_001).allSatisfy { $0 == "42" })
        #expect(column.values.last! == "invalid")
        #expect(column.nullCount == 0)
    }
}
