import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV Double storage choices")
struct CSVDoubleStorageModeTests {
    private let csv = "id,x,flag,label\n9007199254740993,1.5,true,alpha\n2,NULL,false,beta\n3,-0.0,true,gamma\n4,NaN,false,delta\n5,inf,true,epsilon\n6,-inf,false,zeta\n"
    private let schema: [String: ColumnDType] = ["id": .int64, "x": .float64, "flag": .boolean, "label": .utf8]

    @Test("Explicit and inferred reads preserve values, layout, selections and export ownership",
          arguments: [CSVColumnConversionMode.serial, .parallel])
    func eager(_ conversion: CSVColumnConversionMode) async throws {
        let url = try writeFixture()
        defer { try? FileManager.default.removeItem(at: url) }
        for explicit in [false, true] {
            var options = CSVReadOptions()
            options.nullValues = ["NULL"]
            options.columnConversionMode = conversion
            if explicit { options.columnTypeOverrides = schema }
            let reference = try await DataFrame(csv: url, options: options)
            let expected = try #require(reference[column: "x", as: Double.self]).values
            for storage in [CSVDoubleStorage.optional, .compact] {
                options.doubleStorage = storage
                let frame = try await DataFrame(csv: url, options: options)
                #expect(frame.dtypes == reference.dtypes)
                #expect(frame[column: "id", as: Int64.self]?[0] == 9_007_199_254_740_993)
                let column = try #require(frame[column: "x", as: Double.self])
                check(column, expected, storage)
                check(try #require(frame.gathered(at: [5, 1, 1, 0])[column: "x", as: Double.self]),
                      [expected[5], expected[1], expected[1], expected[0]], storage)
                let keys = TypedColumn<Int64>(name: "id", values: [9_007_199_254_740_993, 99])
                let joined = try frame.join(DataFrame(columns: [keys]), on: "id", how: .outer)
                check(try #require(joined[column: "x", as: Double.self]), expected + [nil], storage)
                var first = column.values
                let second = column.values
                let shared = first.withUnsafeBufferPointer { lhs in
                    second.withUnsafeBufferPointer { rhs in lhs.baseAddress == rhs.baseAddress }
                }
                #expect(shared == (storage == .optional))
                first[0] = 999
                #expect(column[0] == 1.5 && second[0] == 1.5)
            }
        }
    }

    @Test("Streamed chunks honor the storage choice with a fixed native schema",
          arguments: [CSVColumnConversionMode.serial, .parallel])
    func stream(_ conversion: CSVColumnConversionMode) async throws {
        let url = try writeFixture()
        defer { try? FileManager.default.removeItem(at: url) }
        var options = CSVReadOptions()
        options.nullValues = ["NULL"]
        options.columnTypeOverrides = schema
        options.columnConversionMode = conversion
        let reference = try await DataFrame(csv: url, options: options)
        let expected = try #require(reference[column: "x", as: Double.self]).values
        for storage in [CSVDoubleStorage.optional, .compact] {
            options.doubleStorage = storage
            var offset = 0
            for try await frame in DataFrame.readCSVStream(contentsOf: url, chunkSize: 2, options: options) {
                #expect(frame.dtypes == reference.dtypes)
                let column = try #require(frame[column: "x", as: Double.self])
                check(column, Array(expected[offset..<(offset + frame.rowCount)]), storage)
                if offset == 0 { #expect(frame[column: "id", as: Int64.self]?[0] == 9_007_199_254_740_993) }
                offset += frame.rowCount
            }
            #expect(offset == expected.count)
        }
    }

    private func check(_ column: TypedColumn<Double>, _ expected: [Double?], _ storage: CSVDoubleStorage) {
        #expect((column.compactNumericStorage != nil) == (storage == .compact))
        #expect(column.count == expected.count)
        #expect(column.nullCount == expected.filter { $0 == nil }.count)
        for row in expected.indices { #expect(column[row]?.bitPattern == expected[row]?.bitPattern) }
    }

    private func writeFixture() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("csv-storage-\(UUID().uuidString).csv")
        try Data(csv.utf8).write(to: url)
        return url
    }
}
