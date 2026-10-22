import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("Eager scoped and indexed CSV parity")
struct CSVEagerBatchParityTests {
    @Test("Batch boundaries preserve mixed native values and original token replay", arguments: [false, true], [CSVColumnConversionMode.parallel, .serial])
    func mixedReplay(_ headerless: Bool, _ mode: CSVColumnConversionMode) throws {
        var options = CSVReadOptions()
        options.columnConversionMode = mode
        options.delimiter = ";"
        options.hasHeader = !headerless
        let header = headerless ? "" : "id;date;label;flag;value\r\n"
        let line = "001;2020-01-02;\"alpha;\n\"\"beta\"\"\";1;2.5\r\n"
        let csv = header + String(repeating: line, count: 70_000) + "late;bad;tail;true;3.25"
        try compare(csv, options: options)
    }

    @Test("Explicit types, numeric null tokens and ragged rows agree", arguments: [0, 10_001, 70_001], [CSVColumnConversionMode.parallel, .serial])
    func explicitTypes(_ limit: Int, _ mode: CSVColumnConversionMode) throws {
        var options = CSVReadOptions()
        options.columnConversionMode = mode
        options.maxRows = limit
        options.columnTypeOverrides = ["id": .int64, "date": .date32, "flag": .boolean, "value": .float64]
        options.columnDateFormats = ["date": "d/M/yyyy"]
        options.nullValues = ["", "7", "0.25"]
        let header = "id,date,flag,value,label\n"
        let line = "7,2/1/2020,true,0.25,text\n"
        let csv = header + String(repeating: line, count: 70_000) + "bad,invalid,maybe,not-a-number\n"
        try compare(csv, options: options)
    }

    private func compare(_ csv: String, options: CSVReadOptions) throws {
        try Array(csv.utf8).withUnsafeBufferPointer { buffer in
            let parser = SystemsCSVParser(delimiterByte: UInt8(options.delimiter.utf8.first!))
            let records = parser.parseIndex(buffer: buffer)
            let start = options.hasHeader ? 1 : 0
            let names = options.hasHeader ? records.row(at: 0).map {
                VectorizedByteParsers.parseString(buffer: buffer, offset: $0)
            } : (0..<records.row(at: 0).count).map { "col\($0)" }
            let count = min(records.count - start, options.maxRows ?? Int.max)
            let expected = try CSVReader.buildFrame(buffer: buffer, records: records,
                startRow: start, rowCount: count, headers: names, options: options)
            let actual = try CSVEagerReader.read(buffer: buffer, options: options)
            #expect(actual.columnNames == expected.columnNames)
            #expect(actual.rowCount == expected.rowCount)
            #expect(actual.dtypes == expected.dtypes)
            for name in names {
                check(Int64.self, name, actual, expected)
                check(Double.self, name, actual, expected)
                check(Bool.self, name, actual, expected)
                check(Date.self, name, actual, expected)
                check(String.self, name, actual, expected)
            }
        }
    }

    private func check<T: SupportedType>(_ type: T.Type, _ name: String, _ actual: DataFrame, _ expected: DataFrame) {
        #expect(actual[column: name, as: type]?.values == expected[column: name, as: type]?.values)
        #expect(actual[column: name, as: type]?.nullCount == expected[column: name, as: type]?.nullCount)
    }
}
