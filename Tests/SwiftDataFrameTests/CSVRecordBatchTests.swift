import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV scoped record batches")
struct CSVRecordBatchTests {
    @Test("Batch boundaries preserve quoted records and EOF", arguments: [1, 2, 7])
    func records(_ size: Int) throws {
        let text = "id,value\r\n1,\"line\nwith \"\"quotes\"\"\"\r\n2,plain\r\n3,\"last\""
        let bytes = Array(text.utf8)
        let expected = [["id", "value"], ["1", "line\nwith \"quotes\""], ["2", "plain"], ["3", "last"]]
        var actual = [[String]]()
        try bytes.withUnsafeBufferPointer { buffer in
            var offset = 0
            let parser = SystemsCSVParser()
            while offset < buffer.count {
                let remaining = UnsafeBufferPointer(rebasing: buffer[offset...])
                let batch = try parser.scanBatch(buffer: remaining, maximumRows: size)
                #expect(batch.records.count <= size)
                #expect(batch.nextOffset > 0)
                for row in 0..<batch.records.count {
                    actual.append(batch.records.row(at: row).map {
                        VectorizedByteParsers.parseString(buffer: remaining, offset: $0)
                    })
                }
                offset += batch.nextOffset
            }
        }
        #expect(actual == expected)
    }

    @Test("Unfinished trailing fields retain the full parser's behavior")
    func unfinishedEOF() throws {
        for text in ["a,b\nlast,", "a,b\n", "", "a,b\n\"unterminated,tail"] {
            let bytes = Array(text.utf8)
            try bytes.withUnsafeBufferPointer { buffer in
                let parser = SystemsCSVParser()
                let full = parser.parse(buffer: buffer)
                var offset = 0
                var rows = [[String]]()
                while offset < buffer.count {
                    let remaining = UnsafeBufferPointer(rebasing: buffer[offset...])
                    let batch = try parser.scanBatch(buffer: remaining, maximumRows: 1)
                    for row in 0..<batch.records.count {
                        rows.append(batch.records.row(at: row).map {
                            VectorizedByteParsers.parseString(buffer: remaining, offset: $0)
                        })
                    }
                    offset += batch.nextOffset
                }
                let expected = full.map { row in row.map { VectorizedByteParsers.parseString(buffer: buffer, offset: $0) } }
                #expect(rows == expected)
            }
        }
    }
    @Test("Large escaped UTF-8 fields cross scan checkpoints", arguments: ["\"", "é\""])
    func largeEscapedField(_ token: String) throws {
        let value = String(repeating: token, count: 40_000)
        let escaped = value.replacingOccurrences(of: "\"", with: "\"\"")
        let bytes = Array(("\"" + escaped + "\"\nnext\n").utf8)
        try bytes.withUnsafeBufferPointer { buffer in
            let batch = try SystemsCSVParser().scanBatch(buffer: buffer, maximumRows: 1)
            #expect(batch.records.count == 1)
            #expect(batch.nextOffset < bytes.count)
            let offset = try #require(batch.records.field(row: 0, column: 0))
            #expect(VectorizedByteParsers.parseString(buffer: buffer, offset: offset) == value)
        }
    }

    @Test("Quotes and record boundaries span SIMD blocks", arguments: Array(0..<34))
    func vectorBoundaries(_ padding: Int) throws {
        let lead = String(repeating: "x", count: padding)
        let escaped = "\"" + lead + "\"\"tail\nline\",value\r\nnext,last"
        let expected = [[lead + "\"tail\nline", "value"], ["next", "last"]]
        let bytes = Array(escaped.utf8)
        try bytes.withUnsafeBufferPointer { buffer in
            let parser = SystemsCSVParser()
            let full = parser.parse(buffer: buffer)
            let batch = try parser.scanBatch(buffer: buffer, maximumRows: 2)
            let decoded = (0..<batch.records.count).map { row -> [String] in
                batch.records.row(at: row).map {
                    VectorizedByteParsers.parseString(buffer: buffer, offset: $0)
                }
            }
            #expect(decoded == expected)
            #expect(full.map { $0.map { VectorizedByteParsers.parseString(buffer: buffer, offset: $0) } } == expected)
        }
    }

    @Test("Long fields cross cancellation windows without changing framing", arguments: [65_535, 65_536, 65_537])
    func longPlainField(_ length: Int) throws {
        let bytes = Array((String(repeating: "x", count: length) + "\nnext\n").utf8)
        try bytes.withUnsafeBufferPointer { buffer in
            let batch = try SystemsCSVParser().scanBatch(buffer: buffer, maximumRows: 1)
            #expect(batch.records.count == 1)
            #expect(batch.nextOffset == length + 1)
            #expect(batch.records.field(row: 0, column: 0)?.length == length)
        }
    }

}
