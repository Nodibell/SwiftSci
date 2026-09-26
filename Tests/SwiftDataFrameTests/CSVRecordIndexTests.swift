import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("Flat CSV record index")
struct CSVRecordIndexTests {
    @Test("Flat and public records preserve field boundaries and decoded values")
    func recordBoundaries() {
        let fixtures: [(String, [[String]])] = [
            ("", []),
            ("a,b,c\n1,2,3\n", [["a", "b", "c"], ["1", "2", "3"]]),
            ("a,b\r\n1,2\r\n", [["a", "b"], ["1", "2"]]),
            ("a,b\n1\n2,3,4\n", [["a", "b"], ["1"], ["2", "3", "4"]]),
            ("a,b\n\"first\nsecond\",\"a,b\"", [["a", "b"], ["first\nsecond", "a,b"]]),
            ("a,b\n\"a \"\"quote\"\"\",2\n", [["a", "b"], ["a \"quote\"", "2"]]),
            ("a,b\n,\n", [["a", "b"], ["", ""]]),
            ("a,b\n1,2", [["a", "b"], ["1", "2"]]),
            // An unfinished record ending in a delimiter retains the legacy EOF behavior.
            ("a,b\n1,", [["a", "b"]]),
            ("\n\n", [[""], [""]]),
        ]
        for (text, expected) in fixtures {
            Data(text.utf8).withUnsafeBytes { raw in
                let buffer = raw.bindMemory(to: UInt8.self)
                let parser = SystemsCSVParser()
                let index = parser.parseIndex(buffer: buffer)
                let publicRows = parser.parse(buffer: buffer)
                #expect(index.count == expected.count)
                #expect(index.fields.count == expected.reduce(0) { $0 + $1.count })
                #expect(index.rowStarts.first == 0)
                #expect(index.rowStarts.last == index.fields.count)
                #expect(publicRows.count == expected.count)
                for row in expected.indices {
                    let fields = Array(index.row(at: row))
                    #expect(fields.count == expected[row].count)
                    #expect(fields.map { VectorizedByteParsers.parseString(buffer: buffer, offset: $0) } == expected[row])
                    for column in fields.indices {
                        let field = fields[column]
                        let legacy = publicRows[row][column]
                        #expect(field.startOffset == legacy.startOffset)
                        #expect(field.length == legacy.length)
                        #expect(field.escapedQuotesPresent == legacy.escapedQuotesPresent)
                        #expect(index.field(row: row, column: column)?.startOffset == field.startOffset)
                    }
                    #expect(index.field(row: row, column: fields.count) == nil)
                }
                #expect(index.field(row: index.count, column: 0) == nil)
            }
        }
    }

    @Test("Offsets retain quote flags and original byte coordinates")
    func originalByteCoordinates() {
        let data = Data("a,b\r\n\"x\"\"y\",z\r\n".utf8)
        data.withUnsafeBytes { raw in
            let index = SystemsCSVParser().parseIndex(buffer: raw.bindMemory(to: UInt8.self))
            #expect(index.rowStarts == [0, 2, 4])
            #expect(index.fields.map(\.startOffset) == [0, 2, 5, 12])
            #expect(index.fields.map(\.length) == [1, 1, 6, 1])
            #expect(index.fields.map(\.escapedQuotesPresent) == [false, false, true, false])
        }
    }

    @Test("Custom delimiters preserve quoted delimiters")
    func customDelimiter() {
        Data("one;two\n\"a;b\";c\n".utf8).withUnsafeBytes { raw in
            let buffer = raw.bindMemory(to: UInt8.self)
            let index = SystemsCSVParser(delimiterByte: 59).parseIndex(buffer: buffer)
            let values = index.row(at: 1).map {
                VectorizedByteParsers.parseString(buffer: buffer, offset: $0)
            }
            #expect(values == ["a;b", "c"])
        }
    }

    @Test("File and stream readers retain ragged rows, quotes and output ownership")
    func fileAndStream() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let csv = "name,value,note\r\nalpha,10,\"a,b\"\r\nbeta,20\r\ngamma,30,\"line1\nline2\",extra\r\ndelta,40,\"a \"\"quote\"\"\"\r\n"
        try Data(csv.utf8).write(to: url)
        var options = CSVReadOptions()
        options.inferTypes = false
        options.columnTypeOverrides = ["value": .int64]
        let frame = try await DataFrame(csv: url, options: options)
        var chunks = [DataFrame]()
        for try await chunk in DataFrame.readCSVStream(contentsOf: url, chunkSize: 2, options: options) {
            chunks.append(chunk)
        }
        try FileManager.default.removeItem(at: url)

        let names: [String?] = ["alpha", "beta", "gamma", "delta"]
        let values: [Int64?] = [10, 20, 30, 40]
        let notes: [String?] = ["a,b", nil, "line1\nline2", "a \"quote\""]
        #expect(frame.shape.rows == 4)
        #expect(frame.shape.columns == 3)
        #expect(frame[column: "name", as: String.self]?.values == names)
        #expect(frame[column: "value", as: Int64.self]?.values == values)
        #expect(frame[column: "note", as: String.self]?.values == notes)
        #expect(chunks.map { $0.shape.rows } == [2, 2])
        #expect(chunks.flatMap { $0[column: "name", as: String.self]!.values } == names)
        #expect(chunks.flatMap { $0[column: "value", as: Int64.self]!.values } == values)
        #expect(chunks.flatMap { $0[column: "note", as: String.self]!.values } == notes)
    }

    @Test("File limits and headerless readers retain their existing row contract")
    func limitsAndHeaderless() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("a;b\nc\nd;e;f\n".utf8).write(to: url)
        var options = CSVReadOptions()
        options.delimiter = ";"
        options.hasHeader = false
        options.inferTypes = false
        options.maxRows = 2
        let limited = try await DataFrame(csv: url, options: options)
        #expect(limited[column: "col0", as: String.self]?.values == ["a", "c"])
        #expect(limited[column: "col1", as: String.self]?.values == ["b", nil])

        options.maxRows = nil
        var chunks = [DataFrame]()
        for try await chunk in DataFrame.readCSVStream(contentsOf: url, chunkSize: 2, options: options) {
            chunks.append(chunk)
        }
        #expect(chunks.flatMap { $0[column: "col0", as: String.self]!.values } == ["a", "c", "d"])
        #expect(chunks.flatMap { $0[column: "col1", as: String.self]!.values } == ["b", nil, "e"])
    }

    @Test("Empty and header-only files produce empty readers")
    func emptyFiles() async throws {
        for text in ["", "a,b\n"] {
            let url = temporaryURL()
            defer { try? FileManager.default.removeItem(at: url) }
            try Data(text.utf8).write(to: url)
            let frame = try await DataFrame(csv: url)
            #expect(frame.shape.rows == 0)
            var count = 0
            for try await _ in DataFrame.readCSVStream(contentsOf: url, chunkSize: 2) {
                count += 1
            }
            #expect(count == 0)
        }
    }

    private func temporaryURL() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("csv-index-\(UUID().uuidString).csv")
    }
}
