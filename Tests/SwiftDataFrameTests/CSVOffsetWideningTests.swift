import Testing
@testable import SwiftDataFrame

@Suite("CSV record offset widening")
struct CSVOffsetWideningTests {
    @Test("A bounded representation retries without changing record grammar or row limits",
          arguments: [1, 2])
    func widening(_ limit: Int) throws {
        let long = String(repeating: "x", count: 40)
        let fixtures = ["small,first\nnext,last\n", long + ",tail\nnext,last\n",
                        "\"" + long + "\"\"tail\",value\r\nnext,last", long]
        for text in fixtures {
            let bytes = Array(text.utf8)
            try bytes.withUnsafeBufferPointer { buffer in
                let parser = SystemsCSVParser()
                let expected = try parser.scanBatch(buffer: buffer, maximumRows: limit,
                    fieldType: PackedCSVField.self)
                let actual = try parser.scanBatch(buffer: buffer, maximumRows: limit,
                    fieldType: BoundedField.self)
                #expect(actual.nextOffset == expected.nextOffset)
                #expect(actual.records.rowStarts == expected.records.rowStarts)
                for row in 0..<expected.records.count {
                    let fields = expected.records.row(at: row)
                    for column in fields.indices {
                        let field = try #require(actual.records.field(row: row, column: column))
                        #expect(field.startOffset == fields[column].startOffset)
                        #expect(field.length == fields[column].length)
                        #expect(field.escapedQuotesPresent == fields[column].escapedQuotesPresent)
                    }
                }
            }
        }
    }

    private struct BoundedField: CSVIndexField {
        static var maximumEndOffset: Int { 31 }
        let field: PackedCSVField

        init(startOffset: Int, length: Int, escapedQuotesPresent: Bool) {
            field = PackedCSVField(startOffset: startOffset, length: length,
                                   escapedQuotesPresent: escapedQuotesPresent)
        }

        static func recordIndex(blocks: CSVFieldBlocks<Self>, rowStarts: [Int]) -> CSVRecordIndex {
            let fields = blocks.finished().flatMap { block in block.map(\.field) }
            return CSVRecordIndex(fields: fields, rowStarts: rowStarts)
        }
    }
}
