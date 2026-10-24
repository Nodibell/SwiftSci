import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("Scoped CSV field access")
struct CSVScopedAccessTests {
    @Test("Borrowed flat access preserves empty and ragged rows")
    func flatAccess() {
        let fields = [
            PackedCSVField(startOffset: 0, length: 1, escapedQuotesPresent: false),
            PackedCSVField(startOffset: 2, length: 0, escapedQuotesPresent: true),
        ]
        let index = CSVRecordIndex(fields: fields, rowStarts: [0, 1, 1, 2])
        let checked = index.withFlatFields { view in
            for row in 0...index.count {
                for column in 0...2 {
                    compare(view.field(row: row, column: column), index.field(row: row, column: column))
                }
            }
            return true
        }
        #expect(checked == true)
        #expect(index.withCompactFields { _ in true } == nil)
        let empty = CSVRecordIndex(fields: [], rowStarts: [0])
        #expect(empty.withFlatFields { $0.field(row: 0, column: 0) == nil } == true)
    }

    @Test("Compact and wide scans agree across block boundaries and ragged EOF")
    func scanParity() {
        for width in [1, 7, 16_383, 16_384, 16_385] {
            let row = Array(repeating: "\"a\"\"b\"", count: width).joined(separator: ",")
            let text = row + "\n\"short\"\n" + row + "\n" + row + ","
            let bytes = Array(text.utf8)
            bytes.withUnsafeBufferPointer { buffer in
                let parser = SystemsCSVParser()
                let compact = parser.scanIndex(buffer: buffer, range: 0..<buffer.count, fieldType: CompactCSVField.self)
                let wide = parser.scanIndex(buffer: buffer, range: 0..<buffer.count, fieldType: PackedCSVField.self)
                let automatic = parser.parseIndex(buffer: buffer)
                #expect(compact.rowStarts == wide.rowStarts)
                #expect(compact.rowStarts == automatic.rowStarts)
                #expect(compact.withFlatFields { _ in true } == nil)
                #expect(wide.withFlatFields { _ in true } == nil)
                #expect(wide.withCompactFields { _ in true } == nil)
                let checked = compact.withCompactFields { view in
                    for r in 0...compact.count {
                        let count = r < compact.count ? compact.row(at: r).count : 0
                        for c in 0...count {
                            let expected = wide.field(row: r, column: c)
                            compare(view.field(row: r, column: c), expected)
                            compare(automatic.field(row: r, column: c), expected)
                        }
                    }
                    return true
                }
                #expect(checked == true)
                let fullColumn = CSVReader.buildColumn(
                    buffer: buffer, records: wide, startRowIdx: 0, dataRowsToRead: wide.count,
                    colIndex: 0, name: "value", options: CSVReadOptions()
                ) as? TypedColumn<String>
                #expect(fullColumn?.values == ["a\"b", "short", "a\"b"])
            }
        }
    }

    private func compare(_ actual: CSVFieldOffset?, _ expected: CSVFieldOffset?) {
        #expect(actual?.startOffset == expected?.startOffset)
        #expect(actual?.length == expected?.length)
        #expect(actual?.escapedQuotesPresent == expected?.escapedQuotesPresent)
    }
}
