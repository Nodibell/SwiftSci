import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("Packed CSV field storage")
struct CSVFieldStorageTests {
    @Test("Quote flags fit alongside all valid field lengths")
    func boundaryRoundTrips() {
        let coordinates = [0, 1, Int(UInt32.max), Int.max]
        for start in coordinates {
            for length in coordinates {
                for escaped in [false, true] {
                    let field = PackedCSVField(
                        startOffset: start,
                        length: length,
                        escapedQuotesPresent: escaped
                    )
                    #expect(field.startOffset == start)
                    #expect(field.length == length)
                    #expect(field.escapedQuotesPresent == escaped)
                    #expect(field.offset.startOffset == start)
                    #expect(field.offset.length == length)
                    #expect(field.offset.escapedQuotesPresent == escaped)
                }
            }
        }
    }

    @Test("Field storage requires two machine words")
    func storageFootprint() {
        #expect(MemoryLayout<PackedCSVField>.stride == 2 * MemoryLayout<Int>.stride)
        #expect(MemoryLayout<PackedCSVField>.stride < MemoryLayout<CSVFieldOffset>.stride)
    }

    @Test("Packed ragged rows preserve absent fields and escaped empty fields")
    func raggedRows() {
        let fields = [
            PackedCSVField(startOffset: 0, length: 1, escapedQuotesPresent: false),
            PackedCSVField(startOffset: 2, length: 0, escapedQuotesPresent: true),
            PackedCSVField(startOffset: 3, length: 2, escapedQuotesPresent: false),
        ]
        let index = CSVRecordIndex(fields: fields, rowStarts: [0, 2, 2, 3])
        #expect(index.count == 3)
        #expect(!index.isEmpty)
        #expect(index.row(at: 0).map(\.startOffset) == [0, 2])
        #expect(index.row(at: 1).isEmpty)
        #expect(index.row(at: 2).map(\.length) == [2])
        #expect(index.field(row: 0, column: 1)?.length == 0)
        #expect(index.field(row: 0, column: 1)?.escapedQuotesPresent == true)
        #expect(index.field(row: 0, column: 2) == nil)
        #expect(index.field(row: 1, column: 0) == nil)
        #expect(index.field(row: 3, column: 0) == nil)
    }
}
