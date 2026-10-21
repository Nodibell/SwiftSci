import Testing
@testable import SwiftDataFrame

@Suite("Compact CSV field storage")
struct CSVCompactStorageTests {
    @Test("Compact fields use eight bytes and preserve boundary values")
    func representation() {
        #expect(MemoryLayout<CompactCSVField>.stride == 8)
        let limit = Int(Int32.max)
        for start in [0, 1, limit] {
            for length in [0, 1, limit] {
                for escaped in [false, true] {
                    let field = CompactCSVField(
                        startOffset: start, length: length, escapedQuotesPresent: escaped
                    ).offset
                    #expect(field.startOffset == start)
                    #expect(field.length == length)
                    #expect(field.escapedQuotesPresent == escaped)
                }
            }
        }
    }

    @Test("Selection uses the absolute range end, including the length limit")
    func selectionLimit() {
        for end in [0, 1, Int(Int32.max)] {
            #expect(CompactCSVField.canRepresent(endOffset: end))
        }
        for end in [-1, Int(Int32.max) + 1, Int(UInt32.max), Int.max] {
            #expect(!CompactCSVField.canRepresent(endOffset: end))
        }
    }

    @Test("Block truncation and reuse preserve both field widths")
    func blockTruncation() {
        checkBlocks(CompactCSVField.self)
        checkBlocks(PackedCSVField.self)
    }

    private func checkBlocks<Field: CSVIndexField>(_ type: Field.Type) {
        for target in [0, 1, 16_383, 16_384, 16_385, 32_768, 32_769] {
            var blocks = CSVFieldBlocks<Field>(byteCount: 0)
            for i in 0..<32_769 {
                blocks.append(Field(startOffset: i, length: i % 3, escapedQuotesPresent: i % 2 == 0))
            }
            blocks.truncate(to: target)
            #expect(blocks.count == target)
            blocks.append(Field(startOffset: 99, length: 2, escapedQuotesPresent: true))
            let index = Field.recordIndex(blocks: blocks, rowStarts: [0, blocks.count])
            #expect(index.row(at: 0).count == target + 1)
            if target > 0 {
                #expect(index.field(row: 0, column: target - 1)?.startOffset == target - 1)
            }
            #expect(index.field(row: 0, column: target)?.startOffset == 99)
            #expect(index.field(row: 0, column: target)?.escapedQuotesPresent == true)
            #expect(index.field(row: 0, column: target + 1) == nil)
        }
        let empty = CSVFieldBlocks<Field>(byteCount: 0)
        let index = Field.recordIndex(blocks: empty, rowStarts: [0])
        #expect(index.isEmpty)
        #expect(index.field(row: 0, column: 0) == nil)
    }
}
