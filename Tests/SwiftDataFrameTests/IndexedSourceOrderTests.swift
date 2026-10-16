import Testing
@testable import SwiftDataFrame

@Suite("Source-ordered indexed occurrences", .serialized)
struct IndexedSourceOrderTests {
    @Test func editsValidityAndProvenanceFollowLogicalOccurrences() throws {
        let payload = Double(bitPattern: 0x7ff8_0000_0000_0042)
        var original = try PreparedNumericBatch(columnNames: ["x", "y"], columns:
            [(0..<8).map(Double.init), [10, payload, 12, 13, 14, 15, 16, 17]])
        try original.updateColumn(at: 0, rows: [1], values: [nil])
        let source = try original.selectingRows([7, 1, 3, 1, 6])
        var view = try IndexedNumericBatch(source, selectingRows: [4, 1, 4, 0, 3, 2, 1])
        try view.updateColumn(at: 0, rows: [0, 2, 4], values: [-0.0, .nan, nil])
        var ordered = try view.sourceOrdered()
        let positions = try #require(ordered.logicalPositions)
        #expect(positions == [3, 1, 6, 5, 4, 0, 2])
        #expect(ordered.batch.sourcePositions == [0, 1, 1, 2, 3, 4, 4])
        #expect(ordered.batch.originalRowIndices == positions.map { view.originalRowIndices[$0] })
        for c in 0..<view.columnCount {
            for r in positions.indices {
                #expect(ordered.batch[r, c]?.bitPattern == view[positions[r], c]?.bitPattern)
            }
        }
        try ordered.batch.updateColumn(at: 0, rows: [0], values: [nil])
        #expect(view[3, 0] == 7)
        #expect(ordered.batch[0, 0] == nil)
    }

    @Test func identityAndEmptyOrdersNeedNoPermutation() throws {
        let source = try PreparedNumericBatch(columnNames: ["x"], columns: [[1, 2, 3, 4]])
        for positions in [[], [0], [0, 1, 1, 3]] {
            let view = try IndexedNumericBatch(source, selectingRows: positions)
            let ordered = try view.sourceOrdered()
            #expect(ordered.logicalPositions == nil)
            #expect(ordered.batch.sourcePositions == positions)
        }
    }
    @Test func anUnsortedTailStaysInItsOriginalPositions() throws {
        let source = try PreparedNumericBatch(columnNames: ["x"], columns: [[1, 2, 3, 4, 5]])
        let view = try IndexedNumericBatch(source, selectingRows: [4, 1, 4, 0, 3, 2, 1])
        let ordered = try view.sourceOrdered(prefixCount: 4)
        #expect(ordered.logicalPositions == [3, 1, 0, 2, 4, 5, 6])
        #expect(ordered.batch.sourcePositions == [0, 1, 4, 4, 3, 2, 1])
        #expect(try view.sourceOrdered(prefixCount: 0).logicalPositions == nil)
        for prefix in [-1, view.rowCount + 1] {
            #expect(throws: SwiftMLError.self) { try view.sourceOrdered(prefixCount: prefix) }
        }
    }

}
