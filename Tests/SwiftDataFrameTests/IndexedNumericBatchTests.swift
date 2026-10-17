import Testing
@testable import SwiftDataFrame

@Suite("Indexed numeric snapshots", .serialized)
struct IndexedNumericBatchTests {
    @Test func editsPreserveOccurrencesValidityAndSnapshots() throws {
        let payload = Double(bitPattern: 0x7ff8_0000_0000_0042)
        var source = try PreparedNumericBatch(columnNames: ["x", "y"],
            columns: [[1, payload, -0.0, 7, .infinity], [10, 11, 12, 13, 14]])
        try source.updateColumn(at: 0, rows: [3], values: [nil])
        let selection = [1, 1, 3, 0, 2]
        var view = try IndexedNumericBatch(source, selectingRows: selection)
        let snapshot = view
        var expected = try source.selectingRows(selection)
        let rows = [0, 0, 1, 4]
        let values: [Double?] = [nil, -0.0, .nan, nil]
        try view.updateColumn(at: 0, rows: rows, values: values)
        try expected.updateColumn(at: 0, rows: rows, values: values)
        check(view, expected)
        #expect(view.editedColumns.count == 1)
        #expect(view.editedColumns[0].data.nullCount == expected.nullCount(inColumn: 0))
        try source.updateColumn(at: 0, rows: [1], values: [99])
        #expect(snapshot[0, 0]?.bitPattern == payload.bitPattern)
        #expect(snapshot[2, 0] == nil)
        let editedSnapshot = view
        try view.updateColumn(at: 1, rows: [2, 2], values: [8, nil])
        try expected.updateColumn(at: 1, rows: [2, 2], values: [8, nil])
        check(view, expected)
        #expect(editedSnapshot[2, 1] == 13)
        #expect(view.originalRowIndices == selection)
    }

    @Test func invalidUpdatesAreAtomicAndProvenanceIsComposed() throws {
        let source = try PreparedNumericBatch(columnNames: ["x"], columns: [[1, 2, 3, 4, 5]])
        let selected = try source.selectingRows([4, 1, 1, 0])
        var view = try IndexedNumericBatch(selected, selectingRows: [2, 0, 2, 3])
        #expect(view.originalRowIndices == [1, 4, 1, 0])
        for (column, rows, values) in [(-1, [0], [9.0]), (1, [0], [9.0]),
            (0, [0, 99], [9.0, 10.0]), (0, [-1], [9.0]), (0, [0], [])] {
            #expect(throws: SwiftMLError.self) {
                try view.updateColumn(at: column, rows: rows, values: values.map(Optional.some))
            }
            #expect(view.editedColumns.isEmpty)
            #expect(view[0, 0] == 2)
        }
        for indices in [[-1], [5]] {
            #expect(throws: SwiftMLError.self) { try IndexedNumericBatch(source, selectingRows: indices) }
        }
        var empty = try IndexedNumericBatch(source, selectingRows: [])
        try empty.updateColumn(at: 0, rows: [], values: [])
        #expect(empty.rowCount == 0 && empty.editedColumns.isEmpty)
    }

    @Test func uneditedColumnsShareTheCapturedSourceStorage() throws {
        let source = try PreparedNumericBatch(columnNames: ["x", "y"],
            columns: [Array(repeating: 1, count: 4096), Array(repeating: 2, count: 4096)])
        var view = try IndexedNumericBatch(source, selectingRows: Array((0..<4096).reversed()))
        try view.updateColumn(at: 0, rows: [0], values: [7])
        source.columns[1].values.withUnsafeBufferPointer { original in
            view.source.columns[1].values.withUnsafeBufferPointer { retained in
                #expect(original.baseAddress == retained.baseAddress)
            }
        }
        #expect(view.editedColumns.count == 1)
        #expect(view.editedColumns[0].data.values.count == 4096)
        #expect(source[4095, 0] == 1 && view[0, 0] == 7)
    }

    private func check(_ view: IndexedNumericBatch, _ expected: PreparedNumericBatch) {
        for c in 0..<expected.columnCount {
            for r in 0..<expected.rowCount { #expect(view[r, c]?.bitPattern == expected[r, c]?.bitPattern) }
        }
    }
}
