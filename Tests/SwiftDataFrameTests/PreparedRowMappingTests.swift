import Testing
@testable import SwiftDataFrame

@Suite("Prepared row mapping metadata")
struct PreparedRowMappingTests {
    @Test func implicitAndSelectedMappings() throws {
        let source = try PreparedNumericBatch(columnNames: ["x"], columns: [[10, 20, 30]])
        #expect(source.originalRowMapping == nil)
        for row in 0..<source.rowCount {
            #expect(source.originalRowIndex(at: row) == row)
        }

        let selected = try source.selectingRows([2, 0, 2])
        #expect(selected.originalRowMapping == [2, 0, 2])
        #expect(selected.originalRowIndex(at: 0) == 2)
        #expect(selected.originalRowIndex(at: 1) == 0)
        #expect(selected.originalRowIndex(at: 2) == 2)
        let composed = try selected.selectingRows([2, 1])
        #expect(composed.originalRowMapping == [2, 0])
        #expect(try source.selectingRows([]).originalRowMapping == [])
        #expect(source.originalRowMapping == nil)
    }

    @Test func mappingAndValuesRetainSnapshots() throws {
        var source = try PreparedNumericBatch(columnNames: ["x"], columns: [[10, 20, 30]])
        var selected = try source.selectingRows([2, 0, 2])
        let snapshot = selected
        var mapping = try #require(selected.originalRowMapping)
        mapping[0] = 99
        try source.updateColumn(at: 0, rows: [2], values: [300])
        try selected.updateColumn(at: 0, rows: [0], values: [400])

        #expect(mapping == [99, 0, 2])
        #expect(selected.originalRowMapping == [2, 0, 2])
        #expect(snapshot.originalRowMapping == [2, 0, 2])
        #expect(snapshot[0, 0] == 30 && snapshot[2, 0] == 30)
        #expect(selected[0, 0] == 400 && selected[2, 0] == 30)
        #expect(source[2, 0] == 300)
        #expect(source.originalRowMapping == nil)
    }
}
