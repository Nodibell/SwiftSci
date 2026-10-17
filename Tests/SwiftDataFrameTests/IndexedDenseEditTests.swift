import Testing
@testable import SwiftDataFrame

@Suite("Indexed dense edits", .serialized)
struct IndexedDenseEditTests {
    @Test func everyColumnCanBeEditedInReverseOrder() throws {
        let names: [String] = (0..<17).map { c -> String in "c\(c)" }
        let columns: [[Double]] = (0..<17).map { column -> [Double] in
            (0..<65).map { row -> Double in
                let value: Int = column * 100 + row
                return Double(value)
            }
        }
        let source = try PreparedNumericBatch(columnNames: names, columns: columns)
        let positions = [64, 2, 2, 0, 33]
        var view = try IndexedNumericBatch(source, selectingRows: positions)
        let snapshot = view
        var gathered = try source.selectingRows(positions)
        for c in (0..<17).reversed() {
            let rows = [0, 1, 1, 3]
            let values: [Double?] = [nil, .nan, -0.0, Double(c)]
            try view.updateColumn(at: c, rows: rows, values: values)
            try gathered.updateColumn(at: c, rows: rows, values: values)
        }
        #expect(view.editedColumns.map(\.index) == Array(0..<17))
        for c in 0..<17 {
            for r in 0..<positions.count {
                #expect(view[r, c]?.bitPattern == gathered[r, c]?.bitPattern)
                #expect(snapshot[r, c] == source[positions[r], c])
            }
        }
        let editedSnapshot = view
        try view.updateColumn(at: 8, rows: [4], values: [nil])
        #expect(editedSnapshot[4, 8] == source[33, 8])
        #expect(view[4, 8] == nil)
    }
}
