import Testing
@testable import SwiftDataFrame

@Suite("Prepared supervised snapshots")
struct PreparedSupervisedBatchTests {
    @Test func selectionKeepsTargetsAlignedAndPreservesSnapshot() throws {
        var batch = try PreparedNumericBatch(
            columnNames: ["a", "target", "b"], columns: [[1, 2, 3], [10, 20, 30], [4, 5, 6]]
        )
        let selected = try PreparedSupervisedBatch(batch, targetColumn: "target").selectingRows([2, 0, 2])
        try batch.updateColumn(at: 0, rows: [2], values: [99])
        #expect(selected.features.columnNames == ["a", "b"])
        #expect(selected.features.rowValues() == [[3, 6], [1, 4], [3, 6]])
        #expect(selected.targets == [30, 10, 30])
        #expect(selected.features.originalRowIndices == [2, 0, 2])
        let again = try selected.selectingRows([1, 2])
        #expect(again.features.originalRowIndices == [0, 2])
        #expect(again.targets == [10, 30])
        #expect(throws: SwiftMLError.invalidParameter("Row selection contains an out-of-bounds index")) {
            try selected.selectingRows([3])
        }
    }

    @Test func invalidSupervisedSchemasAreRejected() throws {
        let expected = SwiftMLError.invalidParameter("Supervised batch requires unique feature and target names")
        let single = try PreparedNumericBatch(columnNames: ["target"], columns: [[1]])
        let duplicate = try PreparedNumericBatch(columnNames: ["x", "x"], columns: [[1], [2]])
        let valid = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[1], [2]])
        #expect(throws: expected) { try PreparedSupervisedBatch(single, targetColumn: "target") }
        #expect(throws: expected) { try PreparedSupervisedBatch(duplicate, targetColumn: "x") }
        #expect(throws: expected) { try PreparedSupervisedBatch(valid, targetColumn: "missing") }
    }

    @Test func emptySelectionRetainsSchema() throws {
        let batch = try PreparedNumericBatch(columnNames: ["target", "x"], columns: [[1], [2]])
        let empty = try PreparedSupervisedBatch(batch, targetColumn: "target").selectingRows([])
        #expect(empty.features.columnNames == ["x"])
        #expect(empty.features.rowCount == 0)
        #expect(empty.features.rowValues().isEmpty)
        #expect(empty.features.originalRowIndices.isEmpty)
        #expect(empty.targets.isEmpty)
    }

    @Test func rowFallbackAndFiniteValidationPreserveMissingSemantics() throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Double>(name: "x", values: [1, nil, .nan, .infinity]),
            TypedColumn<Double>(name: "target", values: [2, 3, nil, 5])
        ])
        let batch = try frame.prepareNumericBatch(["x", "target"])
        let rows = batch.rowValues()
        #expect(rows[0] == [1, 2])
        #expect(rows[1][0].isNaN && rows[1][1] == 3)
        #expect(rows[2][0].isNaN && rows[2][1].isNaN)
        #expect(rows[3][0] == .infinity && rows[3][1] == 5)
        let supervised = try PreparedSupervisedBatch(batch, targetColumn: "target")
        #expect(supervised.targets[2].isNaN)
        #expect(supervised.features.nullCount(inColumn: 0) == 1)
        let expected = SwiftMLError.invalidParameter("Regression input contains missing or nonfinite values")
        #expect(throws: expected) { try batch.requireFinite() }
        for value in [Double.nan, .infinity, -.infinity] {
            let nonfinite = try PreparedNumericBatch(columnNames: ["x"], columns: [[value]])
            #expect(throws: expected) { try nonfinite.requireFinite() }
        }
        try batch.selectingRows([0]).requireFinite()
    }

    @Test func replacingRowsPreservesProvenanceAndChecksShape() throws {
        let batch = try PreparedNumericBatch(columnNames: ["a", "b"], columns: [[1, 2, 3], [4, 5, 6]])
            .selectingRows([2, 0, 2])
        let replaced = try batch.replacingRows([[30, 60], [10, 40], [30, 60]])
        #expect(replaced.columnNames == batch.columnNames)
        #expect(replaced.originalRowIndices == [2, 0, 2])
        #expect(replaced.rowValues() == [[30, 60], [10, 40], [30, 60]])
        #expect(batch.rowValues() == [[3, 6], [1, 4], [3, 6]])
        let expected = SwiftMLError.invalidParameter("Prepared transformer must preserve row count and feature schema")
        #expect(throws: expected) { try batch.replacingRows([[1, 2]]) }
        #expect(throws: expected) { try batch.replacingRows([[1], [2], [3]]) }
        let empty = try batch.selectingRows([]).replacingRows([])
        #expect(empty.rowCount == 0 && empty.columnNames == ["a", "b"])
    }
}
