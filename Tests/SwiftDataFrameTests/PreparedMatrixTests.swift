import Testing
@testable import SwiftDataFrame

@Suite("Prepared matrix ownership and layout")
struct PreparedMatrixTests {
    @Test(arguments: [NumericMatrixOrder.rowMajor, .columnMajor])
    func replacementPreservesLayoutProvenanceAndSnapshots(_ order: NumericMatrixOrder) throws {
        let batch = try DataFrame(columns: [
            TypedColumn<Double>(name: "a", values: [1, 2, 3]),
            TypedColumn<Double>(name: "b", values: [10, 20, 30]),
            TypedColumn<Double>(name: "c", values: [100, 200, 300])
        ]).prepareNumericBatch(["a", "b", "c"])
        let matrices = [
            try batch.matrix(order: order),
            try batch.selectingRows([2, 0, 2]).matrix(order: order),
            try batch.selectingRows([]).matrix(order: order)
        ]
        let provenance: [[Int]] = [[0, 1, 2], [2, 0, 2], []]
        for (index, source) in matrices.enumerated() {
            let originalValues = source.values
            let expected: [[Double]] = (0..<source.rowCount).map { row in
                [Double(row * 10 + 1), Double(row * 10 + 2)]
            }
            var values: [Double] = order == .rowMajor
                ? expected.flatMap { $0 }
                : (0..<2).flatMap { column in expected.map { $0[column] } }
            var output = source.replacingValues(values, columnNames: ["PC1", "PC2"])
            let snapshot = output
            #expect(output.rowCount == source.rowCount && output.columnCount == 2)
            #expect(output.columnNames == ["PC1", "PC2"] && output.order == order)
            #expect(output.originalRowIndices == provenance[index])
            for row in expected.indices {
                for column in 0..<2 { #expect(output[row, column] == expected[row][column]) }
            }
            if !values.isEmpty {
                values[0] = -100
                #expect(output[0, 0] == 1)
                output.withMutableValues { $0[0] = -200 }
                #expect(snapshot[0, 0] == 1 && output[0, 0] == -200)
            } else {
                #expect(output.values.isEmpty)
            }
            #expect(source.values == originalValues && source.columnNames == ["a", "b", "c"])
        }
    }

    @Test func selectionCompositionAndLayout() throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Double>(name: "a", values: [1, 2, 3, 4]),
            TypedColumn<Int64>(name: "b", values: [10, 20, 30, 40])
        ])
        let source = try frame.prepareNumericBatch(["a", "b"])
        let selected = try source.selectingRows([3, 1, 1, 0]).selectingRows([2, 0, 1])
        let row = try selected.matrix(), column = try selected.matrix(order: .columnMajor)
        #expect(row.values == [2, 20, 4, 40, 2, 20])
        #expect(column.values == [2, 4, 2, 20, 40, 20])
        #expect(row.originalRowIndices == [1, 3, 1] && column.originalRowIndices == [1, 3, 1])
        for r in 0..<3 { for c in 0..<2 { #expect(row[r,c] == column[r,c]) } }
        #expect(try source.matrix().values == [1, 10, 2, 20, 3, 30, 4, 40])
    }

    @Test func validitySelectionAndPolicies() throws {
        var values = (0..<130).map { Optional(Double($0)) }
        values[63] = nil; values[64] = .nan; values[129] = nil
        let batch = try DataFrame(columns: [TypedColumn<Double>(name: "x", values: values)]).prepareNumericBatch(["x"])
        let selection = try batch.selectingRows([129, 64, 63, 0, 129])
        #expect(selection.nullCount(inColumn: 0) == 3)
        #expect(selection[1,0]?.isNaN == true && selection[2,0] == nil)
        #expect(throws: (any Error).self) { try selection.matrix(missing: .reject) }
        let nan = try batch.selectingRows([64]).matrix(missing: .reject)
        #expect(nan.values[0].isNaN)
        #expect(try batch.selectingRows([]).matrix(missing: .reject).values.isEmpty)
        #expect(throws: (any Error).self) { try batch.selectingRows([-1]) }
        #expect(throws: (any Error).self) { try batch.selectingRows([130]) }
        #expect(batch.nullCount(inColumn: 0) == 2)
    }

    @Test func noColumnsRetainRowCountAndIdentity() throws {
        let batch = try DataFrame(columns: [TypedColumn<Double>(name: "x", values: [1,2,3])]).prepareNumericBatch([])
        let matrix = try batch.selectingRows([2, 0]).matrix(order: .columnMajor)
        #expect(matrix.rowCount == 2 && matrix.columnCount == 0 && matrix.values.isEmpty)
        #expect(matrix.originalRowIndices == [2, 0])
    }
}
