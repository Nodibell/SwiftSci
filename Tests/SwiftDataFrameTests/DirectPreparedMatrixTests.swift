import Testing
import SwiftDataFrame

@Suite("Direct dataframe matrix extraction")
struct DirectPreparedMatrixTests {
    @Test func selectionLayoutAndConversion() throws {
        let special = Double(bitPattern: 0x7ff800000000002a)
        let doubles: [Double?] = [-0.0, special, .infinity, nil]
        let integers: [Int64?] = [9_007_199_254_740_993, .max, .min, nil]
        let booleans: [Bool?] = [false, true, nil, true]
        let frame = try DataFrame(columns: [
            TypedColumn<Double>(name: "d", values: doubles),
            TypedColumn<Int64>(name: "i", values: integers),
            TypedColumn<Bool>(name: "b", values: booleans)
        ])
        let names = ["b", "i", "d", "b"]
        let selections: [[Int]?] = [nil, [], [3, 1, 1, 0, 2]]
        for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
            for rows in selections {
                let actual = try frame.prepareNumericMatrix(names, rows: rows, order: order)
                var batch = try frame.prepareNumericBatch(names)
                if let rows { batch = try batch.selectingRows(rows) }
                let expected = try batch.matrix(order: order)
                #expect(actual.columnNames == names)
                #expect(actual.rowCount == (rows?.count ?? 4))
                #expect(actual.originalRowIndices == expected.originalRowIndices)
                #expect(actual.values.map(\.bitPattern) == expected.values.map(\.bitPattern))
                for r in 0..<actual.rowCount {
                    for c in 0..<actual.columnCount {
                        #expect(actual[r, c].bitPattern == expected[r, c].bitPattern)
                    }
                }
            }
        }
    }

    @Test func emptyShapesPreserveSchemaAndRows() throws {
        let frame = try DataFrame(columns: [TypedColumn<Double>(name: "x", values: [1, 2, 3])])
        for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
            let zeroRows = try frame.prepareNumericMatrix(["x", "x"], rows: [], order: order)
            #expect(zeroRows.rowCount == 0 && zeroRows.columnNames == ["x", "x"])
            #expect(zeroRows.values.isEmpty && zeroRows.originalRowIndices.isEmpty)
            let zeroColumns = try frame.prepareNumericMatrix([], rows: [2, 0, 2], order: order)
            #expect(zeroColumns.rowCount == 3 && zeroColumns.columnCount == 0)
            #expect(zeroColumns.values.isEmpty && zeroColumns.originalRowIndices == [2, 0, 2])
            let allRows = try frame.prepareNumericMatrix([], order: order)
            #expect(allRows.rowCount == 3 && allRows.originalRowIndices == [0, 1, 2])
            let empty = try DataFrame(columns: [TypedColumn<Double>(name: "x", values: [Double]())])
            let result = try empty.prepareNumericMatrix(["x"], order: order, missing: .reject)
            #expect(result.rowCount == 0 && result.columnCount == 1 && result.values.isEmpty)
        }
    }

    @Test func snapshotSurvivesSourceReplacement() throws {
        var frame = try DataFrame(columns: [TypedColumn<Double>(name: "x", values: [1, 2, 3])])
        let result = try frame.prepareNumericMatrix(["x"], rows: [2, 0])
        let snapshot = result
        frame = try frame.withColumn("x", column: TypedColumn<Double>(name: "x", values: [9, 9, 9]))
        #expect(result.values == [3, 1] && snapshot.values == [3, 1])
        #expect(try frame.prepareNumericMatrix(["x"]).values == [9, 9, 9])
    }
}
