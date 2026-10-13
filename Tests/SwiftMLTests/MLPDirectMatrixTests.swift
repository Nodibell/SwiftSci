import Testing
import SwiftDataFrame
import SwiftML

@Suite("MLP predictions from direct dataframe extraction")
struct MLPDirectMatrixTests {
    @Test func selectedPredictionsMatchRows() async throws {
        let rows = matrixMLPFeatures(65, width: 7)
        let names: [String] = (0..<7).map { "x\($0)" }
        let columns: [TypedColumn<Double>] = (0..<7).map { c in
            let values: [Double] = rows.map { $0[c] }
            return TypedColumn<Double>(name: names[c], values: values)
        }
        let frame = try DataFrame(columns: columns.reversed())
        let model = MLPRegressor(hiddenLayerSizes: [9], maxIter: 2, seed: 42, requestedDevice: .cpu)
        let targets: [Double] = rows.map { $0[0] - $0[2] }
        try await model.fit(features: rows, targets: targets)
        let selections: [[Int]] = [[], [64, 1, 1], [64, 2, 2, 0] + Array(0..<33)]
        for selected in selections {
            let expectedRows: [[Double]] = selected.map { rows[$0] }
            let expected = try await model.predict(features: expectedRows)
            for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
                let matrix = try frame.prepareNumericMatrix(names, rows: selected, order: order, missing: .reject)
                matrixMLPExpectClose(try await model.predict(features: matrix), expected)
                #expect(matrix.originalRowIndices == selected)
            }
        }
    }
}
