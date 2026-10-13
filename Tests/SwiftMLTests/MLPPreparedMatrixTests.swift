import Testing
import SwiftDataFrame
@testable import SwiftML

@Suite("Prepared matrix MLP regression")
struct MLPPreparedMatrixTests {
    @Test(arguments: [ActivationFunction.relu, .sigmoid, .tanh])
    func matchesRowPrediction(_ activation: ActivationFunction) async throws {
        let training = matrixMLPFeatures(18, width: 7)
        let targets: [Double] = training.map { 2 * $0[0] - $0[1] }
        for hidden in [[], [9], [11, 5]] {
            let model = MLPRegressor(hiddenLayerSizes: hidden, activation: activation,
                maxIter: 3, seed: 42, requestedDevice: .cpu)
            try await model.fit(features: training, targets: targets)
            for count in [0, 1, 31, 32, 33, 129] {
                let rows = matrixMLPFeatures(count, width: 7)
                let expected = try await model.predict(features: rows)
                for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
                    let matrix = try matrixMLPInput(rows, width: 7, order: order)
                    let original = matrix.values
                    let result = try await model.predict(features: matrix)
                    matrixMLPExpectClose(result, expected)
                    #expect(matrix.values == original)
                }
            }
        }
    }

    @Test func dimensionAndFittedChecks() async throws {
        let model = MLPRegressor(hiddenLayerSizes: [], maxIter: 1, requestedDevice: .cpu)
        let empty = try matrixMLPInput([], width: 2)
        await #expect(throws: SwiftMLError.self) { try await model.predict(features: empty) }
        let rows: [[Double]] = [[1, 2], [3, 4]]
        try await model.fit(features: rows, targets: [1, 2])
        #expect(try await model.predict(features: empty).isEmpty)
        for count in [1, 33] {
            let bad = try matrixMLPInput(matrixMLPFeatures(count, width: 3), width: 3)
            await #expect(throws: SwiftMLError.self) { try await model.predict(features: bad) }
        }
    }

    @Test func scratchBoundaryAndWideFallback() async throws {
        for width in [512, 513] {
            let rows = matrixMLPFeatures(33, width: width)
            let training = Array(rows.prefix(2))
            let model = MLPRegressor(hiddenLayerSizes: [], maxIter: 1, seed: 42, requestedDevice: .cpu)
            try await model.fit(features: training, targets: [1, 2])
            let expected = try await model.predict(features: rows)
            for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
                let matrix = try matrixMLPInput(rows, width: width, order: order)
                matrixMLPExpectClose(try await model.predict(features: matrix), expected)
            }
        }
    }

    @Test func selectionSnapshotAndRefit() async throws {
        let rows = matrixMLPFeatures(65, width: 7)
        var batch = try matrixMLPBatch(rows, width: 7)
        let indices = [64, 2, 2, 0] + Array(0..<33)
        let selected = try batch.selectingRows(indices)
        let matrix = try selected.matrix(order: .rowMajor, missing: .reject)
        let model = MLPRegressor(hiddenLayerSizes: [9], maxIter: 3, seed: 42, requestedDevice: .cpu)
        try await model.fit(features: rows, targets: Array(repeating: 1, count: rows.count))
        let expected = try await model.predict(features: indices.map { rows[$0] })
        try batch.updateColumn(at: 0, rows: [0], values: [99])
        let actual = try await model.predict(features: matrix)
        matrixMLPExpectClose(actual, expected)
        #expect(matrix.originalRowIndices == indices)
        try await model.fit(features: rows, targets: Array(repeating: -2, count: rows.count))
        let refitted = try await model.predict(features: matrix)
        let reference = try await model.predict(features: indices.map { rows[$0] })
        matrixMLPExpectClose(refitted, reference)
        #expect(refitted != actual)
    }
}
