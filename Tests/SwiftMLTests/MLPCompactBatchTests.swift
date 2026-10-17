import Testing
import SwiftDataFrame
@testable import SwiftML

@Suite("Compact batch MLP protocol dispatch")
struct MLPCompactBatchTests {
    @Test(arguments: [ActivationFunction.relu, .sigmoid, .tanh])
    func matchesRowsExactly(_ activation: ActivationFunction) async throws {
        let training = matrixMLPFeatures(8, width: 7)
        for hidden in [[], [9], [11, 5]] {
            let model = MLPRegressor(hiddenLayerSizes: hidden, activation: activation,
                maxIter: 1, seed: 42, requestedDevice: .cpu)
            try await model.fit(features: training, targets: training.map { $0[0] })
            let erased: any RegressorEstimator = model
            for count in [0, 1, 31, 32, 33, 65, 129] {
                let rows = matrixMLPFeatures(count, width: 7)
                let batch = try matrixMLPBatch(rows, width: 7)
                let expected = try await model.predict(features: rows)
                let actual = try await erased.predict(features: batch)
                #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
                #expect(batch.rowValues() == rows)
            }
        }
    }
    @Test func wideAndScratchBoundary() async throws {
        for width in [512, 513] {
            let rows = matrixMLPFeatures(33, width: width)
            let model = MLPRegressor(hiddenLayerSizes: [], maxIter: 1, seed: 42, requestedDevice: .cpu)
            try await model.fit(features: Array(rows.prefix(2)), targets: [1, 2])
            let batch = try matrixMLPBatch(rows, width: width)
            let expected = try await model.predict(features: rows)
            let actual = try await model.predict(features: batch)
            #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
        }
    }
}
