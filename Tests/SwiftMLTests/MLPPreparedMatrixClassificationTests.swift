import Testing
import SwiftDataFrame
@testable import SwiftML

@Suite("Prepared matrix MLP classification")
struct MLPPreparedMatrixClassificationTests {
    @Test(arguments: [ActivationFunction.relu, .sigmoid, .tanh])
    func probabilitiesAndLabels(_ activation: ActivationFunction) async throws {
        let training = matrixMLPFeatures(18, width: 7)
        for labels in [[-3.0, 7.0], [-3.0, 7.0, 12.0]] {
            let model = MLPClassifier(hiddenLayerSizes: [11, 5], activation: activation,
                maxIter: 3, seed: 42, requestedDevice: .cpu)
            let targets: [Double] = training.indices.map { labels[$0 % labels.count] }
            try await model.fit(features: training, targets: targets)
            for count in [0, 1, 31, 32, 33, 129] {
                let rows = matrixMLPFeatures(count, width: 7)
                let probabilities = try await model.predictProbability(features: rows)
                let expectedLabels = try await model.predict(features: rows)
                for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
                    let matrix = try matrixMLPInput(rows, width: 7, order: order)
                    let actual = try await model.predictProbability(features: matrix)
                    #expect(actual.count == probabilities.count)
                    for (a, b) in zip(actual, probabilities) { matrixMLPExpectClose(a, b) }
                    #expect(try await model.predict(features: matrix) == expectedLabels)
                }
            }
        }
    }

    @Test func rejectsUnfittedAndWrongWidth() async throws {
        let model = MLPClassifier(hiddenLayerSizes: [], maxIter: 1, seed: 42, requestedDevice: .cpu)
        let empty = try matrixMLPInput([], width: 2)
        await #expect(throws: SwiftMLError.self) { try await model.predict(features: empty) }
        await #expect(throws: SwiftMLError.self) { try await model.predictProbability(features: empty) }
        try await model.fit(features: [[1, 2], [3, 4]], targets: [-3, 7])
        #expect(try await model.predictProbability(features: empty).isEmpty)
        for count in [1, 33] {
            let bad = try matrixMLPInput(matrixMLPFeatures(count, width: 3), width: 3)
            await #expect(throws: SwiftMLError.self) { try await model.predict(features: bad) }
            await #expect(throws: SwiftMLError.self) { try await model.predictProbability(features: bad) }
        }
    }
}
