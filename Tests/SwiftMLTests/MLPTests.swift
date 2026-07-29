import Testing
import Foundation
@testable import SwiftML

@Suite("MLP Classifier & Regressor Tests")
struct MLPTests {
    @Test("MLPClassifier simple training and prediction")
    func testMLPClassifier() async throws {
        let X: [[Double]] = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ]
        let y: [Double] = [0.0, 1.0, 1.0, 0.0]

        let mlp = MLPClassifier(hiddenLayerSizes: [8], maxIter: 500, learningRate: 0.1, seed: 42)
        try await mlp.fit(features: X, targets: y)
        let preds = try await mlp.predict(features: X)
        #expect(preds.count == 4)
    }

    @Test("MLPClassifier with Adam solver converges on XOR problem")
    func testMLPClassifierAdamConvergence() async throws {
        let X: [[Double]] = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ]
        let y: [Double] = [0.0, 1.0, 1.0, 0.0]

        let mlp = MLPClassifier(
            hiddenLayerSizes: [16],
            activation: .relu,
            solver: .adam,
            maxIter: 600,
            learningRate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            seed: 42
        )
        try await mlp.fit(features: X, targets: y)
        let preds = try await mlp.predict(features: X)
        #expect(preds.count == 4)
        #expect(preds[0] == 0)
        #expect(preds[1] == 1)
        #expect(preds[2] == 1)
        #expect(preds[3] == 0)
    }

    @Test("MLPRegressor simple training and prediction")
    func testMLPRegressor() async throws {
        let X: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let y: [Double] = [2.0, 4.0, 6.0, 8.0]

        let mlp = MLPRegressor(hiddenLayerSizes: [8], maxIter: 300, learningRate: 0.05, seed: 42)
        try await mlp.fit(features: X, targets: y)
        let preds = try await mlp.predict(features: X)
        #expect(preds.count == 4)
    }

    @Test("MLPRegressor with Adam solver converges on linear data")
    func testMLPRegressorAdamConvergence() async throws {
        let X: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let y: [Double] = [2.0, 4.0, 6.0, 8.0, 10.0]

        let mlp = MLPRegressor(
            hiddenLayerSizes: [8],
            activation: .relu,
            solver: .adam,
            maxIter: 500,
            learningRate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            seed: 42
        )
        try await mlp.fit(features: X, targets: y)
        let preds = try await mlp.predict(features: X)
        #expect(preds.count == 5)
        #expect(abs(preds[0] - 2.0) < 0.5)
        #expect(abs(preds[4] - 10.0) < 0.5)
    }
}
