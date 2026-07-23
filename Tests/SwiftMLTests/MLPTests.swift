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

    @Test("MLPRegressor simple training and prediction")
    func testMLPRegressor() async throws {
        let X: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let y: [Double] = [2.0, 4.0, 6.0, 8.0]

        let mlp = MLPRegressor(hiddenLayerSizes: [8], maxIter: 300, learningRate: 0.05, seed: 42)
        try await mlp.fit(features: X, targets: y)
        let preds = try await mlp.predict(features: X)
        #expect(preds.count == 4)
    }
}
