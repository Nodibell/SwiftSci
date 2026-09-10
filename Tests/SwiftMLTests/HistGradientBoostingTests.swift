import Testing
import Foundation
@testable import SwiftML

@Suite("HistGradientBoosting Tests")
struct HistGradientBoostingTests {

    @Test("HistGradientBoostingClassifier binary classification")
    func testBinaryClassification() async throws {
        let clf = try HistGradientBoostingClassifier(
            nEstimators: 30,
            learningRate: 0.15,
            maxDepth: 4,
            maxBins: 32,
            minSamplesLeaf: 2
        )

        // Linearly separable dataset: cluster around (-2, -2) for class 0, and (2, 2) for class 1
        var X: [[Double]] = []
        var y: [Double] = []

        for i in 0..<30 {
            let offset = Double(i % 5) * 0.1
            X.append([-2.0 + offset, -2.0 - offset])
            y.append(0.0)

            X.append([2.0 + offset, 2.0 - offset])
            y.append(1.0)
        }

        try await clf.fit(features: X, targets: y)

        let testX: [[Double]] = [
            [-2.1, -1.9],
            [-1.8, -2.2],
            [1.9, 2.1],
            [2.2, 1.8]
        ]

        let preds = try await clf.predict(features: testX)
        #expect(preds.count == 4)
        #expect(preds[0] == 0)
        #expect(preds[1] == 0)
        #expect(preds[2] == 1)
        #expect(preds[3] == 1)

        let probs = try await clf.predictProbability(features: testX)
        #expect(probs.count == 4)
        for p in probs {
            #expect(abs((p[0] + p[1]) - 1.0) < 1e-5)
            #expect(p[0] >= 0.0 && p[0] <= 1.0)
            #expect(p[1] >= 0.0 && p[1] <= 1.0)
        }
        #expect(probs[0][0] > probs[0][1]) // Class 0 probability higher
        #expect(probs[2][1] > probs[2][0]) // Class 1 probability higher
    }

    @Test("HistGradientBoostingRegressor fits continuous target")
    func testRegression() async throws {
        let reg = try HistGradientBoostingRegressor(
            nEstimators: 40,
            learningRate: 0.1,
            maxDepth: 4,
            maxBins: 64,
            minSamplesLeaf: 2
        )

        var X: [[Double]] = []
        var y: [Double] = []

        // y = 3 * x0 - 2 * x1 + 5
        for i in 0..<80 {
            let x0 = Double(i) / 10.0
            let x1 = sin(Double(i)) * 2.0
            let target = 3.0 * x0 - 2.0 * x1 + 5.0
            X.append([x0, x1])
            y.append(target)
        }

        try await reg.fit(features: X, targets: y)

        let testX = [
            [2.5, 0.0],
            [5.0, 1.0]
        ]
        let preds = try await reg.predict(features: testX)
        #expect(preds.count == 2)
        // y_expected[0] = 3 * 2.5 - 0 + 5 = 12.5
        // y_expected[1] = 3 * 5.0 - 2 + 5 = 18.0
        #expect(abs(preds[0] - 12.5) < 3.0)
        #expect(abs(preds[1] - 18.0) < 4.0)
    }

    @Test("HistGradientBoosting parameter validations and error handling")
    func testParameterValidation() async {
        #expect(throws: SwiftMLError.self) {
            _ = try HistGradientBoostingClassifier(nEstimators: 0)
        }
        #expect(throws: SwiftMLError.self) {
            _ = try HistGradientBoostingClassifier(learningRate: -0.1)
        }
        #expect(throws: SwiftMLError.self) {
            _ = try HistGradientBoostingClassifier(maxBins: 300)
        }
        #expect(throws: SwiftMLError.self) {
            _ = try HistGradientBoostingRegressor(minSamplesLeaf: 0)
        }

        let clf = try! HistGradientBoostingClassifier(nEstimators: 5)
        await #expect(throws: SwiftMLError.self) {
            _ = try await clf.predict(features: [[1.0, 2.0]])
        }
        await #expect(throws: SwiftMLError.self) {
            try await clf.fit(features: [], targets: [])
        }
    }
}
