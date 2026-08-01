import Testing
import Foundation
@testable import SwiftML

@Suite("MultiOutput Regressor and Classifier Tests")
struct MultiOutputTests {

    @Test("MultiOutputRegressor parallel fitting and prediction")
    func testMultiOutputRegressor() async throws {
        let X: [[Double]] = [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0]
        ]
        // Target 0: y0 = x0 + x1
        // Target 1: y1 = 2 * x0
        let Y: [[Double]] = [
            [3.0, 2.0],
            [5.0, 4.0],
            [7.0, 6.0],
            [9.0, 8.0]
        ]

        let multiRegressor = MultiOutputRegressor {
            LinearRegression()
        }

        try await multiRegressor.fit(features: X, targets: Y)

        let testX: [[Double]] = [
            [5.0, 6.0]
        ]

        let predictions = try await multiRegressor.predict(features: testX)
        #expect(predictions.count == 1)
        #expect(predictions[0].count == 2)
        #expect(abs(predictions[0][0] - 11.0) < 1e-3)
        #expect(abs(predictions[0][1] - 10.0) < 1e-3)
    }

    @Test("MultiLabelClassifier parallel fitting and prediction")
    func testMultiLabelClassifier() async throws {
        let X: [[Double]] = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0]
        ]
        // Binary labels L0 and L1
        let Y: [[Int]] = [
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0]
        ]

        let multiClassifier = MultiLabelClassifier {
            DecisionTreeClassifier(maxDepth: 3)
        }

        try await multiClassifier.fit(features: X, targets: Y)

        let testX: [[Double]] = [
            [1.0, 0.0],
            [0.0, 1.0]
        ]

        let preds = try await multiClassifier.predict(features: testX)
        #expect(preds.count == 2)
        #expect(preds[0] == [1, 0])
        #expect(preds[1] == [0, 1])
    }
}
