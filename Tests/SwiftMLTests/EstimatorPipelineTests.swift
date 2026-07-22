import Testing
@testable import SwiftML
@testable import SwiftPreprocessing

@Suite("EstimatorPipeline Tests")
struct EstimatorPipelineTests {

    @Test("ClassificationPipeline fits and predicts probability with StandardScaler")
    func testClassificationPipeline() async throws {
        let X: [[Double]] = [
            [1.0, 2.0],
            [2.0, 3.0],
            [10.0, 20.0],
            [11.0, 21.0]
        ]
        let y: [Double] = [0.0, 0.0, 1.0, 1.0]

        let dt = DecisionTreeClassifier(maxDepth: 3)
        let pipeline = ClassificationPipeline(
            transformers: [StandardScaler()],
            estimator: dt
        )

        try await pipeline.fit(features: X, targets: y)
        let preds = try await pipeline.predict(features: X)
        #expect(preds.count == 4)

        let probs = try await pipeline.predictProbability(features: X)
        #expect(probs.count == 4)
        #expect(probs[0].reduce(0, +) == 1.0)
    }

    @Test("RegressionPipeline fits and predicts with MinMaxScaler")
    func testRegressionPipeline() async throws {
        let X: [[Double]] = [
            [1.0],
            [2.0],
            [3.0],
            [4.0]
        ]
        let y: [Double] = [2.0, 4.0, 6.0, 8.0]

        let lr = LinearRegression(device: .cpu)
        let pipeline = RegressionPipeline(
            transformers: [MinMaxScaler()],
            estimator: lr
        )

        try await pipeline.fit(features: X, targets: y)
        let preds = try await pipeline.predict(features: X)
        #expect(preds.count == 4)
    }
}
