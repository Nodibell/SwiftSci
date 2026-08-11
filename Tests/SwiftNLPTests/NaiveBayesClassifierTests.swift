import Testing
import Foundation
import SwiftML
@testable import SwiftNLP

@Suite("NaiveBayesClassifier Actor Estimator Tests")
struct NaiveBayesClassifierTests {

    @Test("NaiveBayesClassifier fits and predicts under ClassifierEstimator protocol")
    func testNaiveBayesClassifier() async throws {
        let features: [[Double]] = [
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 1.0, 3.0],
            [0.0, 0.0, 4.0]
        ]
        let targets: [Double] = [0.0, 0.0, 1.0, 1.0]

        let classifier: any ClassifierEstimator = NaiveBayesClassifier(alpha: 1.0)
        try await classifier.fit(features: features, targets: targets)

        let testFeatures: [[Double]] = [
            [2.5, 0.5, 0.0],
            [0.0, 0.5, 3.5]
        ]

        let preds = try await classifier.predict(features: testFeatures)
        #expect(preds.count == 2)
        #expect(preds[0] == 0)
        #expect(preds[1] == 1)

        let probs = try await classifier.predictProbability(features: testFeatures)
        #expect(probs.count == 2)
        #expect(probs[0][0] > probs[0][1])
        #expect(probs[1][1] > probs[1][0])
    }

    @Test("ComplementNaiveBayesClassifier fits and predicts under ClassifierEstimator protocol")
    func testComplementNaiveBayesClassifier() async throws {
        let features: [[Double]] = [
            [5.0, 0.0],
            [4.0, 1.0],
            [0.0, 5.0],
            [1.0, 4.0]
        ]
        let targets: [Double] = [0.0, 0.0, 1.0, 1.0]

        let classifier: any ClassifierEstimator = ComplementNaiveBayesClassifier(alpha: 1.0)
        try await classifier.fit(features: features, targets: targets)

        let testFeatures: [[Double]] = [
            [4.0, 0.0],
            [0.0, 4.0]
        ]

        let preds = try await classifier.predict(features: testFeatures)
        #expect(preds.count == 2)
        #expect(preds[0] == 0)
        #expect(preds[1] == 1)
    }
}
