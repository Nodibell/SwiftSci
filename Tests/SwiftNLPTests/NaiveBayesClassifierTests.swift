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

    @Test("NaiveBayesClassifier handles error edge cases and unfitted predictions")
    func testNaiveBayesClassifierEdgeCases() async throws {
        let classifier = NaiveBayesClassifier(alpha: -1.0)
        #expect(await classifier.alpha == 0.0)

        await #expect(throws: SwiftMLError.self) {
            try await classifier.predict(features: [[1.0, 2.0]])
        }

        await #expect(throws: SwiftMLError.self) {
            try await classifier.fit(features: [], targets: [])
        }

        await #expect(throws: SwiftMLError.self) {
            try await classifier.fit(features: [[1.0, 2.0]], targets: [0.0, 1.0])
        }

        try await classifier.fit(features: [[1.0, 2.0]], targets: [0.0])
        let emptyPreds = try await classifier.predictProbability(features: [])
        #expect(emptyPreds.isEmpty)

        let mismatchProbs = try await classifier.predictProbability(features: [[1.0]])
        #expect(mismatchProbs.count == 1)
    }

    @Test("ComplementNaiveBayesClassifier handles error edge cases and unfitted predictions")
    func testComplementNaiveBayesClassifierEdgeCases() async throws {
        let classifier = ComplementNaiveBayesClassifier(alpha: -1.0)
        #expect(await classifier.alpha == 0.0)

        await #expect(throws: SwiftMLError.self) {
            try await classifier.predict(features: [[1.0, 2.0]])
        }

        await #expect(throws: SwiftMLError.self) {
            try await classifier.fit(features: [], targets: [])
        }

        await #expect(throws: SwiftMLError.self) {
            try await classifier.fit(features: [[1.0, 2.0]], targets: [0.0, 1.0])
        }

        try await classifier.fit(features: [[1.0, 2.0]], targets: [0.0])
        let emptyPreds = try await classifier.predictProbability(features: [])
        #expect(emptyPreds.isEmpty)

        let mismatchProbs = try await classifier.predictProbability(features: [[1.0]])
        #expect(mismatchProbs.count == 1)
    }
}

