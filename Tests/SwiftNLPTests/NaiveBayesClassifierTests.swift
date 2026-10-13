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

    @Test("NaiveBayesClassifier achieves high predictive accuracy on separable multiclass data")
    func testNaiveBayesHighPredictiveAccuracy() async throws {
        // 3 classes with distinct token features
        let trainFeatures: [[Double]] = [
            [10.0, 8.0, 0.0, 1.0, 0.0, 0.0],
            [9.0, 11.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 12.0, 10.0, 0.0, 0.0],
            [1.0, 0.0, 9.0, 11.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 11.0, 10.0],
            [0.0, 0.0, 1.0, 0.0, 9.0, 12.0]
        ]
        let trainTargets: [Double] = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]

        let classifier = NaiveBayesClassifier(alpha: 1.0)
        try await classifier.fit(features: trainFeatures, targets: trainTargets)

        let testFeatures: [[Double]] = [
            [8.0, 9.0, 0.0, 0.0, 0.0, 0.0],   // Class 0
            [0.0, 0.0, 10.0, 8.0, 0.0, 0.0],  // Class 1
            [0.0, 0.0, 0.0, 0.0, 8.0, 10.0]   // Class 2
        ]
        let preds = try await classifier.predict(features: testFeatures)
        #expect(preds == [0, 1, 2])
    }

    // MARK: - Matrix Validation & Edge Cases for Pure Structs

    @Test("MultinomialNaiveBayes rejects non-rectangular feature matrices and invalid values")
    func testMultinomialNaiveBayesValidation() {
        var mnb = MultinomialNaiveBayes()

        // Non-rectangular
        #expect(throws: SwiftMLError.self) {
            try mnb.fit(X: [[1.0, 2.0, 3.0], [4.0, 5.0]], y: ["a", "b"])
        }

        // NaN
        #expect(throws: SwiftMLError.self) {
            try mnb.fit(X: [[Double.nan, 1.0], [0.0, 1.0]], y: ["a", "b"])
        }

        // Negative value
        #expect(throws: SwiftMLError.self) {
            try mnb.fit(X: [[-1.0, 2.0], [0.0, 1.0]], y: ["a", "b"])
        }
    }

    @Test("MultinomialNaiveBayes preserves previous state on failed refit")
    func testMultinomialNaiveBayesPreservesStateOnFailedRefit() throws {
        var mnb = MultinomialNaiveBayes()
        let validX: [[Double]] = [[5.0, 0.0], [0.0, 5.0]]
        let validY = ["c1", "c2"]
        try mnb.fit(X: validX, y: validY)

        let initialClasses = mnb.classes
        let initialPriors = mnb.classLogPriors

        #expect(throws: SwiftMLError.self) {
            try mnb.fit(X: [[1.0, 2.0], [1.0]], y: ["c1", "c2"])
        }

        #expect(mnb.classes == initialClasses)
        #expect(mnb.classLogPriors == initialPriors)
    }

    @Test("MultinomialNaiveBayes batch predict throws on dimension mismatch")
    func testMultinomialNaiveBayesBatchPredictThrowsOnMismatch() throws {
        var mnb = MultinomialNaiveBayes()
        try mnb.fit(X: [[2.0, 0.0], [0.0, 2.0]], y: ["c1", "c2"])

        #expect(throws: SwiftMLError.self) {
            _ = try mnb.predict(X: [[1.0, 0.0], [1.0, 0.0, 5.0]])
        }

        let preds = try mnb.predict(X: [[1.0, 0.0], [0.0, 1.0]])
        #expect(preds == ["c1", "c2"])
    }

    @Test("ComplementNaiveBayes requires at least 2 distinct classes and validates matrix")
    func testComplementNaiveBayesValidation() {
        var cnb = ComplementNaiveBayes()

        // Single class
        #expect(throws: SwiftMLError.self) {
            try cnb.fit(X: [[1.0, 2.0], [3.0, 4.0]], y: ["onlyOne", "onlyOne"])
        }

        // Ragged
        #expect(throws: SwiftMLError.self) {
            try cnb.fit(X: [[1.0, 2.0], [1.0]], y: ["c1", "c2"])
        }

        // NaN
        #expect(throws: SwiftMLError.self) {
            try cnb.fit(X: [[Double.nan, 2.0], [1.0, 1.0]], y: ["c1", "c2"])
        }

        // Negative
        #expect(throws: SwiftMLError.self) {
            try cnb.fit(X: [[-0.5, 2.0], [1.0, 1.0]], y: ["c1", "c2"])
        }
    }

    @Test("ComplementNaiveBayes batch predict throws on dimension mismatch")
    func testComplementNaiveBayesBatchPredictThrowsOnMismatch() throws {
        var cnb = ComplementNaiveBayes()
        try cnb.fit(X: [[5.0, 0.0], [0.0, 5.0]], y: ["c1", "c2"])

        #expect(throws: SwiftMLError.self) {
            _ = try cnb.predict(X: [[1.0, 0.0], [1.0]])
        }

        let preds = try cnb.predict(X: [[4.0, 0.0], [0.0, 4.0]])
        #expect(preds == ["c1", "c2"])
    }
}


