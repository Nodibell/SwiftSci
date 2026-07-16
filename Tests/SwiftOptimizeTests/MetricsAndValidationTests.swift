import Testing
import Foundation
@testable import SwiftOptimize
@testable import SwiftML

// MARK: - Metrics Tests

@Suite("Metrics Tests")
struct MetricsTests {

    @Test("Accuracy is correct for binary classification")
    func testAccuracy() {
        let yTrue = [0, 0, 1, 1, 1]
        let yPred = [0, 1, 1, 1, 0]
        // Correct: indices 0, 2, 3 → 3/5 = 0.6
        #expect(abs(Metrics.accuracy(yTrue: yTrue, yPred: yPred) - 0.6) < 1e-9)
    }

    @Test("Precision is correct")
    func testPrecision() {
        let yTrue = [1, 1, 0, 0, 1]
        let yPred = [1, 0, 1, 0, 1]
        // TP=2, FP=1 → precision(label:1) = 2/3
        let p = Metrics.precision(yTrue: yTrue, yPred: yPred, label: 1)
        #expect(abs(p - 2.0/3.0) < 1e-9)
    }

    @Test("Recall is correct")
    func testRecall() {
        let yTrue = [1, 1, 0, 0, 1]
        let yPred = [1, 0, 1, 0, 1]
        // TP=2, FN=1 → recall(label:1) = 2/3
        let r = Metrics.recall(yTrue: yTrue, yPred: yPred, label: 1)
        #expect(abs(r - 2.0/3.0) < 1e-9)
    }

    @Test("F1 score is correct")
    func testF1Score() {
        let yTrue = [1, 1, 1, 0, 0]
        let yPred = [1, 1, 0, 0, 1]
        // p = 2/3, r = 2/3 → f1 = 2/3
        let f1 = Metrics.f1Score(yTrue: yTrue, yPred: yPred, label: 1)
        #expect(abs(f1 - 2.0/3.0) < 1e-9)
    }

    @Test("Perfect classifier yields accuracy 1.0 and F1 1.0")
    func testPerfectClassifier() {
        let yTrue = [0, 1, 2, 0, 1]
        let yPred = [0, 1, 2, 0, 1]
        #expect(Metrics.accuracy(yTrue: yTrue, yPred: yPred) == 1.0)
        #expect(Metrics.f1Score(yTrue: yTrue, yPred: yPred, label: 1) == 1.0)
    }

    @Test("MSE is zero for perfect predictions")
    func testMSEZero() {
        let y = [1.0, 2.0, 3.0]
        #expect(Metrics.meanSquaredError(yTrue: y, yPred: y) == 0.0)
    }

    @Test("MSE and RMSE are correct")
    func testMSEandRMSE() {
        let yTrue = [3.0, -0.5, 2.0, 7.0]
        let yPred = [2.5,  0.0, 2.0, 8.0]
        let mse = Metrics.meanSquaredError(yTrue: yTrue, yPred: yPred)
        // (0.25 + 0.25 + 0 + 1.0) / 4 = 0.375
        #expect(abs(mse - 0.375) < 1e-9)
        #expect(abs(Metrics.rootMeanSquaredError(yTrue: yTrue, yPred: yPred) - sqrt(0.375)) < 1e-9)
    }

    @Test("R2 score is 1.0 for perfect fit")
    func testR2Perfect() {
        let y = [1.0, 2.0, 3.0, 4.0, 5.0]
        #expect(Metrics.r2Score(yTrue: y, yPred: y) == 1.0)
    }

    @Test("R2 score is < 1 for imperfect fit")
    func testR2Imperfect() {
        let yTrue = [1.0, 2.0, 3.0, 4.0, 5.0]
        let yPred = [1.1, 1.9, 3.1, 3.9, 5.1]
        let r2 = Metrics.r2Score(yTrue: yTrue, yPred: yPred)
        #expect(r2 < 1.0)
        #expect(r2 > 0.98)
    }

    @Test("Classification Report is structured correctly")
    func testClassificationReport() {
        let yTrue = [0, 0, 1, 1, 1]
        let yPred = [0, 1, 1, 1, 0]
        let report = Metrics.classificationReport(yTrue: yTrue, yPred: yPred)
        #expect(report.perClass.count == 2)
        #expect(report.accuracy > 0.0)
        #expect(report.macroF1 > 0.0)
    }
}

// MARK: - KFold Tests

@Suite("KFold and CrossValidation Tests")
struct ValidationTests {

    @Test("KFold produces correct number of folds")
    func testKFoldSplitCount() {
        let features: [[Double]] = (0..<20).map { [Double($0)] }
        let targets: [Double] = (0..<20).map { Double($0 % 2) }
        let kf = KFold(nSplits: 5)
        let folds = kf.split(features: features, targets: targets)
        #expect(folds.count == 5)
        for fold in folds {
            #expect(!fold.trainFeatures.isEmpty)
            #expect(!fold.valFeatures.isEmpty)
        }
    }

    @Test("KFold validation set sizes are approximately equal")
    func testKFoldValidationSizes() {
        let n = 25
        let features: [[Double]] = (0..<n).map { [Double($0)] }
        let targets = [Double](repeating: 0, count: n)
        let folds = KFold(nSplits: 5).split(features: features, targets: targets)
        for fold in folds {
            #expect(fold.valFeatures.count == 5)
            #expect(fold.trainFeatures.count == 20)
        }
    }

    @Test("CrossValidator returns valid accuracy scores")
    func testCrossValidatorAccuracy() async throws {
        // Linearly separable: lower half → 0, upper half → 1
        let features: [[Double]] = (1...30).map { [Double($0)] }
        let targets: [Double] = (1...30).map { $0 <= 15 ? 0.0 : 1.0 }

        let result = try await CrossValidator.crossValidate(
            classifier: (maxDepth: 5, criterion: .gini),
            features: features,
            targets: targets,
            nSplits: 3
        )
        #expect(result.scores.count == 3)
        #expect(result.mean > 0.8)
    }

    @Test("GridSearchCV finds a best parameter combination")
    func testGridSearch() async throws {
        let features: [[Double]] = (1...30).map { [Double($0)] }
        let targets: [Double] = (1...30).map { $0 <= 15 ? 0.0 : 1.0 }

        let grid = GridSearchCV(maxDepthValues: [2, 5], criterionValues: [.gini], nSplits: 3)
        let best = try await grid.bestParams(features: features, targets: targets)
        #expect(best != nil)
        #expect(best!.meanScore > 0.7)
    }

    @Test("GridSearchCV results are sorted best-first")
    func testGridSearchSorted() async throws {
        let features: [[Double]] = (1...20).map { [Double($0)] }
        let targets: [Double] = (1...20).map { $0 <= 10 ? 0.0 : 1.0 }

        let grid = GridSearchCV(maxDepthValues: [1, 5], criterionValues: [.gini, .entropy], nSplits: 3)
        let results = try await grid.search(features: features, targets: targets)
        #expect(results.count == 4)
        // First result should have the highest mean score
        for i in 1..<results.count {
            #expect(results[0].meanScore >= results[i].meanScore)
        }
    }
}
