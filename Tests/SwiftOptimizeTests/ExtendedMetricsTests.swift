import Testing
@testable import SwiftOptimize
@testable import SwiftML

@Suite("Extended Metrics & Search Tests")
struct ExtendedMetricsTests {

    @Test("Extended classification metrics compute accurately")
    func testExtendedClassificationMetrics() {
        let yTrue = [1, 1, 0, 0, 1, 0, 1, 0]
        let yPred = [1, 1, 0, 0, 0, 0, 1, 1]

        let bAcc = Metrics.balancedAccuracy(yTrue: yTrue, yPred: yPred)
        #expect(bAcc > 0.5)

        let mcc = Metrics.matthewsCorrelationCoefficient(yTrue: yTrue, yPred: yPred)
        #expect(mcc > 0.0)

        let kappa = Metrics.cohenKappa(yTrue: yTrue, yPred: yPred)
        #expect(kappa > 0.0)
    }

    @Test("Probability and ROC curve metrics compute correctly")
    func testProbabilityAndROCMetrics() {
        let yTrue = [1, 1, 0, 0]
        let yScore = [0.9, 0.8, 0.2, 0.1]

        let loss = Metrics.logLoss(yTrue: yTrue, yScore: yScore)
        #expect(loss < 0.3)

        let brier = Metrics.brierScore(yTrue: yTrue, yScore: yScore)
        #expect(brier < 0.1)

        let auc = Metrics.rocAUC(yTrue: yTrue, yScore: yScore)
        #expect(abs(auc - 1.0) < 1e-5)

        let pr = Metrics.prCurve(yTrue: yTrue, yScore: yScore)
        #expect(!pr.isEmpty)
    }

    @Test("RandomizedSearchCV runs random search over hyperparameters")
    func testRandomizedSearchCV() async throws {
        let X: [[Double]] = [
            [1.0, 2.0],
            [2.0, 3.0],
            [10.0, 20.0],
            [11.0, 21.0]
        ]
        let y: [Double] = [0.0, 0.0, 1.0, 1.0]

        let search = RandomizedSearchCV(
            maxDepthValues: [2, 3, 5],
            criterionValues: [.gini, .entropy],
            nIter: 3,
            nSplits: 2
        )

        let results = try await search.search(features: X, targets: y)
        #expect(results.count == 3)

        let best = try await search.bestParams(features: X, targets: y)
        #expect(best != nil)
    }
}
