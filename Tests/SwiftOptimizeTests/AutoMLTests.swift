import Testing
import Foundation
import SwiftML
@testable import SwiftOptimize

@Suite("AutoML Parallel Evaluation Tests")
struct AutoMLTests {

    @Test("AutoML concurrent fold fitting on classification task")
    func testAutoMLClassification() async throws {
        let automl = AutoML(timeBudgetSeconds: 15.0)

        // Generate synthetic classification data
        var X: [[Double]] = []
        var y: [Double] = []

        for i in 0..<20 {
            if i % 2 == 0 {
                X.append([1.0, 2.0 + Double(i) * 0.1])
                y.append(0.0)
            } else {
                X.append([10.0, 20.0 + Double(i) * 0.1])
                y.append(1.0)
            }
        }

        let report = try await automl.fit(features: X, targets: y)
        let bestName = await automl.bestModelName
        let bestScore = await automl.bestScore
        #expect(bestName != nil && !bestName!.isEmpty)
        #expect(bestScore != nil && bestScore! > 0.5)
        #expect(report.metrics["cv_score"] != nil)

        let leaderboard = await automl.leaderboard
        #expect(!leaderboard.isEmpty)
        for entry in leaderboard {
            #expect(!entry.name.isEmpty)
            #expect(entry.cvScore >= 0.0)
            #expect(entry.fitDuration >= 0.0)
        }
    }

    @Test("AutoML concurrent fold fitting on regression task")
    func testAutoMLRegression() async throws {
        let automl = AutoML(timeBudgetSeconds: 15.0)

        var X: [[Double]] = []
        var y: [Double] = []

        for i in 0..<15 {
            let val = Double(i)
            X.append([val, val * 2.0])
            y.append(val * 3.5 + 1.2) // Linear continuous target
        }

        let report = try await automl.fit(features: X, targets: y)
        let bestName = await automl.bestModelName
        #expect(bestName != nil && !bestName!.isEmpty)
        #expect(report.metrics["cv_score"] != nil)

        let leaderboard = await automl.leaderboard
        #expect(!leaderboard.isEmpty)
    }

    @Test("AutoML input validation errors")
    func testAutoMLErrors() async {
        let automl = AutoML()
        await #expect(throws: SwiftMLError.self) {
            _ = try await automl.fit(features: [], targets: [])
        }
        await #expect(throws: SwiftMLError.self) {
            _ = try await automl.fit(features: [[1.0]], targets: [1.0]) // less than 3 samples
        }
        await #expect(throws: SwiftMLError.self) {
            // Dimension mismatch: 3 feature rows, 2 target elements
            _ = try await automl.fit(features: [[1.0], [2.0], [3.0]], targets: [1.0, 2.0])
        }
    }

    @Test("AutoML zero-variance continuous target regression fallback")
    func testAutoMLZeroVarianceTargetRegression() async throws {
        let automl = AutoML(timeBudgetSeconds: 15.0)

        // Non-integer values to trigger regression branch with zero total variance
        let X: [[Double]] = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0], [6.0, 7.0]]
        let y: [Double] = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5]

        let report = try await automl.fit(features: X, targets: y)
        let bestName = await automl.bestModelName
        #expect(bestName != nil && !bestName!.isEmpty)
        #expect(report.metrics["cv_score"] != nil)
        #expect(report.metrics["r2"] != nil)

        let leaderboard = await automl.leaderboard
        #expect(!leaderboard.isEmpty)
        for entry in leaderboard {
            // In zero-variance cases, score is -MSE (<= 0.0)
            #expect(entry.cvScore <= 0.0)
        }
    }

    @Test("AutoML time budget expiration breaks candidate loop early")
    func testAutoMLTimeBudgetCutoff() async throws {
        // Very tight time budget in seconds
        let automl = AutoML(timeBudgetSeconds: 0.000001)

        var X: [[Double]] = []
        var y: [Double] = []
        for i in 0..<12 {
            X.append([Double(i), Double(i * 2)])
            y.append(Double(i % 2))
        }

        let report = try await automl.fit(features: X, targets: y)
        #expect(report.metrics["cv_score"] != nil)

        let leaderboard = await automl.leaderboard
        #expect(!leaderboard.isEmpty)
        // With an immediate budget cutoff, not all candidates are evaluated
        #expect(leaderboard.count < 4)
    }

    @Test("AutoML strategy configurations and property accessors")
    func testAutoMLStrategyAndProperties() async {
        let strategies: [AutoMLStrategy] = [.grid, .random, .bayesian, .hyperband]
        for strategy in strategies {
            let automl = AutoML(timeBudgetSeconds: 42.0, strategy: strategy)
            let budget = await automl.timeBudgetSeconds
            let strat = await automl.strategy
            #expect(budget == 42.0)
            switch (strat, strategy) {
            case (.grid, .grid), (.random, .random), (.bayesian, .bayesian), (.hyperband, .hyperband):
                #expect(true)
            default:
                #expect(Bool(false), "Strategy mismatch")
            }
        }
    }
}
