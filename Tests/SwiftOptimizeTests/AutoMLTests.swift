import Testing
import Foundation
import SwiftML
@testable import SwiftOptimize

@Suite("AutoML Pipeline & Strategy Tests")
struct AutoMLTests {

    @Test("AutoML classification with Stratified K-Fold and real EvaluationHarness metrics")
    func testAutoMLClassification() async throws {
        let automl = AutoML(timeBudgetSeconds: 15.0, strategy: .modelSelection)

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

        // Verify genuine EvaluationHarness metrics and confusion matrix
        #expect(report.metrics["cv_score"] != nil)
        #expect(report.metrics["accuracy"] != nil)
        #expect(report.metrics["macroF1"] != nil)
        #expect(report.metrics["weightedF1"] != nil)
        #expect(report.confusionMatrix != nil)
        #expect(report.confusionMatrix?.count == 2)

        let leaderboard = await automl.leaderboard
        #expect(!leaderboard.isEmpty)
        for entry in leaderboard {
            #expect(!entry.name.isEmpty)
            #expect(entry.cvScore >= 0.0)
            #expect(entry.fitDuration >= 0.0)
        }
    }

    @Test("AutoML regression with genuine R^2, RMSE, and MAE from EvaluationHarness")
    func testAutoMLRegression() async throws {
        let automl = AutoML(timeBudgetSeconds: 15.0, strategy: .modelSelection)

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
        #expect(report.metrics["r2"] != nil)
        #expect(report.metrics["rmse"] != nil)
        #expect(report.metrics["mae"] != nil)
        #expect(report.confusionMatrix == nil)

        let leaderboard = await automl.leaderboard
        #expect(!leaderboard.isEmpty)
    }

    @Test("AutoML input validation and dimension mismatch errors")
    func testAutoMLErrors() async {
        let automl = AutoML(nFolds: 3)
        await #expect(throws: SwiftMLError.self) {
            _ = try await automl.fit(features: [], targets: [])
        }
        await #expect(throws: SwiftMLError.self) {
            _ = try await automl.fit(features: [[1.0]], targets: [1.0]) // less than nFolds samples
        }
        await #expect(throws: SwiftMLError.self) {
            _ = try await automl.fit(features: [[1.0], [2.0], [3.0]], targets: [1.0, 2.0]) // mismatch
        }
    }

    @Test("AutoML rejects constant target with single unique value")
    func testAutoMLConstantTargetRejection() async {
        let automl = AutoML(timeBudgetSeconds: 5.0)
        let X = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        let y = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]

        await #expect(throws: SwiftMLError.self) {
            _ = try await automl.fit(features: X, targets: y)
        }
    }

    @Test("AutoML stratified validation requires each class to have at least nFolds samples")
    func testAutoMLStratifiedValidationSampleCheck() async {
        let automl = AutoML(nFolds: 3)
        // Class 0 has 5 samples, but Class 1 only has 2 samples (< 3 folds)
        let X = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]
        let y = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

        await #expect(throws: SwiftMLError.self) {
            _ = try await automl.fit(features: X, targets: y)
        }
    }

    @Test("AutoML explicit taskType handling")
    func testAutoMLExplicitTaskType() async throws {
        // Explicit .classification with fractional targets should fail
        let automlClass = AutoML(taskType: .classification)
        let X = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        let yCont = [0.2, 0.8, 1.5, 2.1, 0.4, 1.8]
        await #expect(throws: SwiftMLError.self) {
            _ = try await automlClass.fit(features: X, targets: yCont)
        }

        // Explicit .regression on integer targets should succeed as regression
        let automlReg = AutoML(timeBudgetSeconds: 15.0, taskType: .regression)
        let yInt = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let report = try await automlReg.fit(features: X, targets: yInt)
        #expect(report.metrics["r2"] != nil)
        #expect(report.confusionMatrix == nil)
    }

    @Test("AutoML time budget cutoff stops candidate evaluation early")
    func testAutoMLTimeBudgetCutoff() async throws {
        // Budget enough for initial model, but cuts off remaining candidates from 7-candidate grid
        let automl = AutoML(timeBudgetSeconds: 0.05, strategy: .grid)

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
        #expect(leaderboard.count < 7)
    }

    @Test("AutoML grid strategy evaluates discrete hyperparameter candidates")
    func testAutoMLGridStrategy() async throws {
        let automl = AutoML(timeBudgetSeconds: 20.0, strategy: .grid)

        var X: [[Double]] = []
        var y: [Double] = []
        for i in 0..<15 {
            let val = Double(i)
            X.append([val, val * 1.5])
            y.append(val * 2.0 + 1.0)
        }

        let report = try await automl.fit(features: X, targets: y)
        #expect(report.metrics["r2"] != nil)

        let leaderboard = await automl.leaderboard
        #expect(!leaderboard.isEmpty)
        let modelNames = leaderboard.map { $0.name }
        #expect(modelNames.contains(where: { $0.contains("maxDepth") }))
    }

    @Test("AutoML random search strategy limits trials to maxTrials")
    func testAutoMLRandomStrategy() async throws {
        let automl = AutoML(timeBudgetSeconds: 20.0, strategy: .random(maxTrials: 3))

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
        #expect(leaderboard.count <= 3)
    }
}
