import Testing
import Foundation
import SwiftML
@testable import SwiftOptimize

@Suite("AutoML Model Selection Pipeline Tests")
struct AutoMLTests {

    // MARK: - Fixtures

    private static func separableBinary(n: Int = 24) -> ([[Double]], [Double]) {
        var X: [[Double]] = []
        var y: [Double] = []
        for i in 0..<n {
            if i % 2 == 0 {
                X.append([1.0 + Double(i) * 0.01, 2.0 + Double(i) * 0.1]); y.append(0)
            } else {
                X.append([10.0 + Double(i) * 0.01, 20.0 + Double(i) * 0.1]); y.append(1)
            }
        }
        return (X, y)
    }

    private static func linearRegression(n: Int = 18) -> ([[Double]], [Double]) {
        let X = (0..<n).map { [Double($0), Double($0) * 2.0] }
        let y = (0..<n).map { Double($0) * 3.5 + 1.2 }
        return (X, y)
    }

    // MARK: - Core behaviour

    @Test("Classification: per-fold macroF1 leaderboard, pooled report with confusion matrix")
    func classification() async throws {
        let (X, y) = Self.separableBinary()
        let automl = AutoML(timeBudgetSeconds: 30, nFolds: 3)
        let report = try await automl.fit(features: X, targets: y)

        let leaderboard = await automl.leaderboard
        #expect(leaderboard.count == 4)
        #expect(await automl.resolvedTaskType == .classification)
        for entry in leaderboard {
            #expect(entry.metricName == "macroF1")
            #expect(entry.foldScores.count == 3)
            let mean = entry.foldScores.reduce(0, +) / 3
            #expect(abs(entry.meanScore - mean) < 1e-12)
            #expect(entry.stdScore >= 0)
            #expect(entry.fitDuration >= 0)
        }
        // Ranked best-first
        for (a, b) in zip(leaderboard, leaderboard.dropFirst()) {
            #expect(a.meanScore >= b.meanScore)
        }

        #expect(report.metrics["cv_score"] == leaderboard[0].meanScore)
        #expect(report.metrics["cv_std"] == leaderboard[0].stdScore)
        #expect(report.metrics["accuracy"] != nil)
        #expect(report.metrics["macroF1"] != nil)
        #expect(report.metrics["weightedF1"] != nil)
        #expect(report.confusionMatrix?.count == 2)
        #expect(await automl.bestModelName == leaderboard[0].displayName)
        #expect(await automl.bestScore == leaderboard[0].meanScore)
    }

    @Test("Imbalanced classification ranks by macroF1 and pooled report covers all samples")
    func imbalancedClassification() async throws {
        // 27 majority vs 6 minority samples — minority still ≥ nFolds for stratification
        var X: [[Double]] = []
        var y: [Double] = []
        for i in 0..<27 { X.append([Double(i % 5), 1.0]); y.append(0) }
        for i in 0..<6 { X.append([Double(i % 5) + 20.0, 9.0]); y.append(1) }

        let automl = AutoML(timeBudgetSeconds: 30, nFolds: 3)
        let report = try await automl.fit(features: X, targets: y)
        let leaderboard = await automl.leaderboard
        #expect(leaderboard.allSatisfy { $0.metricName == "macroF1" })
        let cm = try #require(report.confusionMatrix)
        #expect(cm.flatMap { $0 }.reduce(0, +) == 33)
    }

    @Test("Regression: R² ranking, RMSE/MAE in report, no confusion matrix")
    func regression() async throws {
        let (X, y) = Self.linearRegression()
        let automl = AutoML(timeBudgetSeconds: 30)
        let report = try await automl.fit(features: X, targets: y)

        #expect(await automl.resolvedTaskType == .regression)
        let leaderboard = await automl.leaderboard
        #expect(!leaderboard.isEmpty)
        #expect(leaderboard.allSatisfy { $0.metricName == "r2" && $0.foldScores.count == 3 })
        #expect(report.metrics["r2"] != nil)
        #expect(report.metrics["rmse"] != nil)
        #expect(report.metrics["mae"] != nil)
        #expect(report.confusionMatrix == nil)
    }

    @Test("Wide data (n ≤ p + 1) regression does not fail on adjusted R²")
    func wideDataRegression() async throws {
        let n = 12, p = 20
        let X = (0..<n).map { i in (0..<p).map { j in Double(i * p + j).truncatingRemainder(dividingBy: 7) + Double(i) } }
        let y = (0..<n).map { Double($0) * 1.7 + 0.3 }
        let automl = AutoML(timeBudgetSeconds: 30, taskType: .regression)
        let report = try await automl.fit(features: X, targets: y)
        #expect(report.metrics["r2"] != nil)
        #expect(!(await automl.leaderboard).isEmpty)
    }

    @Test("Same seed yields identical leaderboard")
    func determinism() async throws {
        let (X, y) = Self.separableBinary(n: 30)
        let a = AutoML(timeBudgetSeconds: 60, seed: 7)
        let b = AutoML(timeBudgetSeconds: 60, seed: 7)
        _ = try await a.fit(features: X, targets: y)
        _ = try await b.fit(features: X, targets: y)
        let la = await a.leaderboard
        let lb = await b.leaderboard
        #expect(la.map(\.displayName) == lb.map(\.displayName))
        #expect(la.map(\.foldScores) == lb.map(\.foldScores))
    }

    // MARK: - Task type resolution

    @Test(".auto infers classification for few integer classes")
    func autoClassification() async throws {
        let (X, y) = Self.separableBinary()
        let automl = AutoML(timeBudgetSeconds: 30)
        _ = try await automl.fit(features: X, targets: y)
        #expect(await automl.resolvedTaskType == .classification)
    }

    @Test(".auto infers regression for high-cardinality integer targets")
    func autoRegressionHighCardinality() async throws {
        let n = 15
        let X = (0..<n).map { [Double($0)] }
        let y = (0..<n).map { Double($0 * 3 + 1) } // 15 unique > max(10, 7)
        let automl = AutoML(timeBudgetSeconds: 30)
        _ = try await automl.fit(features: X, targets: y)
        #expect(await automl.resolvedTaskType == .regression)
    }

    @Test(".auto throws ambiguousTaskType for mid-cardinality integer targets")
    func autoAmbiguous() async {
        // 12 distinct values, n = 36 → 10 < 12 ≤ 18
        let X = (0..<36).map { [Double($0)] }
        let y = (0..<36).map { Double($0 % 12) }
        let automl = AutoML()
        await #expect(throws: AutoMLError.ambiguousTaskType(uniqueValues: 12, sampleCount: 36)) {
            _ = try await automl.fit(features: X, targets: y)
        }
    }

    @Test("Explicit .classification rejects fractional labels")
    func explicitClassificationRejectsFractional() async {
        let X = (0..<6).map { [Double($0)] }
        let y = [0.0, 1.0, 0.0, 1.0, 0.5, 1.0]
        let automl = AutoML(taskType: .classification)
        await #expect(throws: AutoMLError.nonIntegerClassLabel(0.5)) {
            _ = try await automl.fit(features: X, targets: y)
        }
    }

    @Test("Explicit .regression on integer targets runs regression")
    func explicitRegression() async throws {
        let X = (0..<9).map { [Double($0)] }
        let y = [1.0, 2, 3, 1, 2, 3, 1, 2, 3]
        let automl = AutoML(timeBudgetSeconds: 30, taskType: .regression)
        let report = try await automl.fit(features: X, targets: y)
        #expect(await automl.resolvedTaskType == .regression)
        #expect(report.confusionMatrix == nil)
    }

    // MARK: - Validation

    @Test("Constant target throws constantTarget")
    func constantTarget() async {
        let X = (0..<6).map { [Double($0)] }
        let automl = AutoML()
        await #expect(throws: AutoMLError.constantTarget) {
            _ = try await automl.fit(features: X, targets: Array(repeating: 2.5, count: 6))
        }
    }

    @Test("Class with fewer samples than folds throws insufficientClassSamples")
    func insufficientClassSamples() async {
        let X = (0..<7).map { [Double($0)] }
        let y = [0.0, 0, 0, 0, 0, 1, 1]
        let automl = AutoML(nFolds: 3)
        await #expect(throws: AutoMLError.insufficientClassSamples(label: 1, count: 2, required: 3)) {
            _ = try await automl.fit(features: X, targets: y)
        }
    }

    @Test("Empty, mismatched, too-small inputs and invalid nFolds throw")
    func inputValidation() async {
        let automl = AutoML(nFolds: 3)
        await #expect(throws: AutoMLError.self) { _ = try await automl.fit(features: [], targets: []) }
        await #expect(throws: AutoMLError.self) {
            _ = try await automl.fit(features: [[1], [2], [3]], targets: [1, 2])
        }
        await #expect(throws: AutoMLError.insufficientSamples(count: 2, required: 3)) {
            _ = try await automl.fit(features: [[1], [2]], targets: [0, 1])
        }
        let badFolds = AutoML(nFolds: 1)
        await #expect(throws: AutoMLError.self) {
            _ = try await badFolds.fit(features: [[1], [2], [3]], targets: [0, 1, 0])
        }
    }

    @Test("AutoMLError provides descriptions for every case")
    func errorDescriptions() {
        let errors: [AutoMLError] = [
            .invalidInput("x"), .insufficientSamples(count: 1, required: 3), .constantTarget,
            .ambiguousTaskType(uniqueValues: 12, sampleCount: 36), .nonIntegerClassLabel(0.5),
            .insufficientClassSamples(label: 1, count: 2, required: 3),
            .allCandidatesFailed([AutoMLCandidateFailure(name: "M", reason: "r")])
        ]
        for e in errors { #expect(!(e.errorDescription ?? "").isEmpty) }
    }

    // MARK: - Budget & cancellation

    @Test("Zero time budget evaluates exactly one candidate")
    func zeroTimeBudget() async throws {
        let (X, y) = Self.separableBinary()
        let automl = AutoML(timeBudgetSeconds: 0)
        let report = try await automl.fit(features: X, targets: y)
        #expect(await automl.leaderboard.count == 1)
        #expect(await automl.failedCandidates.isEmpty)
        #expect(report.metrics["cv_score"] != nil)
    }

    @Test("Cancelled parent task throws CancellationError")
    func cancellation() async {
        let (X, y) = Self.separableBinary()
        let automl = AutoML(timeBudgetSeconds: 30)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await automl.fit(features: X, targets: y)
        }
        await #expect(throws: CancellationError.self) { _ = try await task.value }
    }

    @Test("Display name sorts parameters deterministically")
    func displayName() {
        let entry = AutoMLLeaderboardEntry(
            name: "RandomForestClassifier", parameters: ["nEstimators": "15", "maxDepth": "5"],
            metricName: "macroF1", meanScore: 1, stdScore: 0, foldScores: [1], fitDuration: 0
        )
        #expect(entry.displayName == "RandomForestClassifier (maxDepth: 5, nEstimators: 15)")
    }
}
