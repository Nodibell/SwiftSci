import Foundation
import SwiftML

/// Specifies the machine learning problem domain for an AutoML session.
public enum AutoMLTaskType: Sendable, Equatable {
    /// Inferred automatically from target characteristics.
    ///
    /// Fallback heuristic: checks if all targets are finite whole numbers (`v == floor(v)`),
    /// unique class count is at least 2 and at most `min(20, numSamples / 2)`.
    /// If targets contain non-integer decimal values or have high unique cardinality, defaults to `.regression`.
    /// If targets are constant (only 1 unique value), throws `SwiftMLError.trainingFailed`.
    case auto
    /// Discrete classification task. Targets must map to discrete class labels.
    case classification
    /// Continuous regression task.
    case regression
}

/// AutoML hyperparameter optimization and model search strategy.
public enum AutoMLStrategy: Sendable, Equatable {
    /// Evaluates canonical baseline architectures with default hyperparameters.
    case modelSelection
    /// Systematic discrete grid search over key hyperparameter combinations.
    case grid
    /// Randomized sampling of hyperparameter combinations bounded by trial budget.
    case random(maxTrials: Int = 8)
}

/// Automated Machine Learning (AutoML) controller providing intelligent model selection,
/// hyperparameter search, and stratified cross-validation benchmarking under time constraints.
public actor AutoML {
    /// Maximum time budget in seconds allocated for training candidate models.
    public private(set) var timeBudgetSeconds: Double
    /// Model search strategy.
    public private(set) var strategy: AutoMLStrategy
    /// Problem domain (inferred or explicit).
    public private(set) var taskType: AutoMLTaskType
    /// Number of cross-validation folds.
    public private(set) var nFolds: Int
    /// Random seed for reproducible fold shuffling.
    public private(set) var seed: Int
    /// Name of the winning candidate model.
    public private(set) var bestModelName: String?
    /// Best cross-validation score achieved.
    public private(set) var bestScore: Double?
    /// Leaderboard of tested candidates with CV score and execution duration in seconds.
    public private(set) var leaderboard: [(name: String, cvScore: Double, fitDuration: Double)] = []

    /// Creates a new AutoML controller.
    /// - Parameters:
    ///   - timeBudgetSeconds: Maximum execution budget in seconds (default: 60.0).
    ///   - strategy: Search strategy (default: .modelSelection).
    ///   - taskType: Problem domain (default: .auto).
    ///   - nFolds: Number of cross-validation folds (default: 3).
    ///   - seed: Random seed for fold partitioning (default: 42).
    public init(
        timeBudgetSeconds: Double = 60.0,
        strategy: AutoMLStrategy = .modelSelection,
        taskType: AutoMLTaskType = .auto,
        nFolds: Int = 3,
        seed: Int = 42
    ) {
        self.timeBudgetSeconds = timeBudgetSeconds
        self.strategy = strategy
        self.taskType = taskType
        self.nFolds = max(2, nFolds)
        self.seed = seed
    }

    /// Evaluates candidate model architectures across cross-validation folds using structured concurrency.
    ///
    /// - Parameters:
    ///   - features: 2D array of training samples of shape `[numSamples, numFeatures]`.
    ///   - targets: 1D array of corresponding target labels or continuous values.
    /// - Returns: An `EvaluationReport` summarizing best performing model and leaderboard metrics.
    /// - Throws: `SwiftMLError` if inputs are invalid or insufficient for cross-validation.
    public func fit(features: [[Double]], targets: [Double]) async throws -> EvaluationReport {
        guard !features.isEmpty, features.count == targets.count else {
            throw SwiftMLError.trainingFailed("Features and targets count mismatch in AutoML")
        }

        let numSamples = features.count
        guard numSamples >= nFolds else {
            throw SwiftMLError.trainingFailed("AutoML requires at least \(nFolds) samples for \(nFolds)-fold cross-validation")
        }

        let startTime = Date()
        let deadline = startTime.addingTimeInterval(timeBudgetSeconds)
        self.leaderboard.removeAll()

        // 1. Resolve task type
        let isClassification = try resolveTaskType(features: features, targets: targets)

        // 2. Generate cross-validation splits
        let folds: [Fold]
        if isClassification {
            // Validate minimum class representation for stratified K-fold
            var classCounts: [Int: Int] = [:]
            for t in targets {
                classCounts[Int(round(t)), default: 0] += 1
            }
            guard classCounts.keys.count >= 2 else {
                throw SwiftMLError.trainingFailed("Classification requires at least 2 distinct classes, found \(classCounts.keys.count)")
            }
            for (cls, count) in classCounts {
                guard count >= nFolds else {
                    throw SwiftMLError.trainingFailed("Class \(cls) has only \(count) sample(s), but at least \(nFolds) folds are required for stratified cross-validation")
                }
            }
            let splitter = StratifiedKFold(nSplits: nFolds, shuffle: true, seed: seed)
            folds = splitter.split(features: features, targets: targets)
        } else {
            let uniqueTargets = Set(targets)
            guard uniqueTargets.count >= 2 else {
                throw SwiftMLError.trainingFailed("Cannot perform AutoML on constant target with only 1 unique value")
            }
            let splitter = KFold(nSplits: nFolds, shuffle: true, seed: seed)
            folds = splitter.split(features: features, targets: targets)
        }

        guard !folds.isEmpty else {
            throw SwiftMLError.trainingFailed("Failed to partition dataset into cross-validation folds")
        }

        // 3. Generate candidate models based on strategy
        let candidates = generateCandidates(isClassification: isClassification)

        var evaluatedLeaderboard: [(name: String, cvScore: Double, fitDuration: Double)] = []
        var classReports: [String: ClassificationEvaluation] = [:]
        var regReports: [String: RegressionEvaluation] = [:]

        // 4. Sequential evaluation with cooperative task cancellation and deadline monitoring
        for candidate in candidates {
            // Check deadline and task cancellation
            if Task.isCancelled || (Date() >= deadline && !evaluatedLeaderboard.isEmpty) {
                break
            }

            let candStart = Date()
            var allValTrues: [Double] = []
            var allValPreds: [Double] = []
            var candidateFailed = false

            for fold in folds {
                if Task.isCancelled || (Date() >= deadline && !evaluatedLeaderboard.isEmpty) {
                    candidateFailed = true
                    break
                }

                do {
                    let preds = try await candidate.fitAndPredict(
                        fold.trainFeatures,
                        fold.trainTargets,
                        fold.valFeatures
                    )
                    guard preds.count == fold.valTargets.count else {
                        candidateFailed = true
                        break
                    }
                    allValTrues.append(contentsOf: fold.valTargets)
                    allValPreds.append(contentsOf: preds)
                } catch {
                    candidateFailed = true
                    break
                }
            }

            let candDuration = Date().timeIntervalSince(candStart)
            guard !candidateFailed, !allValTrues.isEmpty else {
                continue
            }

            // 5. Compute true metrics strictly via EvaluationHarness
            if isClassification {
                do {
                    let eval = try EvaluationHarness.evaluateClassification(
                        yTrue: allValTrues.map { Int(round($0)) },
                        yPred: allValPreds.map { Int(round($0)) }
                    )
                    // Primary ranking metric for classification is macro F1
                    evaluatedLeaderboard.append((candidate.name, eval.macroF1, candDuration))
                    classReports[candidate.name] = eval
                } catch {
                    continue
                }
            } else {
                do {
                    let eval = try EvaluationHarness.evaluateRegression(
                        yTrue: allValTrues,
                        yPred: allValPreds,
                        numFeatures: features.first?.count
                    )
                    // Primary ranking metric for regression is R^2 (or -RMSE if R^2 is non-finite)
                    let score = eval.r2.isFinite ? eval.r2 : -eval.rmse
                    evaluatedLeaderboard.append((candidate.name, score, candDuration))
                    regReports[candidate.name] = eval
                } catch {
                    continue
                }
            }
        }

        // 6. Rank leaderboard
        evaluatedLeaderboard.sort { $0.cvScore > $1.cvScore }
        self.leaderboard = evaluatedLeaderboard

        guard let winner = evaluatedLeaderboard.first else {
            throw SwiftMLError.trainingFailed("All AutoML model candidates failed or timed out during cross-validation")
        }

        self.bestModelName = winner.name
        self.bestScore = winner.cvScore

        // 7. Produce final report
        let timeSpent = Date().timeIntervalSince(startTime)
        if isClassification, let bestEval = classReports[winner.name] {
            var metrics = bestEval.metrics
            metrics["cv_score"] = winner.cvScore
            metrics["time_spent_seconds"] = timeSpent
            return EvaluationReport(metrics: metrics, confusionMatrix: bestEval.confusionMatrix)
        } else if let bestEval = regReports[winner.name] {
            var metrics = bestEval.metrics
            metrics["cv_score"] = winner.cvScore
            metrics["time_spent_seconds"] = timeSpent
            return EvaluationReport(metrics: metrics, confusionMatrix: nil)
        } else {
            throw SwiftMLError.trainingFailed("Evaluation report generation failed for winner: \(winner.name)")
        }
    }

    // MARK: - Private Helpers

    private func resolveTaskType(features: [[Double]], targets: [Double]) throws -> Bool {
        let numSamples = targets.count

        switch self.taskType {
        case .classification:
            for t in targets {
                guard t.isFinite && abs(t.rounded() - t) < 1e-7 else {
                    throw SwiftMLError.trainingFailed("Classification task specified, but target contains non-integer continuous value: \(t)")
                }
            }
            return true

        case .regression:
            return false

        case .auto:
            let isIntegerValued = targets.allSatisfy { $0.isFinite && abs($0.rounded() - $0) < 1e-7 }
            let uniqueTargets = Set(targets)

            guard uniqueTargets.count >= 2 else {
                throw SwiftMLError.trainingFailed("Cannot perform AutoML on constant target with only 1 unique value")
            }

            // Fallback heuristic: integer targets with bounded cardinality relative to sample size
            if isIntegerValued && uniqueTargets.count <= max(2, min(20, numSamples / 2)) {
                return true
            } else {
                return false
            }
        }
    }

    private struct ModelCandidate: Sendable {
        let name: String
        let fitAndPredict: @Sendable ([[Double]], [Double], [[Double]]) async throws -> [Double]
    }

    private func generateCandidates(isClassification: Bool) -> [ModelCandidate] {
        if isClassification {
            return generateClassificationCandidates()
        } else {
            return generateRegressionCandidates()
        }
    }

    private func generateClassificationCandidates() -> [ModelCandidate] {
        switch self.strategy {
        case .modelSelection:
            return [
                ModelCandidate(name: "LogisticRegression") { trainX, trainY, testX in
                    let model = LogisticRegression()
                    try await model.fit(features: trainX, targets: trainY, epochs: 200)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                },
                ModelCandidate(name: "DecisionTreeClassifier (maxDepth: 4)") { trainX, trainY, testX in
                    let model = DecisionTreeClassifier(maxDepth: 4)
                    try await model.fit(features: trainX, targets: trainY)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                },
                ModelCandidate(name: "RandomForestClassifier (n: 15, maxDepth: 5)") { trainX, trainY, testX in
                    let model = RandomForestClassifier(nEstimators: 15, maxDepth: 5)
                    try await model.fit(features: trainX, targets: trainY)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                },
                ModelCandidate(name: "MLPClassifier (16->8)") { trainX, trainY, testX in
                    let model = MLPClassifier(hiddenLayerSizes: [16, 8], maxIter: 50, learningRate: 0.05)
                    try await model.fit(features: trainX, targets: trainY)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                }
            ]

        case .grid:
            return [
                ModelCandidate(name: "LogisticRegression") { trainX, trainY, testX in
                    let model = LogisticRegression()
                    try await model.fit(features: trainX, targets: trainY, epochs: 200)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                },
                ModelCandidate(name: "DecisionTreeClassifier (maxDepth: 3)") { trainX, trainY, testX in
                    let model = DecisionTreeClassifier(maxDepth: 3)
                    try await model.fit(features: trainX, targets: trainY)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                },
                ModelCandidate(name: "DecisionTreeClassifier (maxDepth: 6)") { trainX, trainY, testX in
                    let model = DecisionTreeClassifier(maxDepth: 6)
                    try await model.fit(features: trainX, targets: trainY)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                },
                ModelCandidate(name: "RandomForestClassifier (n: 10, maxDepth: 4)") { trainX, trainY, testX in
                    let model = RandomForestClassifier(nEstimators: 10, maxDepth: 4)
                    try await model.fit(features: trainX, targets: trainY)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                },
                ModelCandidate(name: "RandomForestClassifier (n: 25, maxDepth: 6)") { trainX, trainY, testX in
                    let model = RandomForestClassifier(nEstimators: 25, maxDepth: 6)
                    try await model.fit(features: trainX, targets: trainY)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                },
                ModelCandidate(name: "MLPClassifier (16->8, lr: 0.05)") { trainX, trainY, testX in
                    let model = MLPClassifier(hiddenLayerSizes: [16, 8], maxIter: 50, learningRate: 0.05)
                    try await model.fit(features: trainX, targets: trainY)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                },
                ModelCandidate(name: "MLPClassifier (32->16, lr: 0.01)") { trainX, trainY, testX in
                    let model = MLPClassifier(hiddenLayerSizes: [32, 16], maxIter: 50, learningRate: 0.01)
                    try await model.fit(features: trainX, targets: trainY)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                }
            ]

        case .random(let maxTrials):
            let pool: [ModelCandidate] = [
                ModelCandidate(name: "LogisticRegression") { trainX, trainY, testX in
                    let model = LogisticRegression()
                    try await model.fit(features: trainX, targets: trainY, epochs: 200)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                },
                ModelCandidate(name: "DecisionTreeClassifier (maxDepth: 3)") { trainX, trainY, testX in
                    let model = DecisionTreeClassifier(maxDepth: 3)
                    try await model.fit(features: trainX, targets: trainY)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                },
                ModelCandidate(name: "DecisionTreeClassifier (maxDepth: 5)") { trainX, trainY, testX in
                    let model = DecisionTreeClassifier(maxDepth: 5)
                    try await model.fit(features: trainX, targets: trainY)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                },
                ModelCandidate(name: "RandomForestClassifier (n: 10, maxDepth: 4)") { trainX, trainY, testX in
                    let model = RandomForestClassifier(nEstimators: 10, maxDepth: 4)
                    try await model.fit(features: trainX, targets: trainY)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                },
                ModelCandidate(name: "RandomForestClassifier (n: 20, maxDepth: 5)") { trainX, trainY, testX in
                    let model = RandomForestClassifier(nEstimators: 20, maxDepth: 5)
                    try await model.fit(features: trainX, targets: trainY)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                },
                ModelCandidate(name: "RandomForestClassifier (n: 30, maxDepth: 6)") { trainX, trainY, testX in
                    let model = RandomForestClassifier(nEstimators: 30, maxDepth: 6)
                    try await model.fit(features: trainX, targets: trainY)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                },
                ModelCandidate(name: "MLPClassifier (16, lr: 0.05)") { trainX, trainY, testX in
                    let model = MLPClassifier(hiddenLayerSizes: [16], maxIter: 40, learningRate: 0.05)
                    try await model.fit(features: trainX, targets: trainY)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                },
                ModelCandidate(name: "MLPClassifier (16->8, lr: 0.02)") { trainX, trainY, testX in
                    let model = MLPClassifier(hiddenLayerSizes: [16, 8], maxIter: 50, learningRate: 0.02)
                    try await model.fit(features: trainX, targets: trainY)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                },
                ModelCandidate(name: "MLPClassifier (32->16, lr: 0.01)") { trainX, trainY, testX in
                    let model = MLPClassifier(hiddenLayerSizes: [32, 16], maxIter: 50, learningRate: 0.01)
                    try await model.fit(features: trainX, targets: trainY)
                    let preds = try await model.predict(features: testX)
                    return preds.map { Double($0) }
                }
            ]
            return Array(pool.prefix(max(1, maxTrials)))
        }
    }

    private func generateRegressionCandidates() -> [ModelCandidate] {
        switch self.strategy {
        case .modelSelection:
            return [
                ModelCandidate(name: "LinearRegression") { trainX, trainY, testX in
                    let model = LinearRegression()
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                },
                ModelCandidate(name: "DecisionTreeRegressor (maxDepth: 4)") { trainX, trainY, testX in
                    let model = DecisionTreeRegressor(maxDepth: 4)
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                },
                ModelCandidate(name: "RandomForestRegressor (n: 15, maxDepth: 5)") { trainX, trainY, testX in
                    let model = RandomForestRegressor(nEstimators: 15, maxDepth: 5)
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                },
                ModelCandidate(name: "MLPRegressor (16->8)") { trainX, trainY, testX in
                    let model = MLPRegressor(hiddenLayerSizes: [16, 8], maxIter: 50, learningRate: 0.05)
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                }
            ]

        case .grid:
            return [
                ModelCandidate(name: "LinearRegression") { trainX, trainY, testX in
                    let model = LinearRegression()
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                },
                ModelCandidate(name: "DecisionTreeRegressor (maxDepth: 3)") { trainX, trainY, testX in
                    let model = DecisionTreeRegressor(maxDepth: 3)
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                },
                ModelCandidate(name: "DecisionTreeRegressor (maxDepth: 6)") { trainX, trainY, testX in
                    let model = DecisionTreeRegressor(maxDepth: 6)
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                },
                ModelCandidate(name: "RandomForestRegressor (n: 10, maxDepth: 4)") { trainX, trainY, testX in
                    let model = RandomForestRegressor(nEstimators: 10, maxDepth: 4)
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                },
                ModelCandidate(name: "RandomForestRegressor (n: 25, maxDepth: 6)") { trainX, trainY, testX in
                    let model = RandomForestRegressor(nEstimators: 25, maxDepth: 6)
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                },
                ModelCandidate(name: "MLPRegressor (16->8, lr: 0.05)") { trainX, trainY, testX in
                    let model = MLPRegressor(hiddenLayerSizes: [16, 8], maxIter: 50, learningRate: 0.05)
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                },
                ModelCandidate(name: "MLPRegressor (32->16, lr: 0.01)") { trainX, trainY, testX in
                    let model = MLPRegressor(hiddenLayerSizes: [32, 16], maxIter: 50, learningRate: 0.01)
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                }
            ]

        case .random(let maxTrials):
            let pool: [ModelCandidate] = [
                ModelCandidate(name: "LinearRegression") { trainX, trainY, testX in
                    let model = LinearRegression()
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                },
                ModelCandidate(name: "DecisionTreeRegressor (maxDepth: 3)") { trainX, trainY, testX in
                    let model = DecisionTreeRegressor(maxDepth: 3)
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                },
                ModelCandidate(name: "DecisionTreeRegressor (maxDepth: 5)") { trainX, trainY, testX in
                    let model = DecisionTreeRegressor(maxDepth: 5)
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                },
                ModelCandidate(name: "RandomForestRegressor (n: 10, maxDepth: 4)") { trainX, trainY, testX in
                    let model = RandomForestRegressor(nEstimators: 10, maxDepth: 4)
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                },
                ModelCandidate(name: "RandomForestRegressor (n: 20, maxDepth: 5)") { trainX, trainY, testX in
                    let model = RandomForestRegressor(nEstimators: 20, maxDepth: 5)
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                },
                ModelCandidate(name: "RandomForestRegressor (n: 30, maxDepth: 6)") { trainX, trainY, testX in
                    let model = RandomForestRegressor(nEstimators: 30, maxDepth: 6)
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                },
                ModelCandidate(name: "MLPRegressor (16, lr: 0.05)") { trainX, trainY, testX in
                    let model = MLPRegressor(hiddenLayerSizes: [16], maxIter: 40, learningRate: 0.05)
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                },
                ModelCandidate(name: "MLPRegressor (16->8, lr: 0.02)") { trainX, trainY, testX in
                    let model = MLPRegressor(hiddenLayerSizes: [16, 8], maxIter: 50, learningRate: 0.02)
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                },
                ModelCandidate(name: "MLPRegressor (32->16, lr: 0.01)") { trainX, trainY, testX in
                    let model = MLPRegressor(hiddenLayerSizes: [32, 16], maxIter: 50, learningRate: 0.01)
                    try await model.fit(features: trainX, targets: trainY)
                    return try await model.predict(features: testX)
                }
            ]
            return Array(pool.prefix(max(1, maxTrials)))
        }
    }
}
