import Foundation
import SwiftML

// MARK: - Task Type

/// Specifies the machine learning problem domain for an AutoML session.
public enum AutoMLTaskType: Sendable, Equatable {
    /// Inferred from target values using a conservative rule (see ``AutoML`` documentation).
    /// Throws ``AutoMLError/ambiguousTaskType(uniqueValues:sampleCount:)`` when the rule
    /// cannot decide with confidence; pass an explicit task type in that case.
    case auto
    /// Discrete classification. Every target must be a finite whole number.
    case classification
    /// Continuous regression.
    case regression
}

// MARK: - Strategy

/// AutoML search strategy.
///
/// Only strategies that are fully implemented are exposed. Hyperparameter search
/// (grid / random over a typed search space) is intentionally not offered yet;
/// use ``GridSearchCV`` or ``RandomizedSearchCV`` for tuning a single estimator.
public enum AutoMLStrategy: Sendable, Equatable {
    /// Evaluates a fixed set of baseline architectures with default hyperparameters:
    /// Logistic/Linear Regression, Decision Tree, Random Forest and MLP.
    case modelSelection
}

// MARK: - Errors

/// Errors thrown by ``AutoML``.
public enum AutoMLError: Error, LocalizedError, Sendable, Equatable {
    /// Features/targets are empty or their counts differ.
    case invalidInput(String)
    /// Fewer samples than cross-validation folds.
    case insufficientSamples(count: Int, required: Int)
    /// Target contains a single unique value; nothing can be learned or ranked.
    case constantTarget
    /// `.auto` could not confidently decide between classification and regression.
    case ambiguousTaskType(uniqueValues: Int, sampleCount: Int)
    /// Explicit `.classification` was requested but a target is not a finite whole number.
    case nonIntegerClassLabel(Double)
    /// A class has fewer samples than folds, so stratified CV cannot place it in every fold.
    case insufficientClassSamples(label: Int, count: Int, required: Int)
    /// Every candidate failed on at least one fold.
    case allCandidatesFailed([AutoMLCandidateFailure])

    public var errorDescription: String? {
        switch self {
        case .invalidInput(let msg):
            return "Invalid AutoML input: \(msg)"
        case .insufficientSamples(let count, let required):
            return "AutoML needs at least \(required) samples for cross-validation, got \(count)."
        case .constantTarget:
            return "Target has a single unique value; AutoML cannot train or rank models."
        case .ambiguousTaskType(let unique, let n):
            return "Cannot infer task type: \(unique) distinct integer target values across \(n) samples. Pass taskType: .classification or .regression explicitly."
        case .nonIntegerClassLabel(let v):
            return "Classification requires whole-number class labels, found \(v)."
        case .insufficientClassSamples(let label, let count, let required):
            return "Class \(label) has \(count) sample(s); stratified \(required)-fold cross-validation needs at least \(required)."
        case .allCandidatesFailed(let failures):
            let details = failures.map { "\($0.name): \($0.reason)" }.joined(separator: "; ")
            return "All AutoML candidates failed during cross-validation (\(details))."
        }
    }
}

// MARK: - Leaderboard

/// Cross-validated result for a single candidate model.
public struct AutoMLLeaderboardEntry: Sendable, Equatable {
    /// Estimator type name, e.g. `"RandomForestClassifier"`.
    public let name: String
    /// Hyperparameters used for this candidate.
    public let parameters: [String: String]
    /// Name of the ranking metric: `"macroF1"` (classification) or `"r2"` (regression).
    public let metricName: String
    /// Mean of ``foldScores`` — the ranking key (higher is better).
    public let meanScore: Double
    /// Population standard deviation of ``foldScores``.
    public let stdScore: Double
    /// Ranking metric computed independently on each validation fold.
    public let foldScores: [Double]
    /// Wall-clock time spent fitting and predicting across all folds, in seconds.
    public let fitDuration: TimeInterval

    /// Human-readable label combining name and parameters, e.g. `"RandomForestClassifier (maxDepth: 5, nEstimators: 15)"`.
    public var displayName: String {
        guard !parameters.isEmpty else { return name }
        let params = parameters.keys.sorted().map { "\($0): \(parameters[$0]!)" }.joined(separator: ", ")
        return "\(name) (\(params))"
    }
}

/// A candidate that was excluded from the leaderboard because it failed on at least one fold.
public struct AutoMLCandidateFailure: Sendable, Equatable {
    /// Candidate display name.
    public let name: String
    /// Error description from the failing fold.
    public let reason: String
}

// MARK: - AutoML

/// Automated model selection with stratified cross-validation.
///
/// ## Pipeline
/// 1. Resolve the task type (explicit or `.auto`).
/// 2. Split with ``StratifiedKFold`` (classification) or shuffled ``KFold`` (regression).
/// 3. Fit every candidate on every fold; score each fold with ``EvaluationHarness``.
/// 4. Rank by mean fold score (macro F1 or R²), tie-break by lower std, then candidate order.
/// 5. Return the winner's pooled out-of-fold `EvaluationReport`.
///
/// ## Task Type Inference (`.auto`)
/// - Any non-integer or non-finite target → regression.
/// - Integer targets with at most ``autoClassificationMaxClasses`` distinct values → classification.
/// - Integer targets with more than `max(autoClassificationMaxClasses, n / 2)` distinct values → regression.
/// - Otherwise → throws ``AutoMLError/ambiguousTaskType(uniqueValues:sampleCount:)``.
///
/// ## Time Budget Semantics
/// `timeBudgetSeconds` is a **soft** deadline:
/// - It is checked before each candidate and before each fold.
/// - A fit already in progress is **not** interrupted (SwiftML estimators do not support mid-fit cancellation).
/// - The first candidate always runs to completion, so a successful call returns at least one
///   leaderboard entry. Total wall time can exceed the budget by one candidate-fold fit
///   (or by one full candidate when the budget expires during the first one).
/// - Candidates cut off by the deadline are simply not evaluated; they are not reported as failures.
///
/// ## Cancellation
/// Parent `Task` cancellation is honoured at the same checkpoints and throws `CancellationError`.
///
/// ## Concurrency & Determinism
/// Candidates and folds run sequentially inside the actor: deterministic ordering, bounded memory,
/// and no CPU oversubscription from nested parallel fits. `seed` controls fold shuffling and is
/// forwarded to stochastic estimators (Random Forest, MLP).
public actor AutoML {
    /// Maximum number of distinct integer values for `.auto` to classify targets without ambiguity.
    public static let autoClassificationMaxClasses = 10

    /// Soft time budget in seconds (see *Time Budget Semantics*).
    public private(set) var timeBudgetSeconds: Double
    /// Model search strategy.
    public private(set) var strategy: AutoMLStrategy
    /// Requested task type.
    public private(set) var taskType: AutoMLTaskType
    /// Number of cross-validation folds (must be ≥ 2).
    public private(set) var nFolds: Int
    /// Seed for fold shuffling and stochastic estimators.
    public private(set) var seed: Int

    /// Task type resolved during the last `fit` call.
    public private(set) var resolvedTaskType: AutoMLTaskType?
    /// Display name of the winning candidate.
    public private(set) var bestModelName: String?
    /// Mean cross-validation score of the winning candidate.
    public private(set) var bestScore: Double?
    /// Ranked candidates (best first).
    public private(set) var leaderboard: [AutoMLLeaderboardEntry] = []
    /// Candidates excluded from the leaderboard because they failed on a fold.
    public private(set) var failedCandidates: [AutoMLCandidateFailure] = []

    /// Creates a new AutoML controller.
    /// - Parameters:
    ///   - timeBudgetSeconds: Soft time budget in seconds (default: 60).
    ///   - strategy: Search strategy (default: `.modelSelection`).
    ///   - taskType: Problem domain (default: `.auto`).
    ///   - nFolds: Number of cross-validation folds, at least 2 (default: 3).
    ///   - seed: Random seed (default: 42).
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
        self.nFolds = nFolds
        self.seed = seed
    }

    /// Runs model selection with cross-validation.
    /// - Parameters:
    ///   - features: Training samples of shape `[n, p]`.
    ///   - targets: Class labels (whole numbers) or continuous values, length `n`.
    /// - Returns: Pooled out-of-fold metrics of the winning candidate plus `cv_score`, `cv_std`
    ///   and `time_spent_seconds`. Classification reports include the confusion matrix.
    /// - Throws: ``AutoMLError`` for invalid or ambiguous input, `CancellationError` if the parent task is cancelled.
    public func fit(features: [[Double]], targets: [Double]) async throws -> EvaluationReport {
        guard nFolds >= 2 else {
            throw AutoMLError.invalidInput("nFolds must be at least 2, got \(nFolds)")
        }
        guard !features.isEmpty, !targets.isEmpty else {
            throw AutoMLError.invalidInput("features and targets must not be empty")
        }
        guard features.count == targets.count else {
            throw AutoMLError.invalidInput("features count (\(features.count)) does not match targets count (\(targets.count))")
        }
        guard features.count >= nFolds else {
            throw AutoMLError.insufficientSamples(count: features.count, required: nFolds)
        }
        try Task.checkCancellation()

        let startTime = Date()
        let deadline = startTime.addingTimeInterval(timeBudgetSeconds)
        leaderboard = []
        failedCandidates = []
        bestModelName = nil
        bestScore = nil

        // 1. Task type
        let isClassification = try resolveTaskType(targets: targets)
        resolvedTaskType = isClassification ? .classification : .regression

        // 2. Folds
        let folds: [Fold]
        if isClassification {
            var classCounts: [Int: Int] = [:]
            for t in targets { classCounts[Int(t.rounded()), default: 0] += 1 }
            for label in classCounts.keys.sorted() {
                let count = classCounts[label]!
                guard count >= nFolds else {
                    throw AutoMLError.insufficientClassSamples(label: label, count: count, required: nFolds)
                }
            }
            do {
                folds = try StratifiedKFold(nSplits: nFolds, shuffle: true, seed: seed)
                    .split(features: features, targets: targets)
            } catch let error as ValidationError {
                switch error {
                case .insufficientClassSamples(let label, let count, let req):
                    throw AutoMLError.insufficientClassSamples(label: label, count: count, required: req)
                case .insufficientSamples(let count, let req):
                    throw AutoMLError.insufficientSamples(count: count, required: req)
                default:
                    throw AutoMLError.invalidInput(error.localizedDescription)
                }
            }
        } else {
            do {
                folds = try KFold(nSplits: nFolds, shuffle: true, seed: seed)
                    .split(features: features, targets: targets)
            } catch let error as ValidationError {
                switch error {
                case .insufficientSamples(let count, let req):
                    throw AutoMLError.insufficientSamples(count: count, required: req)
                default:
                    throw AutoMLError.invalidInput(error.localizedDescription)
                }
            }
        }

        // 3. Evaluate candidates
        let candidates = Self.baselineCandidates(isClassification: isClassification, seed: seed)
        let metricName = isClassification ? "macroF1" : "r2"

        struct Evaluated {
            let entry: AutoMLLeaderboardEntry
            let order: Int
            let pooledTrue: [Double]
            let pooledPred: [Double]
        }
        var evaluated: [Evaluated] = []
        var failures: [AutoMLCandidateFailure] = []

        candidateLoop: for (order, candidate) in candidates.enumerated() {
            try Task.checkCancellation()
            let hasResult = !evaluated.isEmpty
            if hasResult && Date() >= deadline { break }

            let candStart = Date()
            var foldScores: [Double] = []
            var pooledTrue: [Double] = []
            var pooledPred: [Double] = []

            for fold in folds {
                try Task.checkCancellation()
                if hasResult && Date() >= deadline { break candidateLoop }

                do {
                    let preds = try await candidate.fitPredict(fold.trainFeatures, fold.trainTargets, fold.valFeatures)
                    guard preds.count == fold.valTargets.count else {
                        throw AutoMLError.invalidInput("predicted \(preds.count) values for \(fold.valTargets.count) validation samples")
                    }
                    let score: Double
                    if isClassification {
                        score = try EvaluationHarness.evaluateClassification(
                            yTrue: fold.valTargets.map { Int($0.rounded()) },
                            yPred: preds.map { Int($0.rounded()) }
                        ).macroF1
                    } else {
                        score = try EvaluationHarness.evaluateRegression(
                            yTrue: fold.valTargets, yPred: preds, numFeatures: nil
                        ).r2
                    }
                    guard score.isFinite else {
                        throw AutoMLError.invalidInput("non-finite \(metricName) on a validation fold")
                    }
                    foldScores.append(score)
                    pooledTrue.append(contentsOf: fold.valTargets)
                    pooledPred.append(contentsOf: preds)
                } catch is CancellationError {
                    throw CancellationError()
                } catch {
                    failures.append(AutoMLCandidateFailure(name: candidate.displayName, reason: error.localizedDescription))
                    continue candidateLoop
                }
            }

            let mean = foldScores.reduce(0, +) / Double(foldScores.count)
            let variance = foldScores.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(foldScores.count)
            let entry = AutoMLLeaderboardEntry(
                name: candidate.name,
                parameters: candidate.parameters,
                metricName: metricName,
                meanScore: mean,
                stdScore: sqrt(variance),
                foldScores: foldScores,
                fitDuration: Date().timeIntervalSince(candStart)
            )
            evaluated.append(Evaluated(entry: entry, order: order, pooledTrue: pooledTrue, pooledPred: pooledPred))
        }

        failedCandidates = failures

        // 4. Rank: mean desc, std asc, candidate order asc
        evaluated.sort { a, b in
            if a.entry.meanScore != b.entry.meanScore { return a.entry.meanScore > b.entry.meanScore }
            if a.entry.stdScore != b.entry.stdScore { return a.entry.stdScore < b.entry.stdScore }
            return a.order < b.order
        }
        leaderboard = evaluated.map(\.entry)

        guard let winner = evaluated.first else {
            throw AutoMLError.allCandidatesFailed(failures)
        }
        bestModelName = winner.entry.displayName
        bestScore = winner.entry.meanScore

        // 5. Pooled out-of-fold report for the winner
        var metrics: [String: Double]
        var confusionMatrix: [[Int]]? = nil
        if isClassification {
            let eval = try EvaluationHarness.evaluateClassification(
                yTrue: winner.pooledTrue.map { Int($0.rounded()) },
                yPred: winner.pooledPred.map { Int($0.rounded()) }
            )
            metrics = eval.metrics
            confusionMatrix = eval.confusionMatrix
        } else {
            metrics = try EvaluationHarness.evaluateRegression(
                yTrue: winner.pooledTrue, yPred: winner.pooledPred, numFeatures: nil
            ).metrics
        }
        metrics["cv_score"] = winner.entry.meanScore
        metrics["cv_std"] = winner.entry.stdScore
        metrics["time_spent_seconds"] = Date().timeIntervalSince(startTime)
        return EvaluationReport(metrics: metrics, confusionMatrix: confusionMatrix)
    }

    // MARK: - Task Type Resolution

    private func resolveTaskType(targets: [Double]) throws -> Bool {
        let isWhole: (Double) -> Bool = { $0.isFinite && abs($0.rounded() - $0) < 1e-7 }
        let uniqueCount = Set(targets).count
        guard uniqueCount >= 2 else { throw AutoMLError.constantTarget }

        switch taskType {
        case .classification:
            if let bad = targets.first(where: { !isWhole($0) }) {
                throw AutoMLError.nonIntegerClassLabel(bad)
            }
            return true
        case .regression:
            return false
        case .auto:
            guard targets.allSatisfy(isWhole) else { return false }
            let maxClasses = Self.autoClassificationMaxClasses
            if uniqueCount <= maxClasses { return true }
            if uniqueCount > max(maxClasses, targets.count / 2) { return false }
            throw AutoMLError.ambiguousTaskType(uniqueValues: uniqueCount, sampleCount: targets.count)
        }
    }

    // MARK: - Candidates

    private struct CandidateSpec: Sendable {
        let name: String
        let parameters: [String: String]
        let fitPredict: @Sendable (_ trainX: [[Double]], _ trainY: [Double], _ testX: [[Double]]) async throws -> [Double]

        var displayName: String {
            guard !parameters.isEmpty else { return name }
            let params = parameters.keys.sorted().map { "\($0): \(parameters[$0]!)" }.joined(separator: ", ")
            return "\(name) (\(params))"
        }
    }

    private static func baselineCandidates(isClassification: Bool, seed: Int) -> [CandidateSpec] {
        if isClassification {
            return [
                CandidateSpec(name: "LogisticRegression", parameters: ["epochs": "200"]) { x, y, tx in
                    let m = LogisticRegression()
                    try await m.fit(features: x, targets: y, epochs: 200)
                    return try await m.predict(features: tx).map(Double.init)
                },
                CandidateSpec(name: "DecisionTreeClassifier", parameters: ["maxDepth": "4"]) { x, y, tx in
                    let m = DecisionTreeClassifier(maxDepth: 4)
                    try await m.fit(features: x, targets: y)
                    return try await m.predict(features: tx).map(Double.init)
                },
                CandidateSpec(name: "RandomForestClassifier", parameters: ["nEstimators": "15", "maxDepth": "5"]) { x, y, tx in
                    let m = RandomForestClassifier(nEstimators: 15, maxDepth: 5, randomState: seed)
                    try await m.fit(features: x, targets: y)
                    return try await m.predict(features: tx).map(Double.init)
                },
                CandidateSpec(name: "MLPClassifier", parameters: ["hiddenLayers": "16-8", "learningRate": "0.05", "maxIter": "50"]) { x, y, tx in
                    let m = MLPClassifier(hiddenLayerSizes: [16, 8], maxIter: 50, learningRate: 0.05, seed: seed)
                    try await m.fit(features: x, targets: y)
                    return try await m.predict(features: tx).map(Double.init)
                }
            ]
        } else {
            return [
                CandidateSpec(name: "LinearRegression", parameters: [:]) { x, y, tx in
                    let m = LinearRegression()
                    try await m.fit(features: x, targets: y)
                    return try await m.predict(features: tx)
                },
                CandidateSpec(name: "DecisionTreeRegressor", parameters: ["maxDepth": "4"]) { x, y, tx in
                    let m = DecisionTreeRegressor(maxDepth: 4)
                    try await m.fit(features: x, targets: y)
                    return try await m.predict(features: tx)
                },
                CandidateSpec(name: "RandomForestRegressor", parameters: ["nEstimators": "15", "maxDepth": "5"]) { x, y, tx in
                    let m = RandomForestRegressor(nEstimators: 15, maxDepth: 5, randomState: seed)
                    try await m.fit(features: x, targets: y)
                    return try await m.predict(features: tx)
                },
                CandidateSpec(name: "MLPRegressor", parameters: ["hiddenLayers": "16-8", "learningRate": "0.05", "maxIter": "50"]) { x, y, tx in
                    let m = MLPRegressor(hiddenLayerSizes: [16, 8], maxIter: 50, learningRate: 0.05, seed: seed)
                    try await m.fit(features: x, targets: y)
                    return try await m.predict(features: tx)
                }
            ]
        }
    }
}
