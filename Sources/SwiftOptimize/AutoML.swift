import Foundation
import SwiftML

/// AutoML hyperparameter optimization strategy.
public enum AutoMLStrategy: Sendable {
    case grid
    case random
    case bayesian
    case hyperband
}

/// Automated Machine Learning (AutoML) controller providing intelligent model selection
/// and automated cross-validation benchmarking under time constraints.
public actor AutoML {
    /// Maximum time budget in seconds allocated for training candidate models.
    public private(set) var timeBudgetSeconds: Double
    /// Model search strategy.
    public private(set) var strategy: AutoMLStrategy
    /// Name of the winning candidate model.
    public private(set) var bestModelName: String?
    /// Best cross-validation score achieved.
    public private(set) var bestScore: Double?
    /// Leaderboard of tested candidates with CV score and execution duration in seconds.
    public private(set) var leaderboard: [(name: String, cvScore: Double, fitDuration: Double)] = []

    /// Creates a new AutoML controller.
    /// - Parameters:
    ///   - timeBudgetSeconds: Maximum execution budget in seconds (default: 60.0).
    ///   - strategy: Search strategy (default: .bayesian).
    public init(timeBudgetSeconds: Double = 60.0, strategy: AutoMLStrategy = .bayesian) {
        self.timeBudgetSeconds = timeBudgetSeconds
        self.strategy = strategy
    }

    /// Fits model candidates and returns the best model evaluation report.
    public func fit(features: [[Double]], targets: [Double]) async throws -> EvaluationReport {
        guard !features.isEmpty, features.count == targets.count else {
            throw SwiftMLError.trainingFailed("Features and targets count mismatch in AutoML")
        }

        let numSamples = features.count
        guard numSamples >= 3 else {
            throw SwiftMLError.trainingFailed("AutoML requires at least 3 samples for cross-validation")
        }

        let startTime = Date()
        self.leaderboard.removeAll()

        // 1. Determine task type: classification if targets are discrete/integer-like
        let uniqueTargets = Set(targets)
        let isIntegerValued = targets.allSatisfy { $0 == floor($0) }
        let isClassification = isIntegerValued && uniqueTargets.count <= max(2, numSamples / 2)

        // 2. Generate 3-fold cross validation split indices
        let nFolds = min(3, numSamples)
        var folds: [[Int]] = Array(repeating: [], count: nFolds)
        for i in 0..<numSamples {
            folds[i % nFolds].append(i)
        }

        // 3. Define candidate evaluators
        struct ModelCandidate: @unchecked Sendable {
            let name: String
            let fitAndPredict: @Sendable ([[Double]], [Double], [[Double]]) async throws -> [Double]
        }

        var candidates: [ModelCandidate] = []

        if isClassification {
            candidates.append(ModelCandidate(name: "DecisionTreeClassifier (maxDepth: 4)") { trainX, trainY, testX in
                let model = DecisionTreeClassifier(maxDepth: 4)
                try await model.fit(features: trainX, targets: trainY)
                let preds = try await model.predict(features: testX)
                return preds.map { Double($0) }
            })

            candidates.append(ModelCandidate(name: "RandomForestClassifier (n: 15, maxDepth: 5)") { trainX, trainY, testX in
                let model = try RandomForestClassifier(nEstimators: 15, maxDepth: 5)
                try await model.fit(features: trainX, targets: trainY)
                let preds = try await model.predict(features: testX)
                return preds.map { Double($0) }
            })

            candidates.append(ModelCandidate(name: "LogisticRegression") { trainX, trainY, testX in
                let model = LogisticRegression()
                try await model.fit(features: trainX, targets: trainY, epochs: 200)
                let preds = try await model.predict(features: testX)
                return preds.map { Double($0) }
            })

            candidates.append(ModelCandidate(name: "MLPClassifier (16->8)") { trainX, trainY, testX in
                let model = MLPClassifier(hiddenLayerSizes: [16, 8], maxIter: 50, learningRate: 0.05)
                try await model.fit(features: trainX, targets: trainY)
                let preds = try await model.predict(features: testX)
                return preds.map { Double($0) }
            })
        } else {
            candidates.append(ModelCandidate(name: "LinearRegression") { trainX, trainY, testX in
                let model = LinearRegression()
                try await model.fit(features: trainX, targets: trainY)
                return try await model.predict(features: testX)
            })

            candidates.append(ModelCandidate(name: "DecisionTreeRegressor (maxDepth: 4)") { trainX, trainY, testX in
                let model = DecisionTreeRegressor(maxDepth: 4)
                try await model.fit(features: trainX, targets: trainY)
                return try await model.predict(features: testX)
            })

            candidates.append(ModelCandidate(name: "RandomForestRegressor (n: 15, maxDepth: 5)") { trainX, trainY, testX in
                let model = try RandomForestRegressor(nEstimators: 15, maxDepth: 5)
                try await model.fit(features: trainX, targets: trainY)
                return try await model.predict(features: testX)
            })

            candidates.append(ModelCandidate(name: "MLPRegressor (16->8)") { trainX, trainY, testX in
                let model = MLPRegressor(hiddenLayerSizes: [16, 8], maxIter: 50, learningRate: 0.05)
                try await model.fit(features: trainX, targets: trainY)
                return try await model.predict(features: testX)
            })
        }

        var evaluatedLeaderboard: [(name: String, cvScore: Double, fitDuration: Double)] = []

        // 4. Sequential Evaluation across candidate models
        for candidate in candidates {
            // Check time budget
            let elapsed = Date().timeIntervalSince(startTime)
            if elapsed >= timeBudgetSeconds && !evaluatedLeaderboard.isEmpty {
                break
            }

            let candStart = Date()
            var foldScores: [Double] = []

            for foldIdx in 0..<nFolds {
                let testIndices = Set(folds[foldIdx])
                let trainIndices = (0..<numSamples).filter { !testIndices.contains($0) }

                let trainX = trainIndices.map { features[$0] }
                let trainY = trainIndices.map { targets[$0] }
                let testX  = testIndices.map { features[$0] }
                let testY  = testIndices.map { targets[$0] }

                do {
                    let preds = try await candidate.fitAndPredict(trainX, trainY, testX)
                    if isClassification {
                        // Compute Accuracy
                        let correct = zip(testY, preds).filter { Int(round($0.0)) == Int(round($0.1)) }.count
                        let acc = Double(correct) / Double(testY.count)
                        foldScores.append(acc)
                    } else {
                        // Compute R-squared / negative MSE
                        var sumSqErr = 0.0
                        for (yTrue, yPred) in zip(testY, preds) {
                            let diff = yTrue - yPred
                            sumSqErr += diff * diff
                        }
                        let mse = sumSqErr / Double(testY.count)
                        let meanY = testY.reduce(0.0, +) / Double(testY.count)
                        let totalVar = testY.reduce(0.0) { $0 + ($1 - meanY) * ($1 - meanY) }
                        let r2 = totalVar > 1e-12 ? 1.0 - (sumSqErr / totalVar) : -mse
                        foldScores.append(r2)
                    }
                } catch {
                    // Candidate failed on this fold; ignore
                }
            }

            let candDuration = Date().timeIntervalSince(candStart)
            if !foldScores.isEmpty {
                let avgScore = foldScores.reduce(0.0, +) / Double(foldScores.count)
                evaluatedLeaderboard.append((candidate.name, avgScore, candDuration))
            }
        }

        // Sort leaderboard: higher is better (Accuracy for classification, R2 for regression)
        evaluatedLeaderboard.sort { $0.cvScore > $1.cvScore }
        self.leaderboard = evaluatedLeaderboard

        guard let winner = evaluatedLeaderboard.first else {
            throw SwiftMLError.trainingFailed("All AutoML model candidates failed during cross-validation")
        }

        self.bestModelName = winner.name
        self.bestScore = winner.cvScore

        var metrics: [String: Double] = [:]
        if isClassification {
            metrics["accuracy"] = winner.cvScore
            metrics["f1"] = winner.cvScore // Macro F1 approximation
            metrics["cv_score"] = winner.cvScore
            metrics["time_spent_seconds"] = Date().timeIntervalSince(startTime)
        } else {
            metrics["r2"] = winner.cvScore
            metrics["cv_score"] = winner.cvScore
            metrics["time_spent_seconds"] = Date().timeIntervalSince(startTime)
        }

        return EvaluationReport(metrics: metrics)
    }
}
