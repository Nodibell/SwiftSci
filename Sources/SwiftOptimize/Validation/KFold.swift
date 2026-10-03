import Foundation
import SwiftML

// MARK: - Fold

/// A single train/validation split.
public struct Fold: Sendable {
    /// The train features.
    public let trainFeatures: [[Double]]
    /// The train targets.
    public let trainTargets: [Double]
    /// The val features.
    public let valFeatures: [[Double]]
    /// The val targets.
    public let valTargets: [Double]
}

// MARK: - KFold

/// Splits a dataset into K folds for cross-validation.
public struct KFold: Sendable {
    /// Number of splits.
    public let nSplits: Int
    /// Whether to shuffle samples before splitting.
    public let shuffle: Bool
    /// Random seed for deterministic shuffling.
    public let seed: Int

    /// Creates a new KFold cross-validator.
    /// - Parameters:
    ///   - nSplits: Number of splits (must be at least 2).
    ///   - shuffle: Whether to shuffle sample indices.
    ///   - seed: Random seed.
    /// - Throws: `ValidationError.invalidFoldCount` if `nSplits < 2`.
    public init(nSplits: Int = 5, shuffle: Bool = true, seed: Int = 42) throws {
        guard nSplits >= 2 else {
            throw ValidationError.invalidFoldCount(nSplits)
        }
        self.nSplits = nSplits
        self.shuffle = shuffle
        self.seed = seed
    }

    /// Returns K Fold objects for the given dataset.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    ///   - targets: 1D array of ground-truth target values of length `N`.
    /// - Throws: `ValidationError` if data is empty, mismatched, or samples count < nSplits.
    /// - Returns: The computed [Fold] result instance.
    public func split(features: [[Double]], targets: [Double]) throws -> [Fold] {
        let n = features.count
        guard n > 0 else {
            throw ValidationError.emptyDataset
        }
        guard n == targets.count else {
            throw ValidationError.dimensionMismatch(features: n, targets: targets.count)
        }
        guard n >= nSplits else {
            throw ValidationError.insufficientSamples(samples: n, required: nSplits)
        }

        var indices = Array(0..<n)

        if shuffle {
            var rng = SeededRandom(seed: seed)
            indices.shuffle(using: &rng)
        }

        let baseSize = n / nSplits
        let remainder = n % nSplits
        var folds = [Fold]()
        folds.reserveCapacity(nSplits)

        var currentStart = 0
        for k in 0..<nSplits {
            let currentSize = baseSize + (k < remainder ? 1 : 0)
            let start = currentStart
            let end   = start + currentSize
            currentStart = end
            
            let valIdx   = Array(indices[start..<end])
            let trainIdx = Array(indices[0..<start]) + Array(indices[end..<n])

            let trainFeatures = trainIdx.map { features[$0] }
            let trainTargets  = trainIdx.map { targets[$0] }
            let valFeatures   = valIdx.map { features[$0] }
            let valTargets    = valIdx.map { targets[$0] }

            folds.append(Fold(trainFeatures: trainFeatures, trainTargets: trainTargets,
                              valFeatures: valFeatures, valTargets: valTargets))
        }
        return folds
    }
}

// MARK: - Cross Validation Result

/// Represents cross validation result.
public struct CrossValidationResult: Sendable {
    /// The individual fold scores.
    public let scores: [Double]
    /// The arithmetic mean across fold scores.
    public let mean: Double
    /// The standard deviation across fold scores.
    public let std: Double

    /// Creates a new cross-validation result instance.
    /// - Parameters:
    ///   - scores: Array of non-empty fold evaluation scores.
    /// - Throws: `ValidationError.invalidParameter` if scores array is empty.
    public init(scores: [Double]) throws {
        guard !scores.isEmpty else {
            throw ValidationError.invalidParameter("CrossValidationResult requires at least one score")
        }
        self.scores = scores
        let mean = scores.reduce(0, +) / Double(scores.count)
        let variance = scores.map { pow($0 - mean, 2) }.reduce(0, +) / Double(scores.count)
        self.mean = mean
        self.std = sqrt(variance)
    }
}

// MARK: - Cross Validator

/// Performs K-Fold cross-validation using a score closure.
/// Each fold is evaluated concurrently with TaskGroup.
public enum CrossValidator {

    /// Cross-validates a Decision Tree Classifier.
    /// - Parameters:
    ///   - classifier: Trained classifier estimator instance.
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    ///   - targets: 1D array of ground-truth target values of length `N`.
    ///   - nSplits: Number of cross-validation splitting folds.
    ///   - seed: Random number generator seed for deterministic reproducibility.
    /// - Throws: `ValidationError` or `SwiftMLError` if parameter grids are empty, folds are invalid, or evaluations fail.
    /// - Returns: The computed CrossValidationResult result instance.
    public static func crossValidate(
        classifier: (maxDepth: Int, criterion: SplitCriterion),
        features: [[Double]],
        targets: [Double],
        nSplits: Int = 5,
        seed: Int = 42
    ) async throws -> CrossValidationResult {
        let folds = try KFold(nSplits: nSplits, shuffle: true, seed: seed)
            .split(features: features, targets: targets)

        let scores: [Double] = try await withThrowingTaskGroup(of: Double.self) { group in
            for fold in folds {
                group.addTask {
                    let tree = DecisionTreeClassifier(
                        maxDepth: classifier.maxDepth,
                        criterion: classifier.criterion
                    )
                    try await tree.fit(features: fold.trainFeatures, targets: fold.trainTargets)
                    let preds = try await tree.predict(features: fold.valFeatures)
                    let trueLabels = fold.valTargets.map { Int($0) }
                    return Metrics.accuracy(yTrue: trueLabels, yPred: preds)
                }
            }
            var results = [Double]()
            for try await score in group { results.append(score) }
            return results
        }
        return try CrossValidationResult(scores: scores)
    }

    /// Cross-validates a Decision Tree Regressor using R² score.
    /// - Parameters:
    ///   - maxDepth: Maximum allowable depth of the decision tree.
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    ///   - targets: 1D array of ground-truth target values of length `N`.
    ///   - nSplits: Number of cross-validation splitting folds.
    ///   - seed: Random number generator seed for deterministic reproducibility.
    /// - Throws: `ValidationError` or `SwiftMLError` if parameter grids are empty, folds are invalid, or evaluations fail.
    /// - Returns: The computed CrossValidationResult result instance.
    public static func crossValidateRegressor(
        maxDepth: Int,
        features: [[Double]],
        targets: [Double],
        nSplits: Int = 5,
        seed: Int = 42
    ) async throws -> CrossValidationResult {
        let folds = try KFold(nSplits: nSplits, shuffle: true, seed: seed)
            .split(features: features, targets: targets)

        let scores: [Double] = try await withThrowingTaskGroup(of: Double.self) { group in
            for fold in folds {
                group.addTask {
                    let tree = DecisionTreeRegressor(maxDepth: maxDepth)
                    try await tree.fit(features: fold.trainFeatures, targets: fold.trainTargets)
                    let preds = try await tree.predict(features: fold.valFeatures)
                    return Metrics.r2Score(yTrue: fold.valTargets, yPred: preds)
                }
            }
            var results = [Double]()
            for try await score in group { results.append(score) }
            return results
        }
        return try CrossValidationResult(scores: scores)
    }

    /// Generic cross-validation for any ClassifierEstimator.
    /// - Parameters:
    ///   - estimatorFactory: Closure creating a fresh estimator instance for each fold.
    ///   - features: Feature matrix (N samples × D features).
    ///   - targets: Target label array.
    ///   - nSplits: Number of cross-validation folds.
    ///   - seed: Random seed for fold shuffling.
    /// - Throws: Any error thrown during model fitting or evaluation.
    /// - Returns: Cross-validation result containing accuracy scores per fold.
    public static func crossValidate<E: ClassifierEstimator>(
        _ estimatorFactory: @escaping @Sendable () -> E,
        features: [[Double]],
        targets: [Double],
        nSplits: Int = 5,
        seed: Int = 42
    ) async throws -> CrossValidationResult {
        let folds = try KFold(nSplits: nSplits, shuffle: true, seed: seed)
            .split(features: features, targets: targets)

        let scores: [Double] = try await withThrowingTaskGroup(of: Double.self) { group in
            for fold in folds {
                group.addTask {
                    let estimator = estimatorFactory()
                    try await estimator.fit(features: fold.trainFeatures, targets: fold.trainTargets)
                    let preds = try await estimator.predict(features: fold.valFeatures)
                    let trueLabels = fold.valTargets.map { Int($0) }
                    return Metrics.accuracy(yTrue: trueLabels, yPred: preds)
                }
            }
            var results = [Double]()
            for try await score in group { results.append(score) }
            return results
        }
        return try CrossValidationResult(scores: scores)
    }

    /// Generic cross-validation for any RegressorEstimator.
    /// - Parameters:
    ///   - estimatorFactory: Closure creating a fresh estimator instance for each fold.
    ///   - features: Feature matrix (N samples × D features).
    ///   - targets: Continuous target value array.
    ///   - nSplits: Number of cross-validation folds.
    ///   - seed: Random seed for fold shuffling.
    /// - Throws: Any error thrown during model fitting or evaluation.
    /// - Returns: Cross-validation result containing metrics per fold.
    public static func crossValidateRegressor<E: RegressorEstimator>(
        _ estimatorFactory: @escaping @Sendable () -> E,
        features: [[Double]],
        targets: [Double],
        nSplits: Int = 5,
        seed: Int = 42
    ) async throws -> CrossValidationResult {
        let folds = try KFold(nSplits: nSplits, shuffle: true, seed: seed)
            .split(features: features, targets: targets)

        let scores: [Double] = try await withThrowingTaskGroup(of: Double.self) { group in
            for fold in folds {
                group.addTask {
                    let estimator = estimatorFactory()
                    try await estimator.fit(features: fold.trainFeatures, targets: fold.trainTargets)
                    let preds = try await estimator.predict(features: fold.valFeatures)
                    return Metrics.r2Score(yTrue: fold.valTargets, yPred: preds)
                }
            }
            var results = [Double]()
            for try await score in group { results.append(score) }
            return results
        }
        return try CrossValidationResult(scores: scores)
    }
}
