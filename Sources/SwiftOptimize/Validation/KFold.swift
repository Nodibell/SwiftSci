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
    /// The n splits.
    public let nSplits: Int
    /// The shuffle.
    public let shuffle: Bool
    /// The seed.
    public let seed: Int

    /// Creates a new instance.
    /// - Parameters:
    ///   - nSplits: The n splits.
    ///   - shuffle: The shuffle.
    ///   - seed: The seed.
    public init(nSplits: Int = 5, shuffle: Bool = true, seed: Int = 42) {
        precondition(nSplits >= 2, "KFold requires at least 2 splits")
        self.nSplits = nSplits
        self.shuffle = shuffle
        self.seed = seed
    }

    /// Returns K Fold objects for the given dataset.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    ///   - targets: 1D array of ground-truth target values of length `N`.
    /// - Returns: The computed [Fold] result instance.
    public func split(features: [[Double]], targets: [Double]) -> [Fold] {
        let n = features.count
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
    /// The scores.
    public let scores: [Double]
    /// The mean.
    public let mean: Double
    /// The std.
    public let std: Double

    /// Creates a new instance.
    /// - Parameters:
    ///   - scores: The scores.
    public init(scores: [Double]) {
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
    /// - Throws: `SwiftMLError` if parameter grids are empty, folds are invalid, or evaluations fail.
    /// - Returns: The computed CrossValidationResult result instance.
    public static func crossValidate(
        classifier: (maxDepth: Int, criterion: SplitCriterion),
        features: [[Double]],
        targets: [Double],
        nSplits: Int = 5,
        seed: Int = 42
    ) async throws -> CrossValidationResult {
        let folds = KFold(nSplits: nSplits, shuffle: true, seed: seed)
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
        return CrossValidationResult(scores: scores)
    }

    /// Cross-validates a Decision Tree Regressor using R² score.
    /// - Parameters:
    ///   - maxDepth: Maximum allowable depth of the decision tree.
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    ///   - targets: 1D array of ground-truth target values of length `N`.
    ///   - nSplits: Number of cross-validation splitting folds.
    ///   - seed: Random number generator seed for deterministic reproducibility.
    /// - Throws: `SwiftMLError` if parameter grids are empty, folds are invalid, or evaluations fail.
    /// - Returns: The computed CrossValidationResult result instance.
    public static func crossValidateRegressor(
        maxDepth: Int,
        features: [[Double]],
        targets: [Double],
        nSplits: Int = 5,
        seed: Int = 42
    ) async throws -> CrossValidationResult {
        let folds = KFold(nSplits: nSplits, shuffle: true, seed: seed)
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
        return CrossValidationResult(scores: scores)
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
        let folds = KFold(nSplits: nSplits, shuffle: true, seed: seed)
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
        return CrossValidationResult(scores: scores)
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
        let folds = KFold(nSplits: nSplits, shuffle: true, seed: seed)
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
        return CrossValidationResult(scores: scores)
    }
}
