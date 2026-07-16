import Foundation
import SwiftML

// MARK: - Fold

/// A single train/validation split.
public struct Fold: Sendable {
    public let trainFeatures: [[Double]]
    public let trainTargets: [Double]
    public let valFeatures: [[Double]]
    public let valTargets: [Double]
}

// MARK: - KFold

/// Splits a dataset into K folds for cross-validation.
public struct KFold: Sendable {
    public let nSplits: Int
    public let shuffle: Bool
    public let seed: Int

    public init(nSplits: Int = 5, shuffle: Bool = true, seed: Int = 42) {
        precondition(nSplits >= 2, "KFold requires at least 2 splits")
        self.nSplits = nSplits
        self.shuffle = shuffle
        self.seed = seed
    }

    /// Returns K Fold objects for the given dataset.
    public func split(features: [[Double]], targets: [Double]) -> [Fold] {
        let n = features.count
        var indices = Array(0..<n)

        if shuffle {
            var rng = SeededRandom(seed: seed)
            for i in stride(from: n - 1, through: 1, by: -1) {
                let j = rng.nextInt(upperBound: i + 1)
                indices.swapAt(i, j)
            }
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

public struct CrossValidationResult: Sendable {
    public let scores: [Double]
    public let mean: Double
    public let std: Double

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
