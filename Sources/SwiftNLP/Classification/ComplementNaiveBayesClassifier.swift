import Foundation
import SwiftML

/// Actor-based Complement Naive Bayes classifier conforming to `ClassifierEstimator`.
public actor ComplementNaiveBayesClassifier: ClassifierEstimator {
    /// Additive Laplace smoothing hyperparameter.
    public let alpha: Double

    /// Learned target classes (numeric representations).
    public private(set) var classes: [Double] = []

    private var featureWeights: [Double: [Double]] = [:]
    private var isFitted: Bool = false

    /// Creates a new `ComplementNaiveBayesClassifier` instance.
    /// - Parameter alpha: Additive Laplace smoothing hyperparameter. Defaults to 1.0.
    public init(alpha: Double = 1.0) {
        self.alpha = max(0.0, alpha)
    }

    /// Fits Complement Naive Bayes on count matrix X and class target array y.
    public func fit(features: [[Double]], targets: [Double]) async throws {
        guard !features.isEmpty, !features[0].isEmpty, features.count == targets.count else {
            throw SwiftMLError.invalidInput("Features and targets must not be empty and must have matching lengths.")
        }

        let nFeatures = features[0].count
        let labelSet = Array(Set(targets)).sorted()
        self.classes = labelSet

        var totalFeatureSums = Array(repeating: 0.0, count: nFeatures)
        var classFeatureSums: [Double: [Double]] = [:]

        for c in labelSet {
            classFeatureSums[c] = Array(repeating: 0.0, count: nFeatures)
        }

        for (rowIdx, label) in targets.enumerated() {
            for colIdx in 0..<nFeatures {
                let val = features[rowIdx][colIdx]
                totalFeatureSums[colIdx] += val
                classFeatureSums[label]?[colIdx] += val
            }
        }

        var weights: [Double: [Double]] = [:]

        for c in labelSet {
            let classSums = classFeatureSums[c] ?? Array(repeating: 0.0, count: nFeatures)
            var complementSums = Array(repeating: 0.0, count: nFeatures)
            var totalComplementSum = 0.0

            for j in 0..<nFeatures {
                let compVal = totalFeatureSums[j] - classSums[j]
                complementSums[j] = compVal
                totalComplementSum += compVal
            }

            let denominator = totalComplementSum + alpha * Double(nFeatures)
            var weightRow = Array(repeating: 0.0, count: nFeatures)
            var weightSum = 0.0

            for j in 0..<nFeatures {
                let smoothedVal = complementSums[j] + alpha
                let w = log(smoothedVal / denominator)
                weightRow[j] = w
                weightSum += abs(w)
            }

            if weightSum > 0 {
                weightRow = weightRow.map { $0 / weightSum }
            }
            weights[c] = weightRow
        }

        self.featureWeights = weights
        self.isFitted = true
    }

    /// Predicts target class integer indices for a feature matrix.
    public func predict(features: [[Double]]) async throws -> [Int] {
        let probs = try await predictProbability(features: features)
        return probs.map { $0.argmax() }
    }

    /// Predicts normalized probability scores for each class.
    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        guard isFitted, !classes.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        guard !features.isEmpty else { return [] }

        var results: [[Double]] = []
        results.reserveCapacity(features.count)

        for x in features {
            var negScores: [Double] = []
            for c in classes {
                guard let w = featureWeights[c], w.count == x.count else {
                    negScores.append(Double.infinity)
                    continue
                }
                var score = 0.0
                for (j, val) in x.enumerated() {
                    if val > 0 {
                        score += val * w[j]
                    }
                }
                // CNB decision rule minimizes complement weight sum, so invert for probability
                negScores.append(-score)
            }

            let maxScore = negScores.max() ?? 0.0
            let exps = negScores.map { exp($0 - maxScore) }
            let sumExp = exps.reduce(0.0, +)
            let probs = sumExp > 0 ? exps.map { $0 / sumExp } : Array(repeating: 1.0 / Double(classes.count), count: classes.count)
            results.append(probs)
        }

        return results
    }
}
