import Foundation
import SwiftML
import Accelerate

/// Complement Naive Bayes classifier designed for imbalanced text datasets, conforming to `ClassifierEstimator`.
public actor ComplementNaiveBayesClassifier: ClassifierEstimator {
    /// Additive Laplace smoothing hyperparameter.
    public let alpha: Double

    /// Learned target classes.
    public private(set) var classes: [Double] = []

    private var featureWeights: [Double: [Double]] = [:]
    private var isFitted: Bool = false

    /// Creates a new `ComplementNaiveBayesClassifier` instance.
    /// - Parameter alpha: Additive Laplace smoothing hyperparameter. Defaults to 1.0.
    public init(alpha: Double = 1.0) {
        self.alpha = max(0.0, alpha)
    }

    /// Fits Complement Naive Bayes on count matrix X and class target array y.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    ///   - targets: 1D array of ground-truth target values of length `N`.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    public func fit(features: [[Double]], targets: [Double]) async throws {
        guard !features.isEmpty, !features[0].isEmpty, features.count == targets.count else {
            throw SwiftMLError.invalidInput("Features and targets must not be empty and must have matching lengths.")
        }

        let nSamples = features.count
        let nFeatures = features[0].count
        let labelSet = Array(Set(targets)).sorted()
        self.classes = labelSet
        let nClasses = labelSet.count

        var classIndexMap: [Double: Int] = [:]
        for (idx, c) in labelSet.enumerated() {
            classIndexMap[c] = idx
        }

        var totalFeatureSums = [Double](repeating: 0.0, count: nFeatures)
        var flatClassSums = [Double](repeating: 0.0, count: nClasses * nFeatures)

        for rowIdx in 0..<nSamples {
            let label = targets[rowIdx]
            guard let cIdx = classIndexMap[label] else { continue }
            let offset = cIdx * nFeatures
            let row = features[rowIdx]
            for colIdx in 0..<nFeatures {
                let val = row[colIdx]
                totalFeatureSums[colIdx] += val
                flatClassSums[offset + colIdx] += val
            }
        }

        var weights: [Double: [Double]] = [:]

        for (cIdx, c) in labelSet.enumerated() {
            let offset = cIdx * nFeatures
            var complementSums = [Double](repeating: 0.0, count: nFeatures)
            var totalComplementSum = 0.0

            for j in 0..<nFeatures {
                let compVal = totalFeatureSums[j] - flatClassSums[offset + j]
                complementSums[j] = compVal
                totalComplementSum += compVal
            }

            let denominator = totalComplementSum + alpha * Double(nFeatures)
            var weightRow = [Double](repeating: 0.0, count: nFeatures)
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
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    /// - Returns: Array of predicted discrete class labels for input observations.
    public func predict(features: [[Double]]) async throws -> [Int] {
        let probs = try await predictProbability(features: features)
        return probs.map { $0.argmax() }
    }

    /// Predicts normalized probability scores for each class.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    /// - Returns: 2D array of predicted class probabilities across samples of shape `[N, K]`.
    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        guard isFitted, !classes.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        guard !features.isEmpty else { return [] }

        let nClasses = classes.count
        let weightsArray = classes.map { featureWeights[$0] ?? [] }

        var results: [[Double]] = []
        results.reserveCapacity(features.count)

        for x in features {
            var negScores = [Double](repeating: 0.0, count: nClasses)
            for cIdx in 0..<nClasses {
                let w = weightsArray[cIdx]
                guard w.count == x.count else {
                    negScores[cIdx] = Double.infinity
                    continue
                }
                var dotVal = 0.0
                vDSP_dotprD(x, 1, w, 1, &dotVal, vDSP_Length(x.count))
                negScores[cIdx] = dotVal
            }

            let minScore = negScores.min() ?? 0.0
            var exps = [Double](repeating: 0.0, count: nClasses)
            var sumExp = 0.0
            for cIdx in 0..<nClasses {
                let val = exp(-(negScores[cIdx] - minScore))
                exps[cIdx] = val
                sumExp += val
            }

            let probs = sumExp > 0 ? exps.map { $0 / sumExp } : Array(repeating: 1.0 / Double(nClasses), count: nClasses)
            results.append(probs)
        }

        return results
    }
}
