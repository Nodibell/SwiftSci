import Foundation
import SwiftML

/// Actor-based Multinomial Naive Bayes classifier conforming to `ClassifierEstimator`.
public actor NaiveBayesClassifier: ClassifierEstimator {
    /// Additive Laplace smoothing hyperparameter.
    public let alpha: Double

    /// Learned target classes (numeric representations).
    public private(set) var classes: [Double] = []
    
    private var classLogPriors: [Double: Double] = [:]
    private var featureLogProbs: [Double: [Double]] = [:]
    private var isFitted: Bool = false

    /// Creates a new `NaiveBayesClassifier` instance.
    /// - Parameter alpha: Additive Laplace smoothing hyperparameter. Defaults to 1.0.
    public init(alpha: Double = 1.0) {
        self.alpha = max(0.0, alpha)
    }

    /// Fits the classifier model given a feature count matrix and target array.
    public func fit(features: [[Double]], targets: [Double]) async throws {
        guard !features.isEmpty, !features[0].isEmpty, features.count == targets.count else {
            throw SwiftMLError.invalidInput("Features and targets must not be empty and must have matching lengths.")
        }

        let nSamples = features.count
        let nFeatures = features[0].count

        let labelSet = Array(Set(targets)).sorted()
        self.classes = labelSet

        var classCounts: [Double: Int] = [:]
        var classFeatureSums: [Double: [Double]] = [:]

        for c in labelSet {
            classCounts[c] = 0
            classFeatureSums[c] = Array(repeating: 0.0, count: nFeatures)
        }

        for (rowIdx, label) in targets.enumerated() {
            classCounts[label, default: 0] += 1
            for colIdx in 0..<nFeatures {
                classFeatureSums[label]?[colIdx] += features[rowIdx][colIdx]
            }
        }

        var logPriors: [Double: Double] = [:]
        var logProbs: [Double: [Double]] = [:]

        for c in labelSet {
            let count = Double(classCounts[c] ?? 0)
            logPriors[c] = log(count / Double(nSamples))

            let featureSums = classFeatureSums[c] ?? Array(repeating: 0.0, count: nFeatures)
            let totalSum = featureSums.reduce(0.0, +) + alpha * Double(nFeatures)

            var featureLogProbRow: [Double] = Array(repeating: 0.0, count: nFeatures)
            for j in 0..<nFeatures {
                let smoothedNum = featureSums[j] + alpha
                featureLogProbRow[j] = log(smoothedNum / totalSum)
            }
            logProbs[c] = featureLogProbRow
        }

        self.classLogPriors = logPriors
        self.featureLogProbs = logProbs
        self.isFitted = true
    }

    /// Predicts target class integer indices for a given feature matrix.
    public func predict(features: [[Double]]) async throws -> [Int] {
        let probs = try await predictProbability(features: features)
        return probs.map { row in
            row.argmax()
        }
    }

    /// Predicts class probabilities for each feature sample.
    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        guard isFitted, !classes.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        guard !features.isEmpty else {
            return []
        }

        var results: [[Double]] = []
        results.reserveCapacity(features.count)

        for x in features {
            var logPosteriors: [Double] = []
            for c in classes {
                guard let prior = classLogPriors[c], let fProbs = featureLogProbs[c], fProbs.count == x.count else {
                    logPosteriors.append(-Double.infinity)
                    continue
                }
                var lp = prior
                for (j, val) in x.enumerated() {
                    if val > 0 {
                        lp += val * fProbs[j]
                    }
                }
                logPosteriors.append(lp)
            }

            let maxLog = logPosteriors.max() ?? 0.0
            let exps = logPosteriors.map { exp($0 - maxLog) }
            let sumExp = exps.reduce(0.0, +)
            let probs = sumExp > 0 ? exps.map { $0 / sumExp } : Array(repeating: 1.0 / Double(classes.count), count: classes.count)
            results.append(probs)
        }

        return results
    }
}
