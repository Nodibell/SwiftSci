import Foundation
import SwiftML
import Accelerate

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

        var classCounts = [Int](repeating: 0, count: nClasses)
        var flatFeatureSums = [Double](repeating: 0.0, count: nClasses * nFeatures)

        for rowIdx in 0..<nSamples {
            let label = targets[rowIdx]
            guard let cIdx = classIndexMap[label] else { continue }
            classCounts[cIdx] += 1
            let offset = cIdx * nFeatures
            let row = features[rowIdx]
            for colIdx in 0..<nFeatures {
                flatFeatureSums[offset + colIdx] += row[colIdx]
            }
        }

        var logPriors: [Double: Double] = [:]
        var logProbs: [Double: [Double]] = [:]

        for (cIdx, c) in labelSet.enumerated() {
            let count = Double(classCounts[cIdx])
            logPriors[c] = log(count / Double(nSamples))

            let offset = cIdx * nFeatures
            var totalSum = alpha * Double(nFeatures)
            for j in 0..<nFeatures {
                totalSum += flatFeatureSums[offset + j]
            }

            var featureLogProbRow = [Double](repeating: 0.0, count: nFeatures)
            for j in 0..<nFeatures {
                let smoothedNum = flatFeatureSums[offset + j] + alpha
                featureLogProbRow[j] = log(smoothedNum / totalSum)
            }
            logProbs[c] = featureLogProbRow
        }

        self.classLogPriors = logPriors
        self.featureLogProbs = logProbs
        self.isFitted = true
    }

    /// Predicts target class integer indices for a given feature matrix.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    /// - Returns: Array of predicted discrete class labels for input observations.
    public func predict(features: [[Double]]) async throws -> [Int] {
        let probs = try await predictProbability(features: features)
        return probs.map { row in
            row.argmax()
        }
    }

    /// Predicts class probabilities for each feature sample.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    /// - Returns: 2D array of predicted class probabilities across samples of shape `[N, K]`.
    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        guard isFitted, !classes.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        guard !features.isEmpty else {
            return []
        }

        let nClasses = classes.count
        let classPriorsArray = classes.map { classLogPriors[$0] ?? -Double.infinity }
        let classLogProbsArray = classes.map { featureLogProbs[$0] ?? [] }

        var results: [[Double]] = []
        results.reserveCapacity(features.count)

        for x in features {
            var logPosteriors = [Double](repeating: 0.0, count: nClasses)
            for cIdx in 0..<nClasses {
                let prior = classPriorsArray[cIdx]
                let fProbs = classLogProbsArray[cIdx]
                guard fProbs.count == x.count else {
                    logPosteriors[cIdx] = -Double.infinity
                    continue
                }
                var dotVal = 0.0
                vDSP_dotprD(x, 1, fProbs, 1, &dotVal, vDSP_Length(x.count))
                logPosteriors[cIdx] = prior + dotVal
            }

            let maxLog = logPosteriors.max() ?? 0.0
            var exps = [Double](repeating: 0.0, count: nClasses)
            var sumExp = 0.0
            for cIdx in 0..<nClasses {
                let expVal = exp(logPosteriors[cIdx] - maxLog)
                exps[cIdx] = expVal
                sumExp += expVal
            }
            let probs = sumExp > 0 ? exps.map { $0 / sumExp } : Array(repeating: 1.0 / Double(nClasses), count: nClasses)
            results.append(probs)
        }

        return results
    }
}
