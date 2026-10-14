import Foundation
import SwiftDataFrame

/// Pure Swift implementation of Multinomial Naive Bayes text classifier with Laplace smoothing.
public struct MultinomialNaiveBayes: Sendable, Codable {

    /// Additive Laplace smoothing parameter (alpha >= 0).
    public let alpha: Double

    /// Learned class prior probabilities (log scale).
    public private(set) var classLogPriors: [String: Double] = [:]
    /// Learned feature log probabilities [class: [featureIndex: logProb]].
    public private(set) var featureLogProbs: [String: [Double]] = [:]
    /// List of target class labels.
    public private(set) var classes: [String] = []

    /// Creates a Multinomial Naive Bayes text classifier instance.
    /// - Parameter alpha: Additive Laplace smoothing hyperparameter. Defaults to 1.0.
    public init(alpha: Double = 1.0) {
        self.alpha = max(0.0, alpha)
    }

    /// Fits the classifier model given a feature count matrix and label array.
    /// - Parameters:
    ///   - X: Feature matrix (row: document, col: token count or TF-IDF feature).
    ///   - y: Class labels corresponding to each row.
    /// - Throws: `SwiftMLError` if inputs are empty, dimensions mismatch, or values are negative/non-finite.
    public mutating func fit(X: [[Double]], y: [String]) throws {
        guard !X.isEmpty else {
            throw SwiftMLError.emptyInput
        }
        guard X.count == y.count else {
            throw SwiftMLError.dimensionMismatch(expected: X.count, got: y.count)
        }

        let nSamples = X.count
        let nFeatures = X[0].count
        guard nFeatures > 0 else {
            throw SwiftMLError.invalidInput("Feature matrix cannot have zero columns.")
        }

        for (rowIdx, row) in X.enumerated() {
            guard row.count == nFeatures else {
                throw SwiftMLError.dimensionMismatch(expected: nFeatures, got: row.count)
            }
            for (colIdx, val) in row.enumerated() {
                if val.isNaN {
                    throw SwiftMLError.containsNaN
                }
                if val.isInfinite {
                    throw SwiftMLError.containsInfinity
                }
                if val < 0.0 {
                    throw SwiftMLError.invalidInput("Multinomial Naive Bayes requires non-negative feature counts, but found \(val) at row \(rowIdx), column \(colIdx).")
                }
            }
        }

        let labelSet = Array(Set(y)).sorted()
        guard !labelSet.isEmpty else {
            throw SwiftMLError.invalidInput("Target labels array cannot be empty.")
        }

        var classCounts: [String: Int] = [:]
        var classFeatureSums: [String: [Double]] = [:]

        for c in labelSet {
            classCounts[c] = 0
            classFeatureSums[c] = Array(repeating: 0.0, count: nFeatures)
        }

        for (rowIdx, label) in y.enumerated() {
            classCounts[label, default: 0] += 1
            for colIdx in 0..<nFeatures {
                classFeatureSums[label]?[colIdx] += X[rowIdx][colIdx]
            }
        }

        var logPriors: [String: Double] = [:]
        var logProbs: [String: [Double]] = [:]

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

        self.classes = labelSet
        self.classLogPriors = logPriors
        self.featureLogProbs = logProbs
    }

    /// Predicts the class label for a given feature vector.
    /// - Parameter x: Input numeric feature vector.
    /// - Returns: Predicted class label string.
    /// - Throws: `SwiftMLError` if model is unfitted, dimensions mismatch, or values are non-finite or negative.
    public func predict(x: [Double]) throws -> String {
        guard !classes.isEmpty, !featureLogProbs.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        let expectedFeatures = featureLogProbs[classes[0]]?.count ?? 0
        guard x.count == expectedFeatures else {
            throw SwiftMLError.dimensionMismatch(expected: expectedFeatures, got: x.count)
        }
        for (idx, val) in x.enumerated() {
            if val.isNaN { throw SwiftMLError.containsNaN }
            if val.isInfinite { throw SwiftMLError.containsInfinity }
            if val < 0.0 {
                throw SwiftMLError.invalidInput("Multinomial Naive Bayes requires non-negative feature counts, but found \(val) at index \(idx).")
            }
        }

        var maxLogProb = -Double.infinity
        var bestClass: String? = nil

        for c in classes {
            guard let prior = classLogPriors[c], let featureProbs = featureLogProbs[c] else {
                continue
            }
            var logPosterior = prior
            for (j, val) in x.enumerated() {
                if val > 0 {
                    logPosterior += val * featureProbs[j]
                }
            }
            if logPosterior > maxLogProb {
                maxLogProb = logPosterior
                bestClass = c
            }
        }

        guard let predicted = bestClass else {
            throw SwiftMLError.trainingFailed("Failed to compute class log posterior probabilities.")
        }
        return predicted
    }

    /// Batch predicts class labels for a feature matrix.
    /// - Parameter X: Matrix of sample feature vectors.
    /// - Returns: Array of predicted class strings matching the length of `X`.
    /// - Throws: `SwiftMLError` if model is unfitted, row dimensions mismatch, or values are non-finite or negative.
    public func predict(X: [[Double]]) throws -> [String] {
        guard !classes.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        guard !X.isEmpty else { return [] }
        return try X.map { try predict(x: $0) }
    }
}
