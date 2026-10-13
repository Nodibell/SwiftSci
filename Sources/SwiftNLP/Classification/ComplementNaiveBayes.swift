import Foundation
import SwiftDataFrame

/// Pure Swift implementation of Complement Naive Bayes classifier optimized for imbalanced text corpora.
public struct ComplementNaiveBayes: Sendable, Codable {

    /// Additive Laplace smoothing parameter (alpha >= 0).
    public let alpha: Double

    public private(set) var featureWeights: [String: [Double]] = [:]
    public private(set) var classes: [String] = []

    /// Creates a Complement Naive Bayes classifier.
    /// - Parameter alpha: Laplace smoothing hyperparameter. Defaults to 1.0.
    public init(alpha: Double = 1.0) {
        self.alpha = max(0.0, alpha)
    }

    /// Fits Complement Naive Bayes on count matrix X and class label array y.
    /// - Parameters:
    ///   - X: Feature matrix (row: document, col: token count or TF-IDF feature).
    ///   - y: Class labels corresponding to each row.
    /// - Throws: `SwiftMLError` if inputs are empty, dimensions mismatch, values are non-finite/negative, or fewer than 2 classes are provided.
    public mutating func fit(X: [[Double]], y: [String]) throws {
        guard !X.isEmpty else {
            throw SwiftMLError.emptyInput
        }
        guard X.count == y.count else {
            throw SwiftMLError.dimensionMismatch(expected: X.count, got: y.count)
        }

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
                    throw SwiftMLError.invalidInput("Complement Naive Bayes requires non-negative feature counts, but found \(val) at row \(rowIdx), column \(colIdx).")
                }
            }
        }

        let labelSet = Array(Set(y)).sorted()
        guard labelSet.count >= 2 else {
            throw SwiftMLError.invalidInput("Complement Naive Bayes requires at least 2 distinct classes to compute complement statistics, but found \(labelSet.count).")
        }

        var totalFeatureSums = Array(repeating: 0.0, count: nFeatures)
        var classFeatureSums: [String: [Double]] = [:]

        for c in labelSet {
            classFeatureSums[c] = Array(repeating: 0.0, count: nFeatures)
        }

        for (rowIdx, label) in y.enumerated() {
            for colIdx in 0..<nFeatures {
                let val = X[rowIdx][colIdx]
                totalFeatureSums[colIdx] += val
                classFeatureSums[label]?[colIdx] += val
            }
        }

        var weights: [String: [Double]] = [:]

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
            guard denominator > 0.0 else {
                throw SwiftMLError.divisionByZero(context: "Complement Naive Bayes denominator for class '\(c)' is zero. Increase alpha > 0.")
            }
            var weightRow = Array(repeating: 0.0, count: nFeatures)
            var weightSum = 0.0

            for j in 0..<nFeatures {
                let smoothedVal = complementSums[j] + alpha
                let w = log(smoothedVal / denominator)
                weightRow[j] = w
                weightSum += abs(w)
            }

            // Normalize weights
            if weightSum > 0 {
                weightRow = weightRow.map { $0 / weightSum }
            }
            weights[c] = weightRow
        }

        self.classes = labelSet
        self.featureWeights = weights
    }

    /// Predicts the class label (minimizing complement weight).
    /// - Parameter x: Input numeric feature vector.
    /// - Returns: Predicted class label string.
    /// - Throws: `SwiftMLError` if model is unfitted, dimensions mismatch, or values are non-finite or negative.
    public func predict(x: [Double]) throws -> String {
        guard !classes.isEmpty, !featureWeights.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        let expectedFeatures = featureWeights[classes[0]]?.count ?? 0
        guard x.count == expectedFeatures else {
            throw SwiftMLError.dimensionMismatch(expected: expectedFeatures, got: x.count)
        }
        for (idx, val) in x.enumerated() {
            if val.isNaN { throw SwiftMLError.containsNaN }
            if val.isInfinite { throw SwiftMLError.containsInfinity }
            if val < 0.0 {
                throw SwiftMLError.invalidInput("Complement Naive Bayes requires non-negative feature counts, but found \(val) at index \(idx).")
            }
        }

        var minScore = Double.infinity
        var bestClass: String? = nil

        for c in classes {
            guard let w = featureWeights[c] else { continue }
            var score = 0.0
            for (j, val) in x.enumerated() {
                if val > 0 {
                    score += val * w[j]
                }
            }
            if score < minScore {
                minScore = score
                bestClass = c
            }
        }

        guard let predicted = bestClass else {
            throw SwiftMLError.trainingFailed("Failed to compute complement weight prediction.")
        }
        return predicted
    }

    /// Predicts target class label names for a batch of sample feature vectors.
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
