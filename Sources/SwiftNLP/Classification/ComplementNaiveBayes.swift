import Foundation

/// Pure Swift implementation of Complement Naive Bayes classifier optimized for imbalanced text corpora.
public struct ComplementNaiveBayes: Sendable {
    public let alpha: Double

    public private(set) var featureWeights: [String: [Double]] = [:]
    public private(set) var classes: [String] = []

    public init(alpha: Double = 1.0) {
        self.alpha = max(0.0, alpha)
    }

    /// Fits Complement Naive Bayes on count matrix X and class label array y.
    public mutating func fit(X: [[Double]], y: [String]) {
        guard !X.isEmpty, X.count == y.count else { return }

        let nFeatures = X[0].count


        let labelSet = Array(Set(y)).sorted()
        self.classes = labelSet

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

        self.featureWeights = weights
    }

    /// Predicts the class label (minimizing complement weight).
    public func predict(x: [Double]) -> String? {
        guard !classes.isEmpty else { return nil }

        var minScore = Double.infinity
        var bestClass: String? = nil

        for c in classes {
            guard let w = featureWeights[c], w.count == x.count else { continue }
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

        return bestClass
    }

    public func predict(X: [[Double]]) -> [String] {
        return X.compactMap { predict(x: $0) }
    }
}
