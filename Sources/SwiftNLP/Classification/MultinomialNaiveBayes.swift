import Foundation

/// Pure Swift implementation of Multinomial Naive Bayes text classifier with Laplace smoothing.
@available(*, deprecated, renamed: "NaiveBayesClassifier", message: "Use actor NaiveBayesClassifier conforming to ClassifierEstimator instead.")
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
    public mutating func fit(X: [[Double]], y: [String]) {
        guard !X.isEmpty, X.count == y.count else { return }

        let nSamples = X.count
        let nFeatures = X[0].count

        let labelSet = Array(Set(y)).sorted()
        self.classes = labelSet

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

        self.classLogPriors = logPriors
        self.featureLogProbs = logProbs
    }

    /// Predicts the class label for a given feature vector.
    public func predict(x: [Double]) -> String? {
        guard !classes.isEmpty else { return nil }

        var maxLogProb = -Double.infinity
        var bestClass: String? = nil

        for c in classes {
            guard let prior = classLogPriors[c], let featureProbs = featureLogProbs[c], featureProbs.count == x.count else {
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

        return bestClass
    }

    /// Batch predicts class labels for a feature matrix.
    public func predict(X: [[Double]]) -> [String] {
        let fallback = classes.first ?? ""
        return X.map { predict(x: $0) ?? fallback }
    }
}
