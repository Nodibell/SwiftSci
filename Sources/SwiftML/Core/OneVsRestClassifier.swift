import Foundation

/// Multi-class One-Vs-Rest classification wrapper.
public actor OneVsRestClassifier: ClassifierEstimator {
    /// The number of target classes configured for the classifier.
    public let numClasses: Int
    /// Sorted unique class labels observed during training.
    public private(set) var classes: [Double] = []
    private var estimators: [LogisticRegression] = []
    private var numFeatures: Int = 0

    /// Returns the number of input features observed during training.
    internal func getNumFeatures() -> Int { numFeatures }

    /// Creates a new instance.
    /// - Parameters:
    ///   - numClasses: The num classes.
    public init(numClasses: Int) {
        self.numClasses = numClasses
    }

    /// Fits one binary classifier per class with default optimization hyperparameters.
    /// - Parameters:
    ///   - features: Feature matrix `[numSamples × numFeatures]`.
    ///   - targets: 1D class label array `[numSamples]`.
    /// - Throws: `SwiftMLError` if inputs are empty, non-finite, or dimensions mismatch.
    public func fit(features: [[Double]], targets: [Double]) async throws {
        try await fit(features: features, targets: targets, learningRate: 2.0, epochs: 1000, onProgress: nil)
    }

    /// Fits one binary classifier per class against all other classes concurrently across CPU/GPU cores using structured TaskGroups.
    ///
    /// ## Concurrency Management
    /// Utilizes `withThrowingTaskGroup` with cooperative task cancellation and deterministic class ordering.
    ///
    /// ## Thread Safety
    /// Thread-safe via actor isolation.
    ///
    /// - Parameters:
    ///   - features: Feature matrix [numSamples × numFeatures]
    ///   - targets: 1D class label array [numSamples] where each value is a class label
    ///   - learningRate: Gradient descent step size. Defaults to 2.0.
    ///   - epochs: Number of training epochs. Defaults to 1000.
    ///   - onProgress: Optional progress callback (completedClasses, totalClasses).
    /// - Throws: `SwiftMLError` if inputs are empty or dimensions mismatch.
    public func fit(
        features: [[Double]],
        targets: [Double],
        learningRate: Float = 2.0,
        epochs: Int = 1000,
        onProgress: (@Sendable (Int, Int) -> Void)? = nil
    ) async throws {
        guard !features.isEmpty, !features[0].isEmpty, features.count == targets.count else {
            throw SwiftMLError.emptyInput
        }
        guard numClasses >= 1 else {
            throw SwiftMLError.invalidParameter("numClasses must be at least 1")
        }

        let featureWidth = features[0].count
        for row in features {
            guard row.count == featureWidth else {
                throw SwiftMLError.dimensionMismatch(expected: featureWidth, got: row.count)
            }
            guard row.allSatisfy(\.isFinite) else {
                throw SwiftMLError.invalidParameter("Training features must be finite")
            }
        }
        guard targets.allSatisfy(\.isFinite) else {
            throw SwiftMLError.invalidParameter("Training targets must be finite")
        }

        let uniqueClasses = Array(Set(targets)).sorted()
        let resolvedClasses: [Double]
        if uniqueClasses.count == numClasses {
            resolvedClasses = uniqueClasses
        } else if uniqueClasses.count < numClasses,
                  uniqueClasses.allSatisfy({ (val: Double) -> Bool in
                      val >= 0 && val < Double(numClasses) && val == Double(Int(val))
                  }) {
            resolvedClasses = (0..<numClasses).map { (c: Int) -> Double in Double(c) }
        } else {
            throw SwiftMLError.invalidParameter(
                "Unique target class count (\(uniqueClasses.count)) does not match numClasses (\(numClasses))"
            )
        }

        let totalClasses = self.numClasses
        let trained: [(Int, LogisticRegression)] = try await withThrowingTaskGroup(of: (Int, LogisticRegression).self) { group in
            for c in 0..<totalClasses {
                let classValue = resolvedClasses[c]
                let binaryTargets = targets.map { (target: Double) -> Double in target == classValue ? 1.0 : 0.0 }
                group.addTask {
                    let est = LogisticRegression(device: .auto)
                    try await est.fit(features: features, targets: binaryTargets, learningRate: learningRate, epochs: epochs)
                    return (c, est)
                }
            }

            var results: [(Int, LogisticRegression)] = []
            results.reserveCapacity(totalClasses)
            for try await res in group {
                results.append(res)
                onProgress?(results.count, totalClasses)
            }
            return results.sorted { (lhs: (Int, LogisticRegression), rhs: (Int, LogisticRegression)) -> Bool in
                lhs.0 < rhs.0
            }
        }

        self.estimators = trained.map { (pair: (Int, LogisticRegression)) -> LogisticRegression in pair.1 }
        self.classes = resolvedClasses
        self.numFeatures = featureWidth
    }

    /// Computes raw linear decision function logits `z_{i,c} = w_c^T x_i + b_c` for each sample and class.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    /// - Throws: `SwiftMLError.modelNotFitted` if not fitted, `SwiftMLError.emptyInput` if `features` is empty,
    ///   or `SwiftMLError.dimensionMismatch` / `SwiftMLError.containsNaN` / `SwiftMLError.containsInfinity` on invalid inputs.
    /// - Returns: 2D array of logits of shape `[N, classes.count]`.
    public func decisionFunction(features: [[Double]]) async throws -> [[Double]] {
        guard !estimators.isEmpty, !classes.isEmpty, numFeatures > 0 else {
            throw SwiftMLError.modelNotFitted
        }
        guard !features.isEmpty else {
            throw SwiftMLError.emptyInput
        }

        for row in features {
            guard row.count == numFeatures else {
                throw SwiftMLError.dimensionMismatch(expected: numFeatures, got: row.count)
            }
            if row.contains(where: \.isNaN) {
                throw SwiftMLError.containsNaN
            }
            guard row.allSatisfy(\.isFinite) else {
                throw SwiftMLError.containsInfinity
            }
        }

        let cCount = classes.count
        var classParams: [(weights: [Double], bias: Double)] = []
        classParams.reserveCapacity(cCount)
        for c in 0..<cCount {
            let (weights, bias) = try await getWeightsAndBias(forClass: c)
            classParams.append((weights, bias))
        }

        var logits: [[Double]] = []
        logits.reserveCapacity(features.count)
        for row in features {
            var rowLogits: [Double] = []
            rowLogits.reserveCapacity(cCount)
            for c in 0..<cCount {
                let w = classParams[c].weights
                var z = classParams[c].bias
                for j in 0..<numFeatures {
                    z += w[j] * row[j]
                }
                rowLogits.append(z)
            }
            logits.append(rowLogits)
        }
        return logits
    }

    /// Predicts L1-normalized One-Vs-Rest class probabilities of shape `[N, classes.count]`.
    ///
    /// Computes marginal probabilities `p_{i,c} = sigmoid(z_{i,c})` for each binary sub-estimator
    /// and normalizes each row by `S_i = sum_c p_{i,c}`, falling back to uniform `[1/C, ...]` if `S_i <= 0` or non-finite.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    /// - Throws: `SwiftMLError` if the model is not fitted or input validation fails.
    /// - Returns: 2D array of normalized class probability vectors of shape `[N, classes.count]`.
    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        let logits = try await decisionFunction(features: features)
        let cCount = classes.count
        let uniform = [Double](repeating: 1.0 / Double(cCount), count: cCount)

        var probabilities: [[Double]] = []
        probabilities.reserveCapacity(logits.count)
        for row in logits {
            var marginals: [Double] = []
            marginals.reserveCapacity(cCount)
            var sum = 0.0
            for z in row {
                let p = sigmoid(z)
                marginals.append(p)
                sum += p
            }
            if !sum.isFinite || sum <= 0.0 {
                probabilities.append(uniform)
            } else {
                let normalized = marginals.map { (p: Double) -> Double in p / sum }
                probabilities.append(normalized)
            }
        }
        return probabilities
    }

    /// Predicts class index for feature vectors.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    /// - Throws: `SwiftMLError` if feature-target dimensions mismatch, inputs are empty, or model is unfitted.
    /// - Returns: Array of predicted discrete class labels for input observations.
    public func predict(features: [[Double]]) async throws -> [Int] {
        guard !estimators.isEmpty, !classes.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        guard !features.isEmpty else {
            return []
        }

        let probabilities = try await predictProbability(features: features)
        let fittedClasses = self.classes
        return probabilities.map { (row: [Double]) -> Int in
            Int(fittedClasses[row.argmax()])
        }
    }

    // MARK: - Model Interpretability

    /// Returns learned bias (intercept) for a specific binary sub-estimator.
    /// - Parameter c: Class index (`0..<classes.count`).
    /// - Throws: `SwiftMLError.modelNotFitted` if not fitted or `SwiftMLError.indexOutOfRange` if `c` is invalid.
    /// - Returns: Learned scalar bias term for class `c`.
    public func getBias(forClass c: Int) async throws -> Double {
        let (_, bias) = try await getWeightsAndBias(forClass: c)
        return bias
    }

    /// Returns learned weights and bias for a specific binary sub-estimator.
    /// - Parameter c: Class index (`0..<classes.count`).
    /// - Throws: `SwiftMLError.modelNotFitted` if not fitted or `SwiftMLError.indexOutOfRange` if `c` is invalid.
    /// - Returns: Tuple `(weights, bias)` for class `c`.
    public func getWeightsAndBias(forClass c: Int) async throws -> (weights: [Double], bias: Double) {
        guard !estimators.isEmpty, !classes.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        guard c >= 0 && c < classes.count else {
            throw SwiftMLError.indexOutOfRange(index: c, count: classes.count)
        }
        let (weights, bias) = await estimators[c].getWeightsAndBias()
        guard let w = weights, let b = bias else {
            throw SwiftMLError.modelNotFitted
        }
        return (w, b)
    }

    /// Returns learned weights for a specific binary sub-estimator.
    /// - Parameter c: Class index (`0..<classes.count`).
    /// - Throws: `SwiftMLError.modelNotFitted` if not fitted or `SwiftMLError.indexOutOfRange` if `c` is invalid.
    /// - Returns: Array of learned weights corresponding to each feature dimension.
    public func getWeights(forClass c: Int) async throws -> [Double] {
        let (weights, _) = try await getWeightsAndBias(forClass: c)
        return weights
    }

    /// Extracts the most influential features (highest positive weights) for a target class.
    /// - Parameters:
    ///   - classIndex: The target class index.
    ///   - topN: Maximum number of features to return (default: 10).
    ///   - featureNames: Optional list of human-readable feature or column names.
    /// - Returns: Ranked list of `FeatureImportance` values in descending order of weight.
    public func topFeatures(
        classIndex: Int,
        topN: Int = 10,
        featureNames: [String]? = nil
    ) async throws -> [FeatureImportance] {
        let weights = try await getWeights(forClass: classIndex)
        let n = weights.count
        var items: [FeatureImportance] = []
        items.reserveCapacity(n)
        for i in 0..<n {
            let name = (featureNames != nil && i < featureNames!.count) ? featureNames![i] : "feature_\(i)"
            items.append(FeatureImportance(feature: name, weight: weights[i], index: i))
        }
        items.sort { $0.weight > $1.weight }
        return Array(items.prefix(max(1, topN)))
    }

    /// Extracts the most influential vocabulary terms for a target class using a token-to-index vocabulary dictionary.
    /// - Parameters:
    ///   - classIndex: The target class index.
    ///   - topN: Maximum number of terms to return.
    ///   - vocabulary: Dictionary mapping string tokens to matrix column indices (e.g. from TFIDFVectorizer).
    /// - Returns: Ranked list of `FeatureImportance` values.
    public func topFeatures(
        classIndex: Int,
        topN: Int = 10,
        vocabulary: [String: Int]
    ) async throws -> [FeatureImportance] {
        let weights = try await getWeights(forClass: classIndex)
        var names = [String](repeating: "", count: weights.count)
        for i in 0..<weights.count {
            names[i] = "token_\(i)"
        }
        for (term, idx) in vocabulary {
            if idx >= 0 && idx < weights.count {
                names[idx] = term
            }
        }
        return try await topFeatures(classIndex: classIndex, topN: topN, featureNames: names)
    }

    /// Returns the top features for each class in the model.
    /// - Parameters:
    ///   - topN: Number of top features per class.
    ///   - featureNames: Optional list of human-readable feature names.
    /// - Returns: Dictionary mapping each class index to its ranked list of `FeatureImportance`.
    public func topFeaturesPerClass(
        topN: Int = 10,
        featureNames: [String]? = nil
    ) async throws -> [Int: [FeatureImportance]] {
        var results: [Int: [FeatureImportance]] = [:]
        for c in 0..<numClasses {
            results[c] = try await topFeatures(classIndex: c, topN: topN, featureNames: featureNames)
        }
        return results
    }

    /// Returns the top features for each class using a token vocabulary dictionary.
    public func topFeaturesPerClass(
        topN: Int = 10,
        vocabulary: [String: Int]
    ) async throws -> [Int: [FeatureImportance]] {
        var results: [Int: [FeatureImportance]] = [:]
        for c in 0..<numClasses {
            results[c] = try await topFeatures(classIndex: c, topN: topN, vocabulary: vocabulary)
        }
        return results
    }
}
