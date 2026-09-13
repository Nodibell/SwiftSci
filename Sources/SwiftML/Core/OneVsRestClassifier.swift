import Foundation

/// Multi-class One-Vs-Rest classification wrapper.
public actor OneVsRestClassifier: Sendable {
    /// The num classes.
    public let numClasses: Int
    private var estimators: [LogisticRegression] = []

    /// Creates a new instance.
    /// - Parameters:
    ///   - numClasses: The num classes.
    public init(numClasses: Int) {
        self.numClasses = numClasses
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
    ///   - targets: 1D class label array [numSamples] where each value is a class index 0..<numClasses
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
        guard !features.isEmpty, features.count == targets.count else {
            throw SwiftMLError.emptyInput
        }

        let numClasses = self.numClasses
        let trained: [(Int, LogisticRegression)] = try await withThrowingTaskGroup(of: (Int, LogisticRegression).self) { group in
            for c in 0..<numClasses {
                let binaryTargets = targets.map { Int($0) == c ? 1.0 : 0.0 }
                group.addTask {
                    let est = LogisticRegression(device: .auto)
                    try await est.fit(features: features, targets: binaryTargets, learningRate: learningRate, epochs: epochs)
                    return (c, est)
                }
            }

            var results: [(Int, LogisticRegression)] = []
            results.reserveCapacity(numClasses)
            for try await res in group {
                results.append(res)
                onProgress?(results.count, numClasses)
            }
            return results.sorted { $0.0 < $1.0 }
        }

        self.estimators = trained.map { $0.1 }
    }

    /// Predicts class index for feature vectors.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    /// - Throws: `SwiftMLError` if feature-target dimensions mismatch, inputs are empty, or optimization fails.
    /// - Returns: Array of predicted discrete class labels for input observations.
    public func predict(features: [[Double]]) async throws -> [Int] {
        guard !estimators.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        
        var classProbs = [[Double]](repeating: [Double](repeating: 0.0, count: numClasses), count: features.count)
        
        for (c, est) in estimators.enumerated() {
            let probs = try await est.predictProbability(features: features)
            for i in 0..<features.count {
                let p1 = probs[i].count > 1 ? probs[i][1] : probs[i][0]
                classProbs[i][c] = p1
            }
        }

        
        return classProbs.map { row in
            row.argmax()
        }
    }

    // MARK: - Model Interpretability

    /// Returns learned weights for a specific binary sub-estimator.
    /// - Parameter c: Class index (0..<numClasses).
    /// - Returns: Array of learned weights corresponding to each feature dimension.
    public func getWeights(forClass c: Int) async throws -> [Double] {
        guard !estimators.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        guard c >= 0 && c < estimators.count else {
            throw SwiftMLError.indexOutOfRange(index: c, count: estimators.count)
        }
        let (weights, _) = await estimators[c].getWeightsAndBias()
        guard let w = weights else {
            throw SwiftMLError.modelNotFitted
        }
        return w
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
