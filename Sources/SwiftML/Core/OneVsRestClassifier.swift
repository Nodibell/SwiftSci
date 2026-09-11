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
    ///   - features: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
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
}
