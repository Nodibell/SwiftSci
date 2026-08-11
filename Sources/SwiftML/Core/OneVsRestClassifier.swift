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

    /// Fits one binary classifier per class against all other classes.
    /// - Parameters:
    ///   - features: Feature matrix [numSamples × numFeatures]
    ///   - targets: 1D class label array [numSamples] where each value is a class index 0..<numClasses
    ///   - learningRate: Gradient descent step size. Defaults to 2.0.
    ///   - epochs: Number of training epochs. Defaults to 1000.
    ///   - onProgress: Optional progress callback (completedClasses, totalClasses).
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
        
        var newEstimators: [LogisticRegression] = []
        for c in 0..<numClasses {
            onProgress?(c, numClasses)
            let binaryTargets = targets.map { Int($0) == c ? 1.0 : 0.0 }
            let est = LogisticRegression(device: .auto)
            try await est.fit(features: features, targets: binaryTargets, learningRate: learningRate, epochs: epochs)
            newEstimators.append(est)
        }
        onProgress?(numClasses, numClasses)
        self.estimators = newEstimators
    }

    /// Predicts class index for feature vectors.
    public func predict(features: [[Double]]) async throws -> [Int] {
        guard !estimators.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        
        var classProbs = [[Double]](repeating: [Double](repeating: 0.0, count: numClasses), count: features.count)
        
        for (c, est) in estimators.enumerated() {
            let probs = try await est.predictProbability1D(features: features)
            for i in 0..<features.count {
                classProbs[i][c] = probs[i]
            }
        }
        
        return classProbs.map { row in
            row.argmax()
        }

    }
}
