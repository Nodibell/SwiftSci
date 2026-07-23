import Foundation

/// Multi-class One-Vs-Rest classification wrapper.
public actor OneVsRestClassifier: Sendable {
    public let numClasses: Int
    private var estimators: [LogisticRegression] = []

    public init(numClasses: Int) {
        self.numClasses = numClasses
    }

    /// Fits one binary classifier per class against all other classes.
    /// - Parameters:
    ///   - features: Feature matrix [numSamples × numFeatures]
    ///   - targets: 1D class label array [numSamples] where each value is a class index 0..<numClasses
    public func fit(features: [[Double]], targets: [Double]) async throws {
        guard !features.isEmpty, features.count == targets.count else {
            throw SwiftMLError.emptyInput
        }
        
        var newEstimators: [LogisticRegression] = []
        for c in 0..<numClasses {
            let binaryTargets = targets.map { Int($0) == c ? 1.0 : 0.0 }
            let est = LogisticRegression(device: .cpu)
            try await est.fit(features: features, targets: binaryTargets, learningRate: 0.5, epochs: 200)
            newEstimators.append(est)
        }
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
            row.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        }
    }
}
