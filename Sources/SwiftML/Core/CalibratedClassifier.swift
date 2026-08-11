import Foundation

/// CalibratedClassifier wraps a base ClassifierEstimator and applies Platt Scaling (logistic sigmoid calibration)
/// to produce well-calibrated class probabilities.
public actor CalibratedClassifier: ClassifierEstimator {
    /// The base estimator.
    public let baseEstimator: any ClassifierEstimator
    private var a: Double = 1.0
    private var b: Double = 0.0
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - baseEstimator: The base estimator.
    public init(baseEstimator: any ClassifierEstimator) {
        self.baseEstimator = baseEstimator
    }
    
    /// Fit.
    /// - Parameters:
    ///   - features: The features.
    ///   - targets: The targets.
    /// - Throws: An error if the operation fails.
    public func fit(features: [[Double]], targets: [Double]) async throws {
        try await baseEstimator.fit(features: features, targets: targets)
        
        let rawProbs = try await baseEstimator.predictProbability(features: features)
        guard !rawProbs.isEmpty else { return }
        
        // Extract probability of class 1
        let p1 = rawProbs.map { $0.count > 1 ? $0[1] : $0[0] }
        let numSamples = p1.count
        
        // Fit Platt Scaling sigmoid parameters (a * z + b) via gradient descent
        var paramA = 1.0
        var paramB = 0.0
        let lr = 0.1
        
        for _ in 0..<200 {
            var gradA = 0.0
            var gradB = 0.0
            for i in 0..<numSamples {
                let z = paramA * p1[i] + paramB
                let calP = sigmoid(z)
                let diff = calP - targets[i]
                gradA += diff * p1[i]
                gradB += diff
            }
            paramA -= lr * (gradA / Double(numSamples))
            paramB -= lr * (gradB / Double(numSamples))
        }
        
        self.a = paramA
        self.b = paramB
    }
    
    /// Predict.
    /// - Parameters:
    ///   - features: The features.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[Int]` result.
    public func predict(features: [[Double]]) async throws -> [Int] {
        let probs = try await predictProbability(features: features)
        return probs.map { $0[1] >= 0.5 ? 1 : 0 }
    }
    
    /// Predict probability.
    /// - Parameters:
    ///   - features: The features.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[[Double]]` result.
    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        let rawProbs = try await baseEstimator.predictProbability(features: features)
        return rawProbs.map { row in
            let rawP1 = row.count > 1 ? row[1] : row[0]
            let z = a * rawP1 + b
            let calP1 = sigmoid(z)
            return [1.0 - calP1, calP1]
        }
    }
}
