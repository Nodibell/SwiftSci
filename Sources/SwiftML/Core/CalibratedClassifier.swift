import Foundation

/// CalibratedClassifier wraps a base ClassifierEstimator and applies Platt Scaling (logistic sigmoid calibration)
/// to produce well-calibrated class probabilities.
public final class CalibratedClassifier: ClassifierEstimator, @unchecked Sendable {
    public let baseEstimator: any ClassifierEstimator
    private var a: Double = 1.0
    private var b: Double = 0.0
    
    public init(baseEstimator: any ClassifierEstimator) {
        self.baseEstimator = baseEstimator
    }
    
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
                let calP = 1.0 / (1.0 + exp(-max(-50.0, min(50.0, z))))
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
    
    public func predict(features: [[Double]]) async throws -> [Int] {
        let probs = try await predictProbability(features: features)
        return probs.map { $0[1] >= 0.5 ? 1 : 0 }
    }
    
    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        let rawProbs = try await baseEstimator.predictProbability(features: features)
        return rawProbs.map { row in
            let rawP1 = row.count > 1 ? row[1] : row[0]
            let z = a * rawP1 + b
            let calP1 = 1.0 / (1.0 + exp(-max(-50.0, min(50.0, z))))
            return [1.0 - calP1, calP1]
        }
    }
}
