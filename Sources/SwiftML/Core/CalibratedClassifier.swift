import Foundation

/// CalibratedClassifier wraps a base ClassifierEstimator and applies Platt Scaling (logistic sigmoid calibration)
/// to produce well-calibrated class probabilities.
public actor CalibratedClassifier: ClassifierEstimator {
    /// The base estimator.
    public let baseEstimator: any ClassifierEstimator
    private var a: Double = 1.0
    private var b: Double = 0.0
    private var classes: [Int] = []
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - baseEstimator: The base estimator.
    public init(baseEstimator: any ClassifierEstimator) {
        self.baseEstimator = baseEstimator
    }
    
    /// Fits the base classifier and learns Platt scaling calibration parameters over binary targets.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    ///   - targets: 1D array of binary target class labels of length `N`.
    /// - Throws: `SwiftMLError` if inputs are empty, dimensions mismatch, or targets are not binary.
    public func fit(features: [[Double]], targets: [Double]) async throws {
        guard !features.isEmpty, !targets.isEmpty else {
            throw SwiftMLError.emptyInput
        }
        guard features.count == targets.count else {
            throw SwiftMLError.dimensionMismatch(expected: features.count, got: targets.count)
        }
        let uniqueClasses = Array(Set(targets.map { Int($0) })).sorted()
        guard uniqueClasses.count == 2 else {
            throw SwiftMLError.invalidParameter("CalibratedClassifier requires binary targets (2 distinct classes), got \(uniqueClasses.count)")
        }

        try await baseEstimator.fit(features: features, targets: targets)
        
        let rawProbs = try await baseEstimator.predictProbability(features: features)
        guard !rawProbs.isEmpty else { return }

        self.classes = uniqueClasses
        let positiveClass = uniqueClasses[1]
        let binaryTargets = targets.map { Int($0) == positiveClass ? 1.0 : 0.0 }
        
        // Extract probability of the positive class
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
                let diff = calP - binaryTargets[i]
                gradA += diff * p1[i]
                gradB += diff
            }
            paramA -= lr * (gradA / Double(numSamples))
            paramB -= lr * (gradB / Double(numSamples))
        }
        
        self.a = paramA
        self.b = paramB
    }
    
    /// Predicts binary class labels from calibrated probabilities.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    /// - Throws: `SwiftMLError` if the classifier is not fitted or base prediction fails.
    /// - Returns: Predicted integer class labels matching the fitted binary class labels.
    public func predict(features: [[Double]]) async throws -> [Int] {
        guard classes.count == 2 else {
            throw SwiftMLError.modelNotFitted
        }
        let probs = try await predictProbability(features: features)
        return probs.map { $0[1] >= 0.5 ? classes[1] : classes[0] }
    }
    
    /// Predicts calibrated class probabilities `[P(classes[0]), P(classes[1])]`.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    /// - Throws: `SwiftMLError` if the classifier is not fitted or base prediction fails.
    /// - Returns: 2D array of calibrated binary class probabilities of shape `[N, 2]`.
    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        guard classes.count == 2 else {
            throw SwiftMLError.modelNotFitted
        }
        let rawProbs = try await baseEstimator.predictProbability(features: features)
        return rawProbs.map { row in
            let rawP1 = row.count > 1 ? row[1] : row[0]
            let z = a * rawP1 + b
            let calP1 = sigmoid(z)
            return [1.0 - calP1, calP1]
        }
    }
}
