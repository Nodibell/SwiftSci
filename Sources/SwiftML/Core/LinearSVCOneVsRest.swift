import Foundation

/// Multi-class One-Vs-Rest wrapper specifically for LinearSVC.
public actor LinearSVCOneVsRest: Sendable {
    /// The number of target classes.
    public let numClasses: Int
    private var estimators: [LinearSVC] = []

    /// Creates a new LinearSVCOneVsRest instance.
    public init(numClasses: Int) {
        self.numClasses = numClasses
    }

    /// Fits one binary LinearSVC per class against all other classes.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    ///   - targets: 1D array of ground-truth target values of length `N`.
    ///   - C: Regularization inverse penalty parameter (larger values enforce smaller margins).
    ///   - learningRate: Step size scaling factor for gradient parameter updates.
    ///   - epochs: Total number of optimization training epochs.
    ///   - onProgress: Progress callback closure receiving normalized completion ratios.
    /// - Throws: `SwiftMLError` if feature-target dimensions mismatch, inputs are empty, or optimization fails.
    public func fit(
        features: [[Double]],
        targets: [Double],
        C: Double = 1.0,
        learningRate: Float = 0.1,
        epochs: Int = 500,
        onProgress: (@Sendable (Int, Int) -> Void)? = nil
    ) async throws {
        guard !features.isEmpty, features.count == targets.count else {
            throw SwiftMLError.emptyInput
        }
        
        var newEstimators: [LinearSVC] = []
        for c in 0..<numClasses {
            onProgress?(c, numClasses)
            let binaryTargets = targets.map { Int($0) == c ? 1.0 : 0.0 }
            let est = LinearSVC(C: C, device: .auto)
            try await est.fit(features: features, targets: binaryTargets, learningRate: learningRate, epochs: epochs)
            newEstimators.append(est)
        }
        onProgress?(numClasses, numClasses)
        self.estimators = newEstimators
    }

    /// Predicts class index for feature vectors using highest decision score (argmax w^T x + b).
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    /// - Throws: `SwiftMLError` if feature-target dimensions mismatch, inputs are empty, or optimization fails.
    /// - Returns: Array of predicted discrete class labels for input observations.
    public func predict(features: [[Double]]) async throws -> [Int] {
        guard !estimators.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        
        var classScores = [[Double]](repeating: [Double](repeating: 0.0, count: numClasses), count: features.count)
        
        for (c, est) in estimators.enumerated() {
            let scores = try await est.decisionFunction(features: features)
            for i in 0..<features.count {
                classScores[i][c] = scores[i]
            }
        }
        
        return classScores.map { row in
            row.argmax()
        }

    }
}
