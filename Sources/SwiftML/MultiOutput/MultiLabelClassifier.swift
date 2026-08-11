import Foundation

/// Fits one independent classifier per binary label column (one-estimator-per-label strategy).
public actor MultiLabelClassifier {
    private let estimatorFactory: @Sendable () -> any ClassifierEstimator
    private var estimators: [any ClassifierEstimator] = []
    
    /// Creates a new MultiLabelClassifier instance.
    /// - Parameter estimatorFactory: A closure returning a fresh instance of a binary classifier.
    public init(estimatorFactory: @escaping @Sendable () -> any ClassifierEstimator) {
        self.estimatorFactory = estimatorFactory
    }
    
    /// Fits independent classifiers for each label column in parallel.
    /// - Parameters:
    ///   - features: Feature matrix `[N x M]`.
    ///   - targets: Multi-label binary matrix `[N x K]` containing 0s and 1s.
    public func fit(features: [[Double]], targets: [[Int]]) async throws {
        guard !features.isEmpty, !targets.isEmpty, features.count == targets.count else {
            throw SwiftMLError.invalidInput("Features and multi-labels count mismatch or empty.")
        }
        
        let numLabels = targets[0].count
        guard numLabels > 0 else {
            throw SwiftMLError.invalidInput("Label vector length must be greater than 0.")
        }
        
        // Extract column-wise labels converted to Double for fit
        var columnLabels = [[Double]](repeating: [], count: numLabels)
        for targetRow in targets {
            guard targetRow.count == numLabels else {
                throw SwiftMLError.dimensionMismatch(expected: numLabels, got: targetRow.count)
            }
            for k in 0..<numLabels {
                columnLabels[k].append(Double(targetRow[k]))
            }
        }
        
        var models = (0..<numLabels).map { _ in estimatorFactory() }
        
        try await withThrowingTaskGroup(of: (Int, any ClassifierEstimator).self) { group in
            for k in 0..<numLabels {
                let model = models[k]
                let yCol = columnLabels[k]
                group.addTask {
                    try await model.fit(features: features, targets: yCol)
                    return (k, model)
                }
            }
            
            for try await (k, fittedModel) in group {
                models[k] = fittedModel
            }
        }
        
        self.estimators = models
    }
    
    /// Predicts multi-label binary values for the given feature matrix.
    /// - Parameter features: Feature matrix `[N x M]`.
    /// - Returns: Predicted binary label matrix `[N x K]`.
    public func predict(features: [[Double]]) async throws -> [[Int]] {
        guard !estimators.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        
        let numLabels = estimators.count
        let numSamples = features.count
        
        var predictionsByColumn = [[Int]](repeating: [], count: numLabels)
        for k in 0..<numLabels {
            let pred = try await estimators[k].predict(features: features)
            predictionsByColumn[k] = pred
        }
        
        var result = [[Int]](repeating: [Int](repeating: 0, count: numLabels), count: numSamples)
        for i in 0..<numSamples {
            for k in 0..<numLabels {
                result[i][k] = predictionsByColumn[k][i]
            }
        }
        
        return result
    }
}
