import Foundation

/// Fits one independent regressor per target column (one-estimator-per-target strategy).
public actor MultiOutputRegressor {
    private let estimatorFactory: @Sendable () -> any RegressorEstimator
    private var estimators: [any RegressorEstimator] = []
    
    /// Creates a new MultiOutputRegressor instance.
    /// - Parameter estimatorFactory: A closure returning a fresh instance of a regressor.
    public init(estimatorFactory: @escaping @Sendable () -> any RegressorEstimator) {
        self.estimatorFactory = estimatorFactory
    }
    
    /// Fits independent regressors for each target column in parallel.
    /// - Parameters:
    ///   - features: Feature matrix `[N x M]`.
    ///   - targets: Multi-target matrix `[N x K]`.
    public func fit(features: [[Double]], targets: [[Double]]) async throws {
        guard !features.isEmpty, !targets.isEmpty, features.count == targets.count else {
            throw SwiftSciError.dataError("Features and multi-targets count mismatch or empty.")
        }
        
        let numTargets = targets[0].count
        guard numTargets > 0 else {
            throw SwiftSciError.dataError("Target vector length must be greater than 0.")
        }
        
        // Extract column-wise targets
        var columnTargets = [[Double]](repeating: [], count: numTargets)
        for targetRow in targets {
            guard targetRow.count == numTargets else {
                throw SwiftSciError.dataError("Dimension mismatch: expected \(numTargets) target columns, got \(targetRow.count)")
            }
            for k in 0..<numTargets {
                columnTargets[k].append(targetRow[k])
            }
        }
        
        // Instantiate factory models
        var models = (0..<numTargets).map { _ in estimatorFactory() }
        
        // Fit each model on its respective target column in parallel
        try await withThrowingTaskGroup(of: (Int, any RegressorEstimator).self) { group in
            for k in 0..<numTargets {
                let model = models[k]
                let yCol = columnTargets[k]
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
    
    /// Predicts multi-target values for the given feature matrix.
    /// - Parameter features: Feature matrix `[N x M]`.
    /// - Returns: Predicted target matrix `[N x K]`.
    public func predict(features: [[Double]]) async throws -> [[Double]] {
        guard !estimators.isEmpty else {
            throw SwiftSciError.predictionError("Model not fitted.")
        }
        
        let numTargets = estimators.count
        let numSamples = features.count
        
        var predictionsByColumn = [[Double]](repeating: [], count: numTargets)
        for k in 0..<numTargets {
            let pred = try await estimators[k].predict(features: features)
            predictionsByColumn[k] = pred
        }
        
        var result = [[Double]](repeating: [Double](repeating: 0.0, count: numTargets), count: numSamples)
        for i in 0..<numSamples {
            for k in 0..<numTargets {
                result[i][k] = predictionsByColumn[k][i]
            }
        }
        
        return result
    }
}
