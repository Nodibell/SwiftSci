import Foundation
import SwiftDataFrame

extension ClassifierEstimator {
    /// Fits the classifier on features and target extracted from a DataFrame.
    /// - Parameters:
    ///   - df: Input DataFrame instance.
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    ///   - target: Target column name or output variable identifier.
    /// - Throws: `SwiftMLError` if feature-target dimensions mismatch, inputs are empty, or optimization fails.
    public func fit(_ df: DataFrame, features: [String], target: String) async throws {
        try await fit(features: try df.toFeatureMatrix(features), targets: try df.toTargetVector(target))
    }

    /// Predicts class labels for features extracted from a DataFrame.
    /// - Parameters:
    ///   - df: Input DataFrame instance.
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    /// - Throws: `SwiftMLError` if feature-target dimensions mismatch, inputs are empty, or optimization fails.
    /// - Returns: Array of predicted discrete class labels for input observations.
    public func predict(_ df: DataFrame, features: [String]) async throws -> [Int] {
        try await predict(features: try df.toFeatureMatrix(features))
    }

    /// Predicts class probabilities for features extracted from a DataFrame.
    /// - Parameters:
    ///   - df: Input DataFrame instance.
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    /// - Throws: `SwiftMLError` if feature-target dimensions mismatch, inputs are empty, or optimization fails.
    /// - Returns: 2D array of predicted class probabilities across samples of shape `[N, K]`.
    public func predictProbability(_ df: DataFrame, features: [String]) async throws -> [[Double]] {
        try await predictProbability(features: try df.toFeatureMatrix(features))
    }
}

extension RegressorEstimator {
    /// Fits the regressor on features and target extracted from a DataFrame.
    /// - Parameters:
    ///   - df: Input DataFrame instance.
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    ///   - target: Target column name or output variable identifier.
    /// - Throws: `SwiftMLError` if feature-target dimensions mismatch, inputs are empty, or optimization fails.
    public func fit(_ df: DataFrame, features: [String], target: String) async throws {
        try await fit(features: try df.toFeatureMatrix(features), targets: try df.toTargetVector(target))
    }

    /// Predicts regression targets for features extracted from a DataFrame.
    /// - Parameters:
    ///   - df: Input DataFrame instance.
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    /// - Throws: `SwiftMLError` if feature-target dimensions mismatch, inputs are empty, or optimization fails.
    /// - Returns: Array of predicted continuous targets for input observations.
    public func predict(_ df: DataFrame, features: [String]) async throws -> [Double] {
        try await predict(features: try df.toFeatureMatrix(features))
    }
}
