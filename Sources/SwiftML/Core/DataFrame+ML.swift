import Foundation
import SwiftDataFrame

extension ClassifierEstimator {
    /// Fits the classifier on features and target extracted from a DataFrame.
    /// - Parameters:
    ///   - df: <#description#>
    ///   - features: <#description#>
    ///   - target: <#description#>
    /// - Throws: <#error description#>
    public func fit(_ df: DataFrame, features: [String], target: String) async throws {
        try await fit(features: try df.toFeatureMatrix(features), targets: try df.toTargetVector(target))
    }

    /// Predicts class labels for features extracted from a DataFrame.
    /// - Parameters:
    ///   - df: <#description#>
    ///   - features: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
    public func predict(_ df: DataFrame, features: [String]) async throws -> [Int] {
        try await predict(features: try df.toFeatureMatrix(features))
    }

    /// Predicts class probabilities for features extracted from a DataFrame.
    /// - Parameters:
    ///   - df: <#description#>
    ///   - features: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
    public func predictProbability(_ df: DataFrame, features: [String]) async throws -> [[Double]] {
        try await predictProbability(features: try df.toFeatureMatrix(features))
    }
}

extension RegressorEstimator {
    /// Fits the regressor on features and target extracted from a DataFrame.
    /// - Parameters:
    ///   - df: <#description#>
    ///   - features: <#description#>
    ///   - target: <#description#>
    /// - Throws: <#error description#>
    public func fit(_ df: DataFrame, features: [String], target: String) async throws {
        try await fit(features: try df.toFeatureMatrix(features), targets: try df.toTargetVector(target))
    }

    /// Predicts regression targets for features extracted from a DataFrame.
    /// - Parameters:
    ///   - df: <#description#>
    ///   - features: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
    public func predict(_ df: DataFrame, features: [String]) async throws -> [Double] {
        try await predict(features: try df.toFeatureMatrix(features))
    }
}
