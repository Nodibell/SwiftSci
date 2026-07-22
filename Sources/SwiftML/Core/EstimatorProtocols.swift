import Foundation
@_exported import SwiftDataFrame

/// Protocol representing a supervised classifier estimator.
public protocol ClassifierEstimator: Sendable {
    /// Fits the classifier model on the provided features and targets.
    func fit(features: [[Double]], targets: [Double]) async throws

    /// Predicts class labels for the given feature matrix.
    func predict(features: [[Double]]) async throws -> [Int]

    /// Predicts class probabilities for the given feature matrix.
    func predictProbability(features: [[Double]]) async throws -> [[Double]]
}

extension ClassifierEstimator {
    /// Default implementation for classifiers that do not support probability estimation.
    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        throw MLError.invalidInput("predictProbability is not supported by \(Self.self)")
    }
}

/// Protocol representing a supervised regressor estimator.
public protocol RegressorEstimator: Sendable {
    /// Fits the regressor model on the provided features and targets.
    func fit(features: [[Double]], targets: [Double]) async throws

    /// Predicts targets for the given feature matrix.
    func predict(features: [[Double]]) async throws -> [Double]
}
