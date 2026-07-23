import Foundation
@_exported import SwiftDataFrame

/// Standardized output structure for model predictions.
public struct PredictionResult: Sendable, Codable {
    public let values: [Double]
    public let probabilities: [[Double]]?
    public let labels: [String]?

    public init(values: [Double], probabilities: [[Double]]? = nil, labels: [String]? = nil) {
        self.values = values
        self.probabilities = probabilities
        self.labels = labels
    }
}

/// Standardized evaluation metrics report.
public struct EvaluationReport: Sendable, Codable {
    public let metrics: [String: Double]
    public let confusionMatrix: [[Int]]?

    public init(metrics: [String: Double], confusionMatrix: [[Int]]? = nil) {
        self.metrics = metrics
        self.confusionMatrix = confusionMatrix
    }
}

/// Feature schema description.
public struct FeatureSchema: Sendable, Codable {
    public let columns: [String]
    public let types: [String]
    public let targetColumn: String?

    public init(columns: [String], types: [String], targetColumn: String? = nil) {
        self.columns = columns
        self.types = types
        self.targetColumn = targetColumn
    }
}

/// Generic predictor protocol.
public protocol Predictor: Sendable {
    func predict(features: [[Double]]) async throws -> PredictionResult
}

/// Generic estimator protocol.
public protocol Estimator: Sendable {
    associatedtype ModelType: Predictor
    func fit(features: [[Double]], targets: [Double]) async throws -> ModelType
}

/// Generic data transformer protocol.
public protocol DataTransformer: Sendable {
    func fitTransform(data: DataFrame) async throws -> DataFrame
    func transform(data: DataFrame) async throws -> DataFrame
}

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
        throw SwiftSciError.predictionError("predictProbability is not supported by \(Self.self)")
    }
}

/// Protocol representing a supervised regressor estimator.
public protocol RegressorEstimator: Sendable {
    /// Fits the regressor model on the provided features and targets.
    func fit(features: [[Double]], targets: [Double]) async throws

    /// Predicts targets for the given feature matrix.
    func predict(features: [[Double]]) async throws -> [Double]
}
