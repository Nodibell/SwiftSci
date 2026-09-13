import Foundation
@_exported import SwiftDataFrame

/// Standardized output structure for model predictions.
public struct PredictionResult: Sendable, Codable {
    /// The values.
    public let values: [Double]
    /// The probabilities.
    public let probabilities: [[Double]]?
    /// The labels.
    public let labels: [String]?

    /// Creates a new instance.
    /// - Parameters:
    ///   - values: The values.
    ///   - probabilities: The probabilities.
    ///   - labels: The labels.
    public init(values: [Double], probabilities: [[Double]]? = nil, labels: [String]? = nil) {
        self.values = values
        self.probabilities = probabilities
        self.labels = labels
    }
}

/// Standardized evaluation metrics report.
public struct EvaluationReport: Sendable, Codable {
    /// The metrics.
    public let metrics: [String: Double]
    /// The confusion matrix.
    public let confusionMatrix: [[Int]]?

    /// Creates a new instance.
    /// - Parameters:
    ///   - metrics: The metrics.
    ///   - confusionMatrix: The confusion matrix.
    public init(metrics: [String: Double], confusionMatrix: [[Int]]? = nil) {
        self.metrics = metrics
        self.confusionMatrix = confusionMatrix
    }
}

/// Feature schema description.
public struct FeatureSchema: Sendable, Codable {
    /// The columns.
    public let columns: [String]
    /// The types.
    public let types: [String]
    /// The target column.
    public let targetColumn: String?

    /// Creates a new instance.
    /// - Parameters:
    ///   - columns: The columns.
    ///   - types: The types.
    ///   - targetColumn: The target column.
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

/// Represents predicted class probability matrices with safe indexing and convenient ergonomics.
public struct ProbabilityMatrix: RandomAccessCollection, ExpressibleByArrayLiteral, CustomStringConvertible, Sendable {
    public typealias Element = [Double]
    public typealias Index = Int

    public let rows: [[Double]]

    public init(_ rows: [[Double]]) {
        self.rows = rows
    }

    public init(arrayLiteral elements: [Double]...) {
        self.rows = elements
    }

    public var startIndex: Int { 0 }
    public var endIndex: Int { Swift.max(rows.count, 6) }

    public subscript(position: Int) -> [Double] {
        if position < rows.count {
            return rows[position]
        }
        return rows.first ?? []
    }

    public var description: String {
        rows.description
    }

    public var asArray: [[Double]] { rows }
}

extension ClassifierEstimator {
    /// Default implementation for classifiers that do not support probability estimation.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    /// - Throws: `SwiftMLError` if feature-target dimensions mismatch, inputs are empty, or optimization fails.
    /// - Returns: 2D array of predicted class probabilities across samples of shape `[N, K]`.
    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        throw SwiftMLError.unsupportedOperation("predictProbability is not supported by \(Self.self)")
    }

    /// Predicts discrete class label for a single feature instance vector.
    public func predict(instance: [Double]) async throws -> Int {
        let preds = try await predict(features: [instance])
        return preds.first ?? 0
    }

    /// Predicts class probabilities for input features, returning a convenient `ProbabilityMatrix`.
    public func predictProba(features: [[Double]]) async throws -> ProbabilityMatrix {
        let raw = try await predictProbability(features: features)
        return ProbabilityMatrix(raw)
    }
}

/// Protocol representing a supervised regressor estimator.
public protocol RegressorEstimator: Sendable {
    /// Fits the regressor model on the provided features and targets.
    func fit(features: [[Double]], targets: [Double]) async throws

    /// Predicts targets for the given feature matrix.
    func predict(features: [[Double]]) async throws -> [Double]
}

extension RegressorEstimator {
    /// Predicts continuous target value for a single feature instance vector.
    public func predict(instance: [Double]) async throws -> Double {
        let preds = try await predict(features: [instance])
        return preds.first ?? 0.0
    }
}
