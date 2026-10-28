import Foundation
import SwiftPreprocessing
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

/// Interpreted feature with its learned weight and column index.
public struct FeatureImportance: Sendable, Codable, Equatable, Comparable {
    /// The feature name, token, or column identifier.
    public let feature: String
    /// The learned model weight or feature importance score.
    public let weight: Double
    /// The original column index.
    public let index: Int

    /// Creates a new FeatureImportance instance.
    public init(feature: String, weight: Double, index: Int) {
        self.feature = feature
        self.weight = weight
        self.index = index
    }

    public static func < (lhs: FeatureImportance, rhs: FeatureImportance) -> Bool {
        lhs.weight < rhs.weight
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
    /// Fits prepared columns. The default converts once for legacy classifiers.
    func fit(features: PreparedNumericBatch, targets: [Double]) async throws

    /// Predicts labels from prepared columns through native or compatibility dispatch.
    func predict(features: PreparedNumericBatch) async throws -> [Int]

    /// Predicts class probabilities from prepared columns.
    func predictProbability(features: PreparedNumericBatch) async throws -> [[Double]]

    /// Fits the classifier model on the provided features and targets.
    func fit(features: [[Double]], targets: [Double]) async throws

    /// Predicts class labels for the given feature matrix.
    func predict(features: [[Double]]) async throws -> [Int]

    /// Predicts class probabilities for the given feature matrix.
    func predictProbability(features: [[Double]]) async throws -> [[Double]]
}

/// Represents predicted class probability matrices with safe indexing and convenient ergonomics.
public struct ProbabilityMatrix: RandomAccessCollection, ExpressibleByArrayLiteral, CustomStringConvertible, Equatable, Sendable {
    /// Probability distribution row element type (`[Double]`).
    public typealias Element = [Double]
    /// Integer row index type.
    public typealias Index = Int

    /// Underlying 2D array of class probability rows of shape `[N, K]`.
    public let rows: [[Double]]

    /// Creates a probability matrix from a 2D array of probability rows.
    /// - Parameter rows: 2D array of predicted class probabilities.
    public init(_ rows: [[Double]]) {
        self.rows = rows
    }

    /// Creates a probability matrix from an array literal of probability rows.
    /// - Parameter elements: Variadic probability row vectors.
    public init(arrayLiteral elements: [Double]...) {
        self.rows = elements
    }

    /// The index of the first probability row (`0`).
    public var startIndex: Int { 0 }
    /// The collection's "past the end" row index (`rows.count`).
    public var endIndex: Int { rows.count }

    /// Accesses the probability vector at the specified row index.
    /// - Parameter position: Valid row index in `0..<endIndex`.
    public subscript(position: Int) -> [Double] {
        precondition(position >= 0 && position < rows.count, "ProbabilityMatrix row index out of range")
        return rows[position]
    }

    /// Textual representation of the probability matrix rows.
    public var description: String {
        rows.description
    }

    /// Returns the underlying 2D array of probability rows.
    public var asArray: [[Double]] { rows }
}

extension ClassifierEstimator {
    /// Compatibility implementation for classifiers accepting nested row arrays.
    public func fit(features: PreparedNumericBatch, targets: [Double]) async throws {
        try features.requireFinite()
        guard features.rowCount == targets.count, targets.allSatisfy(\.isFinite) else {
            throw SwiftMLError.invalidParameter("Classification targets must be finite and match the feature rows")
        }
        try await fit(features: features.rowValues(), targets: targets)
    }

    /// Compatibility implementation retaining existing label prediction behavior.
    public func predict(features: PreparedNumericBatch) async throws -> [Int] {
        try features.requireFinite()
        return try await predict(features: features.rowValues())
    }

    /// Compatibility implementation retaining unsupported-probability errors.
    public func predictProbability(features: PreparedNumericBatch) async throws -> [[Double]] {
        try features.requireFinite()
        return try await predictProbability(features: features.rowValues())
    }

    /// Fits features and targets selected together from one prepared snapshot.
    public func fit(_ data: PreparedSupervisedBatch) async throws {
        try await fit(features: data.features, targets: data.targets)
    }

    /// Predicts prepared class probabilities with the existing convenience wrapper.
    public func predictProba(features: PreparedNumericBatch) async throws -> ProbabilityMatrix {
        ProbabilityMatrix(try await predictProbability(features: features))
    }

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

extension ClassifierEstimator {
    /// Reserves a caller-supplied estimate covering inputs, workspace, copies and model state.
    public func fit(features: PreparedNumericBatch, targets: [Double],
                    budget: MemoryBudget, estimate: MemoryEstimate) async throws {
        try await budget.withReservation(estimate) {
            try await self.fit(features: features, targets: targets)
        }
    }

    /// Fits jointly selected features and targets under an explicit reservation.
    public func fit(_ data: PreparedSupervisedBatch, budget: MemoryBudget,
                    estimate: MemoryEstimate) async throws {
        try await fit(features: data.features, targets: data.targets, budget: budget, estimate: estimate)
    }

    /// Returned labels outlive this reservation and remain the caller's responsibility.
    public func predict(features: PreparedNumericBatch, budget: MemoryBudget,
                        estimate: MemoryEstimate) async throws -> [Int] {
        try await budget.withReservation(estimate) { try await self.predict(features: features) }
    }

    /// Returned probabilities outlive this reservation and remain the caller's responsibility.
    public func predictProbability(features: PreparedNumericBatch, budget: MemoryBudget,
                                   estimate: MemoryEstimate) async throws -> [[Double]] {
        try await budget.withReservation(estimate) { try await self.predictProbability(features: features) }
    }
}

/// Protocol representing a supervised regressor estimator.
public protocol RegressorEstimator: Sendable {
    /// Fits prepared columns. The default converts once for legacy estimators.
    func fit(features: PreparedNumericBatch, targets: [Double]) async throws

    /// Predicts from prepared columns. The default supports existing estimators.
    func predict(features: PreparedNumericBatch) async throws -> [Double]

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


extension RegressorEstimator {
    /// Compatibility implementation for estimators accepting nested row arrays.
    public func fit(features: PreparedNumericBatch, targets: [Double]) async throws {
        try features.requireFinite()
        guard features.rowCount == targets.count, targets.allSatisfy(\.isFinite) else {
            throw SwiftMLError.invalidParameter("Regression targets must be finite and match the feature rows")
        }
        try await fit(features: features.rowValues(), targets: targets)
    }

    /// Compatibility implementation for estimators accepting nested row arrays.
    public func predict(features: PreparedNumericBatch) async throws -> [Double] {
        try features.requireFinite()
        return try await predict(features: features.rowValues())
    }

    /// Fits features and targets selected together from one prepared snapshot.
    public func fit(_ data: PreparedSupervisedBatch) async throws {
        try await fit(features: data.features, targets: data.targets)
    }
}

extension RegressorEstimator {
    /// Fits prepared inputs after acquiring a caller-supplied whole-operation estimate.
    /// Include retained inputs, transformer/solver workspace, possible copies and model outputs.
    public func fit(features: PreparedNumericBatch, targets: [Double],
                    budget: MemoryBudget, estimate: MemoryEstimate) async throws {
        try await budget.withReservation(estimate) {
            try await self.fit(features: features, targets: targets)
        }
    }

    /// Predicts prepared inputs under an explicit shared reservation budget.
    /// Returned predictions persist beyond the reservation and need accounting by their owner.
    public func predict(features: PreparedNumericBatch, budget: MemoryBudget,
                        estimate: MemoryEstimate) async throws -> [Double] {
        try await budget.withReservation(estimate) { try await self.predict(features: features) }
    }

    /// Fits jointly selected features and targets under an explicit reservation budget.
    public func fit(_ data: PreparedSupervisedBatch, budget: MemoryBudget,
                    estimate: MemoryEstimate) async throws {
        try await fit(features: data.features, targets: data.targets, budget: budget, estimate: estimate)
    }
}
