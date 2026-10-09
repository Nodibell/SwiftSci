import Foundation
@_exported import SwiftPreprocessing

/// A supervised classification pipeline that chains zero or more preprocessing transformers
/// with a final classification estimator.
public final class ClassificationPipeline: ClassifierEstimator, @unchecked Sendable {
    /// The transformers.
    public var transformers: [any PreprocessingTransformer]
    /// The estimator.
    public let estimator: any ClassifierEstimator

    /// Creates a new instance.
    /// - Parameters:
    ///   - transformers: The transformers.
    ///   - estimator: The estimator.
    public init(transformers: [any PreprocessingTransformer] = [], estimator: any ClassifierEstimator) {
        self.transformers = transformers
        self.estimator = estimator
    }

    /// Fit.
    /// - Parameters:
    ///   - features: The features.
    ///   - targets: The targets.
    /// - Throws: An error if the operation fails.
    public func fit(features: [[Double]], targets: [Double]) async throws {
        var current = features
        for i in 0..<transformers.count {
            try transformers[i].fit(current)
            current = try transformers[i].transform(current)
        }
        try await estimator.fit(features: current, targets: targets)
    }

    /// Predict.
    /// - Parameters:
    ///   - features: The features.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[Int]` result.
    public func predict(features: [[Double]]) async throws -> [Int] {
        var current = features
        for transformer in transformers {
            current = try transformer.transform(current)
        }
        return try await estimator.predict(features: current)
    }

    /// Predict probability.
    /// - Parameters:
    ///   - features: The features.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[[Double]]` result.
    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        var current = features
        for transformer in transformers {
            current = try transformer.transform(current)
        }
        return try await estimator.predictProbability(features: current)
    }
}

/// Configure transformers before fitting. Serialize fit calls and configuration changes;
/// prediction during fitting is unsupported. Prepared input is opt-in.
/// A supervised regression pipeline that chains zero or more preprocessing transformers
/// with a final regression estimator.
public final class RegressionPipeline: RegressorEstimator, @unchecked Sendable {
    private var fittedFeatureNames: [String]?
    private var needsSuccessfulFit = false

    /// The transformers.
    public var transformers: [any PreprocessingTransformer]
    /// The estimator.
    public let estimator: any RegressorEstimator

    /// Creates a new instance.
    /// - Parameters:
    ///   - transformers: The transformers.
    ///   - estimator: The estimator.
    public init(transformers: [any PreprocessingTransformer] = [], estimator: any RegressorEstimator) {
        self.transformers = transformers
        self.estimator = estimator
    }

    /// Fit.
    /// - Parameters:
    ///   - features: The features.
    ///   - targets: The targets.
    /// - Throws: An error if the operation fails.
    public func fit(features: [[Double]], targets: [Double]) async throws {
        needsSuccessfulFit = true
        var current = features
        for i in 0..<transformers.count {
            try transformers[i].fit(current)
            current = try transformers[i].transform(current)
        }
        try await estimator.fit(features: current, targets: targets)
        fittedFeatureNames = nil
        needsSuccessfulFit = false
    }

    /// Predict.
    /// - Parameters:
    ///   - features: The features.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[Double]` result.
    public func predict(features: [[Double]]) async throws -> [Double] {
        guard !needsSuccessfulFit else { throw SwiftMLError.modelNotFitted }
        var current = features
        for transformer in transformers {
            current = try transformer.transform(current)
        }
        return try await estimator.predict(features: current)
    }

    /// Fits prepared features through protocol dispatch. StandardScaler and LinearRegression
    /// keep compact columns; other conformers may use their compatibility defaults.
    public func fit(features: PreparedNumericBatch, targets: [Double]) async throws {
        guard features.rowCount > 0, features.columnCount > 0 else { throw SwiftMLError.emptyInput }
        guard Set(features.columnNames).count == features.columnCount,
              features.rowCount == targets.count, targets.allSatisfy(\.isFinite) else {
            throw SwiftMLError.invalidParameter("Prepared regression requires unique names and aligned finite targets")
        }
        try features.requireFinite()
        needsSuccessfulFit = true
        var current = features
        for i in transformers.indices {
            try transformers[i].fit(current)
            current = try transformers[i].transform(current)
        }
        try await estimator.fit(features: current, targets: targets)
        fittedFeatureNames = features.columnNames
        needsSuccessfulFit = false
    }

    /// Predicts prepared features in the same named order used during prepared fitting.
    /// A pipeline fitted through unnamed row arrays accepts the caller's column order.
    public func predict(features: PreparedNumericBatch) async throws -> [Double] {
        guard !needsSuccessfulFit else { throw SwiftMLError.modelNotFitted }
        if let fittedFeatureNames, fittedFeatureNames != features.columnNames {
            throw SwiftMLError.invalidParameter("Prediction feature names or order differ from training")
        }
        try features.requireFinite()
        var current = features
        for transformer in transformers { current = try transformer.transform(current) }
        let prediction = try await estimator.predict(features: current)
        guard prediction.count == features.rowCount else {
            throw SwiftMLError.dimensionMismatch(expected: features.rowCount, got: prediction.count)
        }
        return prediction
    }

}
