import Foundation
@_exported import SwiftPreprocessing

/// Chains preprocessing with a classification estimator. Fits exclude predictions and
/// other fits. Changing configuration after fitting requires another successful fit.
public final class ClassificationPipeline: ClassifierEstimator, @unchecked Sendable {
    private struct State {
        var transformers: [any PreprocessingTransformer]
        var fittedFeatureNames: [String]?
        var needsSuccessfulFit = false
        var fitting = false
        var activePredictionCount = 0
        var configurationGeneration: UInt64 = 0
        var hasStartedFit = false
    }

    private let lock = NSLock()
    private var state: State

    /// Transformer configuration. Replacing it after a fit starts requires refitting.
    public var transformers: [any PreprocessingTransformer] {
        get { lock.withLock { state.transformers } }
        set {
            lock.withLock {
                state.transformers = newValue
                state.configurationGeneration &+= 1
                if state.hasStartedFit { state.needsSuccessfulFit = true }
            }
        }
    }
    /// The estimator.
    public let estimator: any ClassifierEstimator

    /// Creates a classification pipeline.
    public init(transformers: [any PreprocessingTransformer] = [], estimator: any ClassifierEstimator) {
        self.state = State(transformers: transformers)
        self.estimator = estimator
    }

    /// Fits preprocessing only on the supplied training rows, then fits the classifier.
    public func fit(features: [[Double]], targets: [Double]) async throws {
        var configuration = try beginFit()
        defer { endFit() }
        var current = features
        for i in configuration.transformers.indices {
            try configuration.transformers[i].fit(current)
            current = try configuration.transformers[i].transform(current)
            guard current.count == features.count else {
                throw SwiftMLError.dimensionMismatch(expected: features.count, got: current.count)
            }
        }
        try await estimator.fit(features: current, targets: targets)
        try commitFit(transformers: configuration.transformers, featureNames: nil,
                      generation: configuration.generation)
    }

    /// Predicts labels after applying the fitted preprocessing steps.
    public func predict(features: [[Double]]) async throws -> [Int] {
        let configuration = try beginPrediction()
        defer { endPrediction() }
        var current = features
        for transformer in configuration.transformers {
            current = try transformer.transform(current)
            guard current.count == features.count else {
                throw SwiftMLError.dimensionMismatch(expected: features.count, got: current.count)
            }
        }
        let labels = try await estimator.predict(features: current)
        guard labels.count == features.count else {
            throw SwiftMLError.dimensionMismatch(expected: features.count, got: labels.count)
        }
        return labels
    }

    /// Predicts probabilities after applying the fitted preprocessing steps.
    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        let configuration = try beginPrediction()
        defer { endPrediction() }
        var current = features
        for transformer in configuration.transformers {
            current = try transformer.transform(current)
            guard current.count == features.count else {
                throw SwiftMLError.dimensionMismatch(expected: features.count, got: current.count)
            }
        }
        let probabilities = try await estimator.predictProbability(features: current)
        try Self.requireProbabilityShape(probabilities, rowCount: features.count)
        return probabilities
    }

    /// Fits prepared training columns through native transformer and estimator dispatch.
    public func fit(features: PreparedNumericBatch, targets: [Double]) async throws {
        try await fit(consuming: features, targets: targets)
    }

    /// Allows uniquely owned training columns to be reused by preprocessing.
    public func fit(consuming features: consuming PreparedNumericBatch, targets: [Double]) async throws {
        guard features.rowCount > 0, features.columnCount > 0 else { throw SwiftMLError.emptyInput }
        guard Set(features.columnNames).count == features.columnCount,
              features.rowCount == targets.count, targets.allSatisfy(\.isFinite) else {
            throw SwiftMLError.invalidParameter("Prepared classification requires unique names and aligned finite targets")
        }
        try features.requireFinite()
        let featureNames = features.columnNames
        let rowCount = features.rowCount
        let mapping = features.originalRowMapping
        var configuration = try beginFit()
        defer { endFit() }
        var current = consume features
        for i in configuration.transformers.indices {
            try configuration.transformers[i].fit(current)
            current = try configuration.transformers[i].transform(consuming: consume current)
            try Self.requireRows(current, count: rowCount, mapping: mapping)
        }
        try await estimator.fit(features: current, targets: targets)
        try commitFit(transformers: configuration.transformers, featureNames: featureNames,
                      generation: configuration.generation)
    }

    /// Predicts labels in the supplied prepared row order.
    public func predict(features: PreparedNumericBatch) async throws -> [Int] {
        try await predict(consuming: features)
    }

    /// Allows preprocessing to reuse uniquely owned prediction columns.
    public func predict(consuming features: consuming PreparedNumericBatch) async throws -> [Int] {
        let configuration = try beginPrediction()
        defer { endPrediction() }
        let rowCount = features.rowCount
        let current = try Self.transform(consuming: consume features,
                                         transformers: configuration.transformers,
                                         featureNames: configuration.featureNames)
        let labels = try await estimator.predict(features: current)
        guard labels.count == rowCount else {
            throw SwiftMLError.dimensionMismatch(expected: rowCount, got: labels.count)
        }
        return labels
    }

    /// Predicts class probabilities in the supplied prepared row order.
    public func predictProbability(features: PreparedNumericBatch) async throws -> [[Double]] {
        try await predictProbability(consuming: features)
    }

    /// Allows preprocessing to reuse uniquely owned probability-input columns.
    public func predictProbability(consuming features: consuming PreparedNumericBatch) async throws -> [[Double]] {
        let configuration = try beginPrediction()
        defer { endPrediction() }
        let rowCount = features.rowCount
        let current = try Self.transform(consuming: consume features,
                                         transformers: configuration.transformers,
                                         featureNames: configuration.featureNames)
        let probabilities = try await estimator.predictProbability(features: current)
        try Self.requireProbabilityShape(probabilities, rowCount: rowCount)
        return probabilities
    }

    private static func transform(consuming features: consuming PreparedNumericBatch,
                                  transformers: [any PreprocessingTransformer],
                                  featureNames: [String]?) throws -> PreparedNumericBatch {
        if let featureNames, featureNames != features.columnNames {
            throw SwiftMLError.invalidParameter("Prediction feature names or order differ from training")
        }
        try features.requireFinite()
        let rowCount = features.rowCount
        let mapping = features.originalRowMapping
        var current = consume features
        for transformer in transformers {
            current = try transformer.transform(consuming: consume current)
            try Self.requireRows(current, count: rowCount, mapping: mapping)
        }
        return current
    }

    private func beginFit() throws -> (transformers: [any PreprocessingTransformer], generation: UInt64) {
        try lock.withLock {
            guard !state.fitting, state.activePredictionCount == 0 else {
                throw SwiftMLError.invalidParameter("Pipeline fitting requires no active fit or predictions")
            }
            state.fitting = true
            state.hasStartedFit = true
            state.needsSuccessfulFit = true
            return (state.transformers, state.configurationGeneration)
        }
    }

    private func commitFit(transformers: [any PreprocessingTransformer], featureNames: [String]?,
                           generation: UInt64) throws {
        try lock.withLock {
            guard state.configurationGeneration == generation else {
                throw SwiftMLError.invalidParameter("Pipeline configuration changed during fitting; fit again")
            }
            state.transformers = transformers
            state.fittedFeatureNames = featureNames
            state.needsSuccessfulFit = false
        }
    }

    private func endFit() {
        lock.withLock { state.fitting = false }
    }

    private func beginPrediction() throws -> (transformers: [any PreprocessingTransformer], featureNames: [String]?) {
        try lock.withLock {
            guard !state.fitting, !state.needsSuccessfulFit else { throw SwiftMLError.modelNotFitted }
            state.activePredictionCount += 1
            return (state.transformers, state.fittedFeatureNames)
        }
    }

    private func endPrediction() {
        lock.withLock { state.activePredictionCount -= 1 }
    }

    private static func requireRows(_ batch: PreparedNumericBatch, count: Int, mapping: [Int]?) throws {
        guard batch.rowCount == count else {
            throw SwiftMLError.dimensionMismatch(expected: count, got: batch.rowCount)
        }
        for row in 0..<count where batch.originalRowIndex(at: row) != (mapping?[row] ?? row) {
            throw SwiftMLError.invalidParameter("Prepared transformers must preserve row order and identity")
        }
    }

    private static func requireProbabilityShape(_ probabilities: [[Double]], rowCount: Int) throws {
        guard probabilities.count == rowCount else {
            throw SwiftMLError.dimensionMismatch(expected: rowCount, got: probabilities.count)
        }
        guard let width = probabilities.first?.count else { return }
        guard width > 0, probabilities.allSatisfy({ $0.count == width }) else {
            throw SwiftMLError.invalidParameter("Classifier probabilities require a consistent nonempty class width")
        }
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
        try await fit(consuming: features, targets: targets)
    }

    /// Fits prepared features while allowing transformers to reuse uniquely
    /// owned input columns. Shared inputs retain their snapshot values.
    public func fit(consuming features: consuming PreparedNumericBatch, targets: [Double]) async throws {
        guard features.rowCount > 0, features.columnCount > 0 else { throw SwiftMLError.emptyInput }
        guard Set(features.columnNames).count == features.columnCount,
              features.rowCount == targets.count, targets.allSatisfy(\.isFinite) else {
            throw SwiftMLError.invalidParameter("Prepared regression requires unique names and aligned finite targets")
        }
        try features.requireFinite()
        needsSuccessfulFit = true
        let featureNames = features.columnNames
        var current = consume features
        for i in transformers.indices {
            try transformers[i].fit(current)
            current = try transformers[i].transform(consuming: consume current)
        }
        try await estimator.fit(features: current, targets: targets)
        fittedFeatureNames = featureNames
        needsSuccessfulFit = false
    }

    /// Predicts prepared features in the same named order used during prepared fitting.
    /// A pipeline fitted through unnamed row arrays accepts the caller's column order.
    public func predict(features: PreparedNumericBatch) async throws -> [Double] {
        try await predict(consuming: features)
    }

    /// Predicts from prepared input whose unique storage can be reused during
    /// preprocessing. Fitting and configuration changes must remain serialized.
    public func predict(consuming features: consuming PreparedNumericBatch) async throws -> [Double] {
        guard !needsSuccessfulFit else { throw SwiftMLError.modelNotFitted }
        if let fittedFeatureNames, fittedFeatureNames != features.columnNames {
            throw SwiftMLError.invalidParameter("Prediction feature names or order differ from training")
        }
        try features.requireFinite()
        let rowCount = features.rowCount
        var current = consume features
        for transformer in transformers { current = try transformer.transform(consuming: consume current) }
        let prediction = try await estimator.predict(features: current)
        guard prediction.count == rowCount else {
            throw SwiftMLError.dimensionMismatch(expected: rowCount, got: prediction.count)
        }
        return prediction
    }

}
