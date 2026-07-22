import Foundation
@_exported import SwiftPreprocessing

/// A supervised classification pipeline that chains zero or more preprocessing transformers
/// with a final classification estimator.
public final class ClassificationPipeline: ClassifierEstimator, @unchecked Sendable {
    public let transformers: [any PreprocessingTransformer]
    public let estimator: any ClassifierEstimator

    public init(transformers: [any PreprocessingTransformer] = [], estimator: any ClassifierEstimator) {
        self.transformers = transformers
        self.estimator = estimator
    }

    public func fit(features: [[Double]], targets: [Double]) async throws {
        var current = features
        for transformer in transformers {
            try transformer.fit(current)
            current = try transformer.transform(current)
        }
        try await estimator.fit(features: current, targets: targets)
    }

    public func predict(features: [[Double]]) async throws -> [Int] {
        var current = features
        for transformer in transformers {
            current = try transformer.transform(current)
        }
        return try await estimator.predict(features: current)
    }

    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        var current = features
        for transformer in transformers {
            current = try transformer.transform(current)
        }
        return try await estimator.predictProbability(features: current)
    }
}

/// A supervised regression pipeline that chains zero or more preprocessing transformers
/// with a final regression estimator.
public final class RegressionPipeline: RegressorEstimator, @unchecked Sendable {
    public let transformers: [any PreprocessingTransformer]
    public let estimator: any RegressorEstimator

    public init(transformers: [any PreprocessingTransformer] = [], estimator: any RegressorEstimator) {
        self.transformers = transformers
        self.estimator = estimator
    }

    public func fit(features: [[Double]], targets: [Double]) async throws {
        var current = features
        for transformer in transformers {
            try transformer.fit(current)
            current = try transformer.transform(current)
        }
        try await estimator.fit(features: current, targets: targets)
    }

    public func predict(features: [[Double]]) async throws -> [Double] {
        var current = features
        for transformer in transformers {
            current = try transformer.transform(current)
        }
        return try await estimator.predict(features: current)
    }
}
