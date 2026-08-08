import Foundation
@_exported import SwiftDataFrame

/// A common protocol for numerical preprocessing estimators that transform 2D Double datasets.
public protocol PreprocessingTransformer: Sendable {
    /// Fits the transformer to the dataset.
    mutating func fit(_ data: [[Double]]) throws
    
    /// Transforms the dataset based on fitted parameters.
    func transform(_ data: [[Double]]) throws -> [[Double]]
    
    /// Fits to data, then transforms it.
    mutating func fitTransform(_ data: [[Double]]) throws -> [[Double]]
}

extension PreprocessingTransformer {
    /// Default implementation of fitTransform.
    public mutating func fitTransform(_ data: [[Double]]) throws -> [[Double]] {
        try fit(data)
        return try transform(data)
    }
}
