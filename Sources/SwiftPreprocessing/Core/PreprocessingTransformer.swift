import Foundation
@_exported import SwiftDataFrame

/// A common protocol for numerical preprocessing estimators that transform 2D Double datasets.
public protocol PreprocessingTransformer: Sendable {
    /// Fits the transformer to the dataset.
    mutating func fit(_ data: [[Double]]) throws
    
    /// Transforms the dataset based on fitted parameters.
    func transform(_ data: [[Double]]) throws -> [[Double]]
    
    /// Fits prepared columns. Existing conformers use the row-array compatibility default.
    mutating func fit(_ data: PreparedNumericBatch) throws

    /// Transforms prepared columns while preserving row count and feature names.
    func transform(_ data: PreparedNumericBatch) throws -> PreparedNumericBatch

    /// Whether Pipeline can compose this transformer's native prepared methods.
    /// The default is false, preserving whole-pipeline row-array compatibility
    /// for legacy steps that may change intermediate feature counts.
    var supportsNativePreparedBatches: Bool { get }

    /// Transfers a prepared input to the transformer. Implementations may reuse
    /// uniquely owned storage; shared snapshots must retain their values.
    func transform(consuming data: consuming PreparedNumericBatch) throws -> PreparedNumericBatch

    /// Fits to data, then transforms it.
    mutating func fitTransform(_ data: [[Double]]) throws -> [[Double]]
}

extension PreprocessingTransformer {
    /// Default implementation of fitTransform.
    /// - Parameters:
    ///   - data: Raw input data array or matrix for transformation.
    /// - Throws: `PreprocessingError` or `SwiftMLError` if columns are missing, types are invalid, or arrays are empty.
    /// - Returns: 2D numerical matrix of shape `[N, P]`.
    public mutating func fitTransform(_ data: [[Double]]) throws -> [[Double]] {
        try fit(data)
        return try transform(data)
    }
}


extension PreprocessingTransformer {
    /// Compatibility path for transformers without a compact-column implementation.
    public mutating func fit(_ data: PreparedNumericBatch) throws { try fit(data.rowValues()) }

    /// Opt in only when fit and transform support prepared-column composition.
    public var supportsNativePreparedBatches: Bool { false }

    /// Existing conformers retain their prepared or row-array implementation.
    public func transform(consuming data: consuming PreparedNumericBatch) throws -> PreparedNumericBatch {
        try transform(data)
    }

    /// Compatibility path for shape-preserving legacy transformers.
    /// Feature-changing transformers must implement their own prepared overload.
    public func transform(_ data: PreparedNumericBatch) throws -> PreparedNumericBatch {
        try data.replacingRows(transform(data.rowValues()))
    }
}
