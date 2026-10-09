import Foundation

/// Pipeline chains multiple PreprocessingTransformers sequentially.
///
/// ## Concurrency Safety & Data Leakage Prevention
/// `Pipeline` provides value semantics. Passing or copying a pipeline across concurrent tasks
/// (e.g. cross-validation folds in a `TaskGroup`) guarantees state isolation, preventing data races
/// and validation data leakage.
///
/// ## Thread Safety
/// Conforms to `Sendable` and `PreprocessingTransformer`. All step transformations operate on value-isolated state.
public struct Pipeline: PreprocessingTransformer, Sendable {
    /// Legacy steps keep the original whole-pipeline row-array conversion.
    public var supportsNativePreparedBatches: Bool { steps.allSatisfy { $0.supportsNativePreparedBatches } }

    /// The steps.
    public var steps: [any PreprocessingTransformer]
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - steps: The steps.
    public init(steps: [any PreprocessingTransformer]) {
        self.steps = steps
    }
    
    /// Fits all the steps in the pipeline sequentially.
    /// - Parameters:
    ///   - data: Raw input data array or matrix for transformation.
    /// - Throws: `PreprocessingError` or `SwiftMLError` if columns are missing, types are invalid, or arrays are empty.
    public mutating func fit(_ data: [[Double]]) throws {
        var current = data
        for i in 0..<steps.count {
            try steps[i].fit(current)
            current = try steps[i].transform(current)
        }
    }
    
    /// Transforms the data through all steps in the pipeline sequentially.
    /// - Parameters:
    ///   - data: Raw input data array or matrix for transformation.
    /// - Throws: `PreprocessingError` or `SwiftMLError` if columns are missing, types are invalid, or arrays are empty.
    /// - Returns: 2D numerical matrix of shape `[N, P]`.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        var current = data
        for step in steps {
            current = try step.transform(current)
        }
        return current
    }

    /// Fits compact steps when every step opts in to prepared composition.
    /// Otherwise, preserves whole-pipeline row-array conversion for legacy steps.
    /// Each native intermediate can be consumed by the following step.
    public mutating func fit(_ data: PreparedNumericBatch) throws {
        guard supportsNativePreparedBatches else {
            try fit(data.rowValues())
            return
        }
        var current = data
        for i in steps.indices {
            try steps[i].fit(current)
            current = try steps[i].transform(consuming: consume current)
        }
    }

    /// Transforms prepared columns while preserving the caller's input snapshot.
    public func transform(_ data: PreparedNumericBatch) throws -> PreparedNumericBatch {
        try transform(consuming: data)
    }

    /// Transfers each intermediate to the next step. Unique storage may be
    /// reused by consuming transformers; legacy transformers keep their defaults.
    public func transform(consuming data: consuming PreparedNumericBatch) throws -> PreparedNumericBatch {
        guard supportsNativePreparedBatches else {
            return try data.replacingRows(transform(data.rowValues()))
        }
        var current = consume data
        if steps.isEmpty {
            // The legacy row adapter produces numeric NaNs for missing inputs,
            // even when there are no steps. Keep that result without copying values.
            for c in current.columns.indices {
                current.columns[c].validity = nil
                current.columns[c].nullCount = 0
            }
        }
        for step in steps {
            current = try step.transform(consuming: consume current)
        }
        return current
    }

}

