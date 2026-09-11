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
}

