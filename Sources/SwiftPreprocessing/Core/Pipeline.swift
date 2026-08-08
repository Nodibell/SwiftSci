import Foundation

/// Pipeline chains multiple PreprocessingTransformers sequentially.
public final class Pipeline: PreprocessingTransformer, @unchecked Sendable {
    /// The steps.
    public var steps: [any PreprocessingTransformer]
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - steps: The steps.
    public init(steps: [any PreprocessingTransformer]) {
        self.steps = steps
    }
    
    /// Fits all the steps in the pipeline sequentially.
    public func fit(_ data: [[Double]]) throws {
        var current = data
        for i in 0..<steps.count {
            try steps[i].fit(current)
            current = try steps[i].transform(current)
        }
    }
    
    /// Transforms the data through all steps in the pipeline sequentially.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        var current = data
        for step in steps {
            current = try step.transform(current)
        }
        return current
    }
}
