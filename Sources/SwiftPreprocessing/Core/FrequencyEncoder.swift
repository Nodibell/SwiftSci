import Foundation

/// Frequency encoder for categorical features using normalized frequency ratios.
public final class FrequencyEncoder: PreprocessingTransformer, @unchecked Sendable {
    private var frequencies: [String: Double] = [:]
    private var isFitted: Bool = false
    
    /// Creates a new instance.
    public init() {}
    
    /// Fits FrequencyEncoder on categorical string values.
    public func fit(categories: [String]) throws {
        guard !categories.isEmpty else { throw PreprocessingError.emptyInput }
        let total = Double(categories.count)
        
        var counts = [String: Int]()
        for cat in categories {
            counts[cat, default: 0] += 1
        }
        
        var freqs = [String: Double]()
        for (cat, count) in counts {
            freqs[cat] = Double(count) / total
        }
        
        self.frequencies = freqs
        self.isFitted = true
    }
    
    /// Fit.
    /// - Throws: An error if the operation fails.
    public func fit(_ data: [[Double]]) throws {
        let categories = data.map { String($0.first ?? 0.0) }
        try fit(categories: categories)
    }
    
    /// Transforms categories into normalized frequency values.
    public func transform(categories: [String]) throws -> [Double] {
        guard isFitted else { throw PreprocessingError.fittingRequired }
        return categories.map { frequencies[$0] ?? 0.0 }
    }
    
    /// Transform.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[[Double]]` result.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        let categories = data.map { String($0.first ?? 0.0) }
        let encoded = try transform(categories: categories)
        return encoded.map { [$0] }
    }
    
    /// Fit transform.
    /// - Parameters:
    ///   - categories: The categories.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[Double]` result.
    public func fitTransform(categories: [String]) throws -> [Double] {
        try fit(categories: categories)
        return try transform(categories: categories)
    }
}
