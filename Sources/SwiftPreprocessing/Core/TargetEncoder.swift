import Foundation

/// Target encoder for categorical features using Empirical Bayes smoothing.
public struct TargetEncoder: PreprocessingTransformer, Sendable {
    /// The smoothing.
    public let smoothing: Double
    
    private var targetMeans: [String: Double] = [:]
    private var globalMean: Double = 0.0
    private var isFitted: Bool = false
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - smoothing: The smoothing.
    public init(smoothing: Double = 10.0) {
        self.smoothing = max(0.0, smoothing)
    }
    
    /// Fits the TargetEncoder on categorical series and target values.
    public mutating func fit(categories: [String], target: [Double]) throws {
        guard !categories.isEmpty else { throw PreprocessingError.emptyInput }
        guard categories.count == target.count else {
            throw PreprocessingError.dimensionMismatch(expected: categories.count, got: target.count)
        }
        
        self.globalMean = target.reduce(0.0, +) / Double(target.count)
        
        var counts = [String: Int]()
        var sums = [String: Double]()
        
        for i in 0..<categories.count {
            let cat = categories[i]
            counts[cat, default: 0] += 1
            sums[cat, default: 0.0] += target[i]
        }
        
        var means = [String: Double]()
        for (cat, count) in counts {
            let catMean = sums[cat]! / Double(count)
            let smoothed = (Double(count) * catMean + smoothing * globalMean) / (Double(count) + smoothing)
            means[cat] = smoothed
        }
        
        self.targetMeans = means
        self.isFitted = true
    }
    
    /// Fits on 2D string matrix (first column used) and 1D target (PreprocessingTransformer protocol).
    public mutating func fit(_ data: [[Double]]) throws {
        let categories = data.map { String($0.first ?? 0.0) }
        let defaultTarget = data.map { $0.last ?? 0.0 }
        try fit(categories: categories, target: defaultTarget)
    }

    
    /// Transforms categorical string array into target-encoded numerical array.
    public func transform(categories: [String]) throws -> [Double] {
        guard isFitted else { throw PreprocessingError.fittingRequired }
        return categories.map { targetMeans[$0] ?? globalMean }
    }
    
    /// Transforms 2D feature matrix into 2D single-column target-encoded matrix (PreprocessingTransformer protocol).
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        let categories = data.map { String($0.first ?? 0.0) }
        let encoded = try transform(categories: categories)
        return encoded.map { [$0] }
    }
    
    /// Fits and transforms categorical string array in one step.
    public mutating func fitTransform(categories: [String], target: [Double]) throws -> [Double] {
        try fit(categories: categories, target: target)
        return try transform(categories: categories)
    }

}
