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
    /// - Parameters:
    ///   - categories: Discrete category levels or categorical column names.
    ///   - target: Target column name or output variable identifier.
    /// - Throws: `PreprocessingError` or `SwiftMLError` if columns are missing, types are invalid, or arrays are empty.
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
    
    /// Fits on 2D matrix (first column is feature, last column is target) (PreprocessingTransformer protocol).
    /// - Parameters:
    ///   - data: Raw input data matrix of at least 2 columns `[rows, cols >= 2]`.
    /// - Throws: `PreprocessingError.emptyInput`, `dimensionMismatch`, or `invalidInput` if columns < 2.
    public mutating func fit(_ data: [[Double]]) throws {
        guard !data.isEmpty, !data[0].isEmpty else { throw PreprocessingError.emptyInput }
        let numCols = data[0].count
        guard numCols >= 2 else {
            throw PreprocessingError.invalidInput("TargetEncoder requires at least 2 columns: feature and target")
        }
        for row in data where row.count != numCols {
            throw PreprocessingError.dimensionMismatch(expected: numCols, got: row.count)
        }
        let categories = data.map { String($0.first ?? 0.0) }
        let defaultTarget = data.map { $0.last ?? 0.0 }
        try fit(categories: categories, target: defaultTarget)
    }

    
    /// Transforms categorical string array into target-encoded numerical array.
    /// - Parameters:
    ///   - categories: Discrete category levels or categorical column names.
    /// - Throws: `PreprocessingError` or `SwiftMLError` if columns are missing, types are invalid, or arrays are empty.
    /// - Returns: Array of computed numeric values.
    public func transform(categories: [String]) throws -> [Double] {
        guard isFitted else { throw PreprocessingError.fittingRequired }
        return categories.map { targetMeans[$0] ?? globalMean }
    }
    
    /// Transforms 2D feature matrix into 2D single-column target-encoded matrix (PreprocessingTransformer protocol).
    /// - Parameters:
    ///   - data: Raw input data matrix `[rows, cols]`.
    /// - Throws: `PreprocessingError` if un-fitted or ragged.
    /// - Returns: 2D numerical matrix of shape `[N, 1]`.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard isFitted else { throw PreprocessingError.fittingRequired }
        guard !data.isEmpty, !data[0].isEmpty else { throw PreprocessingError.emptyInput }
        let numCols = data[0].count
        for row in data where row.count != numCols {
            throw PreprocessingError.dimensionMismatch(expected: numCols, got: row.count)
        }
        let categories = data.map { String($0.first ?? 0.0) }
        let encoded = try transform(categories: categories)
        return encoded.map { [$0] }
    }
    
    /// Fits and transforms categorical string array in one step with Out-Of-Fold (OOF) target leakage protection.
    ///
    /// When `cv > 1`, encodes training observations using out-of-fold category target statistics,
    /// preventing target leakage into training features (Scikit-Learn parity).
    /// Final statistics across the entire dataset are preserved for subsequent `transform(categories:)` calls on validation/test data.
    /// - Parameters:
    ///   - categories: Discrete category levels or categorical column names.
    ///   - target: Target column name or output variable identifier.
    ///   - cv: Number of cross-validation folds for out-of-fold encoding (defaults to 5; pass `<= 1` to disable OOF).
    /// - Throws: `PreprocessingError` or `SwiftMLError` if columns are missing, types are invalid, or arrays are empty.
    /// - Returns: Array of computed numeric values.
    public mutating func fitTransform(categories: [String], target: [Double], cv: Int = 5) throws -> [Double] {
        guard !categories.isEmpty else { throw PreprocessingError.emptyInput }
        guard categories.count == target.count else {
            throw PreprocessingError.dimensionMismatch(expected: categories.count, got: target.count)
        }

        // Fit full dataset statistics for subsequent validation/test transform calls
        try fit(categories: categories, target: target)

        guard cv > 1 && categories.count >= cv else {
            return try transform(categories: categories)
        }

        // Out-of-fold encoding for training data
        let n = categories.count
        var oofEncoded = [Double](repeating: 0.0, count: n)

        for fold in 0..<cv {
            var foldCounts = [String: Int]()
            var foldSums = [String: Double]()
            var foldTotalCount = 0
            var foldTotalSum = 0.0

            for i in 0..<n {
                if i % cv != fold {
                    let cat = categories[i]
                    let y = target[i]
                    foldCounts[cat, default: 0] += 1
                    foldSums[cat, default: 0.0] += y
                    foldTotalCount += 1
                    foldTotalSum += y
                }
            }

            let foldGlobalMean = foldTotalCount > 0 ? (foldTotalSum / Double(foldTotalCount)) : globalMean

            for i in 0..<n where i % cv == fold {
                let cat = categories[i]
                if let count = foldCounts[cat], count > 0 {
                    let catMean = foldSums[cat]! / Double(count)
                    let smoothed = (Double(count) * catMean + smoothing * foldGlobalMean) / (Double(count) + smoothing)
                    oofEncoded[i] = smoothed
                } else {
                    oofEncoded[i] = foldGlobalMean
                }
            }
        }

        return oofEncoded
    }

}
