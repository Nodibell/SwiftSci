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
        self.smoothing = smoothing
    }
    
    /// Fits the TargetEncoder on categorical series and target values.
    /// - Parameters:
    ///   - categories: Discrete category levels or categorical column names.
    ///   - target: Target column name or output variable identifier.
    /// - Throws: `PreprocessingError` or `SwiftMLError` if columns are missing, types are invalid, or arrays are empty.
    public mutating func fit(categories: [String], target: [Double]) throws {
        guard smoothing.isFinite, smoothing >= 0 else {
            throw PreprocessingError.invalidInput("TargetEncoder smoothing must be finite and non-negative")
        }
        guard !categories.isEmpty else { throw PreprocessingError.emptyInput }
        guard categories.count == target.count else {
            throw PreprocessingError.dimensionMismatch(expected: categories.count, got: target.count)
        }
        for value in target {
            if value.isNaN { throw PreprocessingError.containsNaN }
            if !value.isFinite { throw PreprocessingError.containsInfinity }
        }
        
        let computedGlobalMean = Self.centeredMean(target)
        
        var counts = [String: Int]()
        var origins = [String: Double]()
        var centeredSums = [String: Double]()
        
        for i in 0..<categories.count {
            let cat = categories[i]
            let y = target[i]
            counts[cat, default: 0] += 1
            if let origin = origins[cat] {
                centeredSums[cat, default: 0.0] += (y - origin)
            } else {
                origins[cat] = y
                centeredSums[cat] = 0.0
            }
        }
        
        var means = [String: Double]()
        for (cat, count) in counts {
            let catMean = origins[cat]! + centeredSums[cat]! / Double(count)
            let smoothed = (Double(count) * catMean + smoothing * computedGlobalMean) / (Double(count) + smoothing)
            means[cat] = smoothed
        }
        
        self.globalMean = computedGlobalMean
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
        let categories = data.map { Self.formatNumericCategory($0[0]) }
        let defaultTarget = data.map { $0[numCols - 1] }
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
        let categories = data.map { Self.formatNumericCategory($0[0]) }
        let encoded = try transform(categories: categories)
        return encoded.map { [$0] }
    }

    /// Fits training rows with out-of-fold encoding; the final column supplies the target.
    public mutating func fitTransform(_ data: [[Double]]) throws -> [[Double]] {
        guard let first = data.first, !first.isEmpty else { throw PreprocessingError.emptyInput }
        guard first.count >= 2 else {
            throw PreprocessingError.invalidInput("TargetEncoder requires feature and target columns")
        }
        for row in data where row.count != first.count {
            throw PreprocessingError.dimensionMismatch(expected: first.count, got: row.count)
        }
        return try fitTransform(
            categories: data.map { Self.formatNumericCategory($0[0]) },
            target: data.map { $0[first.count - 1] }
        ).map { [$0] }
    }
    
    /// Fits and transforms categorical string array in one step with Out-Of-Fold (OOF) target leakage protection.
    ///
    /// When `cv > 1`, encodes training observations using out-of-fold category target statistics,
    /// preventing target leakage into training features (Scikit-Learn parity).
    /// Final statistics across the entire dataset are preserved for subsequent `transform(categories:)` calls on validation/test data.
    /// - Parameters:
    ///   - categories: Discrete category levels or categorical column names.
    ///   - target: Target column name or output variable identifier.
    ///   - cv: Number of cross-validation folds for out-of-fold encoding (defaults to 5, capped at sample count; pass `<= 1` to disable OOF).
    /// - Throws: `PreprocessingError` or `SwiftMLError` if columns are missing, types are invalid, or arrays are empty.
    /// - Returns: Array of computed numeric values.
    public mutating func fitTransform(categories: [String], target: [Double], cv: Int = 5) throws -> [Double] {
        guard smoothing.isFinite, smoothing >= 0 else {
            throw PreprocessingError.invalidInput("TargetEncoder smoothing must be finite and non-negative")
        }
        guard !categories.isEmpty else { throw PreprocessingError.emptyInput }
        guard categories.count == target.count else {
            throw PreprocessingError.dimensionMismatch(expected: categories.count, got: target.count)
        }
        guard cv <= 1 || categories.count > 1 else {
            throw PreprocessingError.invalidInput("Out-of-fold encoding requires at least two observations")
        }
        let folds = min(cv, categories.count)

        // Fit full dataset statistics for subsequent validation/test transform calls
        try fit(categories: categories, target: target)

        guard cv > 1 else {
            return try transform(categories: categories)
        }

        // Out-of-fold encoding for training data
        let n = categories.count
        var oofEncoded = [Double](repeating: 0.0, count: n)

        for fold in 0..<folds {
            var foldCounts = [String: Int]()
            var foldOrigins = [String: Double]()
            var foldCenteredSums = [String: Double]()
            var foldTotalCount = 0
            let foldTotalOrigin = target.first ?? 0.0
            var foldTotalCenteredSum = 0.0

            for i in 0..<n {
                if i % folds != fold {
                    let cat = categories[i]
                    let y = target[i]
                    foldCounts[cat, default: 0] += 1
                    if let origin = foldOrigins[cat] {
                        foldCenteredSums[cat, default: 0.0] += (y - origin)
                    } else {
                        foldOrigins[cat] = y
                        foldCenteredSums[cat] = 0.0
                    }
                    foldTotalCount += 1
                    foldTotalCenteredSum += (y - foldTotalOrigin)
                }
            }

            let foldGlobalMean = foldTotalCount > 0 ? (foldTotalOrigin + foldTotalCenteredSum / Double(foldTotalCount)) : globalMean

            for i in 0..<n where i % folds == fold {
                let cat = categories[i]
                if let count = foldCounts[cat], count > 0 {
                    let catMean = foldOrigins[cat]! + foldCenteredSums[cat]! / Double(count)
                    let smoothed = (Double(count) * catMean + smoothing * foldGlobalMean) / (Double(count) + smoothing)
                    oofEncoded[i] = smoothed
                } else {
                    oofEncoded[i] = foldGlobalMean
                }
            }
        }

        return oofEncoded
    }

    private static func formatNumericCategory(_ value: Double) -> String {
        if value.isNaN { return "nan" }
        return String(value == 0.0 ? 0.0 : value)
    }

    private static func centeredMean(_ values: [Double]) -> Double {
        let origin = values[0]
        var deltaSum = 0.0
        for value in values { deltaSum += (value - origin) }
        return origin + deltaSum / Double(values.count)
    }

}
