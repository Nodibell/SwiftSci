import Foundation

/// Standardize features by removing the mean and scaling to unit variance.
public final class StandardScaler: @unchecked Sendable {
    public private(set) var mean: [Double]?
    public private(set) var std: [Double]?
    
    public init() {}
    
    /// Fits the scaler to the 2D input dataset.
    /// - Parameter data: A 2D array of features of shape [rows, cols].
    public func fit(_ data: [[Double]]) throws {
        guard !data.isEmpty, !data[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
        
        let rowCount = data.count
        let colCount = data[0].count
        
        var sum = [Double](repeating: 0.0, count: colCount)
        for row in data {
            guard row.count == colCount else {
                throw PreprocessingError.dimensionMismatch(expected: colCount, got: row.count)
            }
            for col in 0..<colCount {
                sum[col] += row[col]
            }
        }
        
        let calculatedMean = sum.map { $0 / Double(rowCount) }
        
        var variance = [Double](repeating: 0.0, count: colCount)
        for row in data {
            for col in 0..<colCount {
                let diff = row[col] - calculatedMean[col]
                variance[col] += diff * diff
            }
        }
        
        let calculatedStd = variance.map { val -> Double in
            let varianceVal = val / Double(rowCount)
            let stdVal = varianceVal.squareRoot()
            // If standard deviation is extremely small or zero, treat it as 1.0 to avoid division by zero.
            return stdVal < 1e-12 ? 1.0 : stdVal
        }
        
        self.mean = calculatedMean
        self.std = calculatedStd
    }
    
    /// Transforms the dataset using the fitted mean and standard deviation.
    /// - Parameter data: A 2D array of features to scale.
    /// - Returns: The standardized 2D array.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard let mean = self.mean, let std = self.std else {
            throw PreprocessingError.fitNotCalled
        }
        guard !data.isEmpty else {
            return []
        }
        
        let colCount = mean.count
        var transformed = [[Double]]()
        transformed.reserveCapacity(data.count)
        
        for row in data {
            guard row.count == colCount else {
                throw PreprocessingError.dimensionMismatch(expected: colCount, got: row.count)
            }
            let scaledRow = (0..<colCount).map { col -> Double in
                (row[col] - mean[col]) / std[col]
            }
            transformed.append(scaledRow)
        }
        
        return transformed
    }
    
    /// Fits to data, then transforms it.
    public func fitTransform(_ data: [[Double]]) throws -> [[Double]] {
        try fit(data)
        return try transform(data)
    }
}
