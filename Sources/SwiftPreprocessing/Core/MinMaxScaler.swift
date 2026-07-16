import Foundation

/// MinMaxScaler scales features to a specified range (default [0, 1]).
public final class MinMaxScaler: @unchecked Sendable {
    public private(set) var dataMin: [Double]?
    public private(set) var dataMax: [Double]?
    public let range: (min: Double, max: Double)
    
    public init(range: (Double, Double) = (0.0, 1.0)) {
        self.range = range
    }
    
    /// Fits the scaler to the 2D input dataset.
    public func fit(_ data: [[Double]]) throws {
        guard !data.isEmpty, !data[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
        
        let colCount = data[0].count
        var minVals = data[0]
        var maxVals = data[0]
        
        for row in data {
            guard row.count == colCount else {
                throw PreprocessingError.dimensionMismatch(expected: colCount, got: row.count)
            }
            for col in 0..<colCount {
                minVals[col] = min(minVals[col], row[col])
                maxVals[col] = max(maxVals[col], row[col])
            }
        }
        
        self.dataMin = minVals
        self.dataMax = maxVals
    }
    
    /// Transforms the dataset using the fitted min and max bounds.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard let dataMin = self.dataMin, let dataMax = self.dataMax else {
            throw PreprocessingError.fitNotCalled
        }
        guard !data.isEmpty else {
            return []
        }
        
        let colCount = dataMin.count
        var transformed = [[Double]]()
        transformed.reserveCapacity(data.count)
        
        for row in data {
            guard row.count == colCount else {
                throw PreprocessingError.dimensionMismatch(expected: colCount, got: row.count)
            }
            let scaledRow = (0..<colCount).map { col -> Double in
                let denom = dataMax[col] - dataMin[col]
                if denom < 1e-12 {
                    return range.min
                }
                let ratio = (row[col] - dataMin[col]) / denom
                return ratio * (range.max - range.min) + range.min
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
