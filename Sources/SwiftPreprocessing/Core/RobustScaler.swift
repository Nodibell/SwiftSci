import Foundation

/// RobustScaler scales features using statistics that are robust to outliers.
public struct RobustScaler: PreprocessingTransformer, Sendable {
    /// The with centering.
    public let withCentering: Bool
    /// The with scaling.
    public let withScaling: Bool
    /// The quantile range.
    public let quantileRange: (Double, Double)
    
    /// The center.
    public private(set) var center: [Double]?
    /// The scale.
    public private(set) var scale: [Double]?
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - withCentering: The with centering.
    ///   - withScaling: The with scaling.
    ///   - quantileRange: The quantile range.
    public init(withCentering: Bool = true, withScaling: Bool = true, quantileRange: (Double, Double) = (25.0, 75.0)) {
        self.withCentering = withCentering
        self.withScaling = withScaling
        self.quantileRange = quantileRange
    }
    
    /// Fits the scaler by calculating the median and Interquartile Range (IQR) for each column.
    public mutating func fit(_ data: [[Double]]) throws {
        guard !data.isEmpty, !data[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
        
        let rowCount = data.count
        let colCount = data[0].count
        
        var computedCenter = [Double](repeating: 0.0, count: colCount)
        var computedScale = [Double](repeating: 1.0, count: colCount)
        
        for col in 0..<colCount {
            var colValues = [Double]()
            colValues.reserveCapacity(rowCount)
            for row in 0..<rowCount {
                guard data[row].count == colCount else {
                    throw PreprocessingError.dimensionMismatch(expected: colCount, got: data[row].count)
                }
                colValues.append(data[row][col])
            }
            colValues.sort()
            
            let medianVal = percentile(colValues, 50.0)
            let qLow = percentile(colValues, quantileRange.0)
            let qHigh = percentile(colValues, quantileRange.1)
            let iqr = qHigh - qLow
            
            computedCenter[col] = medianVal
            computedScale[col] = iqr < 1e-12 ? 1.0 : iqr
        }
        
        self.center = computedCenter
        self.scale = computedScale
    }
    
    /// Scales the dataset.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard let center = self.center, let scale = self.scale else {
            throw PreprocessingError.fitNotCalled
        }
        guard !data.isEmpty else {
            return []
        }
        
        let colCount = center.count
        var transformed = [[Double]]()
        transformed.reserveCapacity(data.count)
        
        for row in data {
            guard row.count == colCount else {
                throw PreprocessingError.dimensionMismatch(expected: colCount, got: row.count)
            }
            
            let scaledRow = (0..<colCount).map { col -> Double in
                var val = row[col]
                if withCentering {
                    val -= center[col]
                }
                if withScaling {
                    val /= scale[col]
                }
                return val
            }
            transformed.append(scaledRow)
        }
        
        return transformed
    }
    
    private func percentile(_ sortedData: [Double], _ q: Double) -> Double {
        let n = sortedData.count
        if n == 0 { return 0.0 }
        if n == 1 { return sortedData[0] }
        let p = q / 100.0
        let idx = p * Double(n - 1)
        let i = Int(floor(idx))
        let j = Int(ceil(idx))
        let fraction = idx - Double(i)
        return sortedData[i] + fraction * (sortedData[j] - sortedData[i])
    }
}
