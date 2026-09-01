import Foundation
import Accelerate

// MARK: - RobustScaler (3.5)
//
// Thread Safety: NSLock guards mutable `center` / `scale`.
// Performance: `transform` uses vDSP subtract and divide per row.

/// Scales features using statistics robust to outliers (median and IQR).
///
/// For every column *j*:
/// ```
///   x' = (x - median[j]) / IQR[j]
/// ```
/// `withCentering` and `withScaling` flags independently enable
/// median subtraction and IQR division.
///
/// ## Performance
/// The `transform` path uses `vDSP` for vectorized subtract / divide.
///
/// ## Thread Safety
/// `fit` and `transform` are individually thread-safe.
public struct RobustScaler: PreprocessingTransformer, @unchecked Sendable {
    /// Whether to subtract the median.
    public let withCentering: Bool
    /// Whether to divide by the IQR.
    public let withScaling: Bool
    /// Quantile range used to compute the IQR (default `(25.0, 75.0)`).
    public let quantileRange: (Double, Double)

    /// Per-column medians computed during `fit`.
    public private(set) var center: [Double]?
    /// Per-column IQR computed during `fit`.
    public private(set) var scale: [Double]?

    private let lock = NSLock()

    /// Creates a new RobustScaler.
    /// - Parameters:
    ///   - withCentering: Subtract the median (default `true`).
    ///   - withScaling: Divide by the IQR (default `true`).
    ///   - quantileRange: Quantile bounds for IQR computation (default `(25.0, 75.0)`).
    public init(
        withCentering: Bool = true,
        withScaling: Bool = true,
        quantileRange: (Double, Double) = (25.0, 75.0)
    ) {
        self.withCentering  = withCentering
        self.withScaling    = withScaling
        self.quantileRange  = quantileRange
    }

    // MARK: - fit

    /// Computes per-column median and IQR from `data`.
    ///
    /// - Parameter data: A 2-D matrix `[rows, cols]`.
    /// - Throws: `PreprocessingError.emptyInput` or `dimensionMismatch`.
    public mutating func fit(_ data: [[Double]]) throws {
        guard !data.isEmpty, !data[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
        let rows = data.count
        let cols = data[0].count
        for row in data where row.count != cols {
            throw PreprocessingError.dimensionMismatch(expected: cols, got: row.count)
        }

        var computedCenter = [Double](repeating: 0.0, count: cols)
        var computedScale  = [Double](repeating: 1.0, count: cols)

        for c in 0..<cols {
            var colValues = [Double](repeating: 0.0, count: rows)
            for r in 0..<rows { colValues[r] = data[r][c] }
            colValues.sort()

            computedCenter[c] = percentile(colValues, 50.0)
            let qLow  = percentile(colValues, quantileRange.0)
            let qHigh = percentile(colValues, quantileRange.1)
            let iqr   = qHigh - qLow
            computedScale[c]  = iqr < 1e-12 ? 1.0 : iqr
        }

        lock.lock()
        self.center = computedCenter
        self.scale  = computedScale
        lock.unlock()
    }

    // MARK: - transform

    /// Scales `data` using the fitted median and IQR.
    ///
    /// - Parameter data: A 2-D matrix, same column count as used in `fit`.
    /// - Returns: Robustly scaled matrix.
    /// - Throws: `PreprocessingError.fitNotCalled` if `fit` was not called first.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        lock.lock()
        let center = self.center
        let scale  = self.scale
        lock.unlock()

        guard let center, let scale else {
            throw PreprocessingError.fitNotCalled
        }
        guard !data.isEmpty else { return [] }

        let cols = center.count
        var transformed = [[Double]](repeating: [Double](repeating: 0.0, count: cols),
                                     count: data.count)

        // Prepare subtracted center buffer (negated median for vDSP add)
        let negCenter = center.map { -$0 }

        for (r, row) in data.enumerated() {
            guard row.count == cols else {
                throw PreprocessingError.dimensionMismatch(expected: cols, got: row.count)
            }

            var result = row

            if withCentering {
                // result = row + (-center)  →  row - center
                vDSP_vaddD(row, 1, negCenter, 1, &result, 1, vDSP_Length(cols))
            }

            if withScaling {
                // result = result / scale
                vDSP_vdivD(scale, 1, result, 1, &result, 1, vDSP_Length(cols))
            }

            transformed[r] = result
        }

        return transformed
    }

    /// Fits to `data` then returns the scaled result.
    public mutating func fitTransform(_ data: [[Double]]) throws -> [[Double]] {
        try fit(data)
        return try transform(data)
    }

    // MARK: - Private helpers

    /// Computes the q-th percentile of a pre-sorted array using linear interpolation.
    private func percentile(_ sortedData: [Double], _ q: Double) -> Double {
        let n = sortedData.count
        guard n > 0 else { return 0.0 }
        guard n > 1  else { return sortedData[0] }
        let p   = q / 100.0
        let idx = p * Double(n - 1)
        let i   = Int(floor(idx))
        let j   = min(i + 1, n - 1)
        return sortedData[i] + (idx - Double(i)) * (sortedData[j] - sortedData[i])
    }
}
