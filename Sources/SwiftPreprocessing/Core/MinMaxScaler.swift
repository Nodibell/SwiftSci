import Foundation
import Accelerate

// MARK: - MinMaxScaler (3.5)
//
// Thread Safety: NSLock guards the mutable `dataMin` / `dataMax` so `fit`
// and `transform` are individually thread-safe.
//
// Performance: `transform` uses vDSP for per-column subtract/divide/
// multiply+add in a single vectorized sweep.

/// Scales features to a specified target range (default `[0, 1]`).
///
/// For every column *j*:
/// ```
///   x' = (x - min[j]) / (max[j] - min[j]) * (rangeMax - rangeMin) + rangeMin
/// ```
/// If `max[j] == min[j]` the column is mapped to `rangeMin`.
///
/// ## Performance
/// All per-column arithmetic in `transform` uses `vDSP` (Accelerate).
///
/// ## Thread Safety
/// `fit` and `transform` are individually thread-safe. Do **not** call
/// `fit` concurrently from multiple threads.
public struct MinMaxScaler: PreprocessingTransformer, @unchecked Sendable {
    /// Per-column minimum values from `fit`.
    public private(set) var dataMin: [Double]?
    /// Per-column maximum values from `fit`.
    public private(set) var dataMax: [Double]?
    /// The target output range.
    public let range: (min: Double, max: Double)

    private let lock = NSLock()

    /// Creates a new MinMaxScaler.
    /// - Parameter range: Desired output range, default `(0.0, 1.0)`.
    public init(range: (Double, Double) = (0.0, 1.0)) {
        self.range = range
    }

    // MARK: - fit

    /// Computes per-column min/max from `data`.
    ///
    /// - Parameter data: A 2-D matrix in row-major format `[rows, cols]`.
    /// - Throws: `PreprocessingError.emptyInput` or `dimensionMismatch`.
    public mutating func fit(_ data: [[Double]]) throws {
        guard !data.isEmpty, !data[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
        let cols = data[0].count
        for row in data where row.count != cols {
            throw PreprocessingError.dimensionMismatch(expected: cols, got: row.count)
        }

        let rows = data.count
        var minVals = [Double](repeating:  Double.infinity, count: cols)
        var maxVals = [Double](repeating: -Double.infinity, count: cols)
        var colBuf  = [Double](repeating: 0.0, count: rows)

        for c in 0..<cols {
            for r in 0..<rows { colBuf[r] = data[r][c] }

            var mn = 0.0, mx = 0.0
            vDSP_minvD(colBuf, 1, &mn, vDSP_Length(rows))
            vDSP_maxvD(colBuf, 1, &mx, vDSP_Length(rows))
            minVals[c] = mn
            maxVals[c] = mx
        }

        lock.lock()
        self.dataMin = minVals
        self.dataMax = maxVals
        lock.unlock()
    }

    // MARK: - transform

    /// Scales `data` using the fitted min/max bounds.
    ///
    /// - Parameter data: A 2-D matrix, same column count as used in `fit`.
    /// - Returns: Scaled matrix in the target range.
    /// - Throws: `PreprocessingError.fitNotCalled` if `fit` was not called first.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        lock.lock()
        let dataMin = self.dataMin
        let dataMax = self.dataMax
        lock.unlock()

        guard let dataMin, let dataMax else {
            throw PreprocessingError.fitNotCalled
        }
        guard !data.isEmpty else { return [] }

        let cols    = dataMin.count
        let rangeW  = range.max - range.min    // target width

        // Pre-compute per-column scale (1 / span * rangeW); span = 0 → 0
        var scales = [Double](repeating: 0.0, count: cols)
        for c in 0..<cols {
            let span = dataMax[c] - dataMin[c]
            scales[c] = span < 1e-12 ? 0.0 : rangeW / span
        }

        var transformed = [[Double]](repeating: [Double](repeating: range.min, count: cols),
                                     count: data.count)
        let negMin = dataMin.map { -$0 }

        for (r, row) in data.enumerated() {
            guard row.count == cols else {
                throw PreprocessingError.dimensionMismatch(expected: cols, got: row.count)
            }
            // shifted = row - dataMin
            var shifted = [Double](repeating: 0.0, count: cols)
            vDSP_vaddD(row, 1, negMin, 1, &shifted, 1, vDSP_Length(cols))

            // scaled = shifted * scales
            var scaled = [Double](repeating: 0.0, count: cols)
            vDSP_vmulD(shifted, 1, scales, 1, &scaled, 1, vDSP_Length(cols))

            // result = scaled + rangeMin
            var offset = range.min
            vDSP_vsaddD(scaled, 1, &offset, &transformed[r], 1, vDSP_Length(cols))
        }

        return transformed
    }

    /// Fits to `data` then returns the scaled result.
    public mutating func fitTransform(_ data: [[Double]]) throws -> [[Double]] {
        try fit(data)
        return try transform(data)
    }
}
