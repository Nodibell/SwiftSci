import Foundation
import Accelerate

// MARK: - StandardScaler (3.5)
//
// Memory layout note: the public API accepts / returns [[Double]] for backward
// compatibility with PreprocessingTransformer.  Internally, `fit` uses
// vDSP column-slice operations (O(N) per column, single pass for mean,
// second pass for variance) and `transform` uses vDSP add/multiply.
//
// Thread Safety: NSLock guards the mutable `mean` and `std` properties so
// that `fit` is safe to call once from any thread while `transform` runs
// concurrently on fitted data. Calling `fit` from multiple threads
// simultaneously is NOT supported (same as sklearn).

/// Standardizes features by removing the mean and scaling to unit variance.
///
/// For every column *j* the transformation is:
/// ```
///   x' = (x - mean[j]) / std[j]
/// ```
/// where `std` is the population standard deviation.  If the std of a
/// column is below `1e-12` it is replaced by `1.0` to avoid division by zero.
///
/// ## Performance
/// All per-column arithmetic uses `vDSP` (Accelerate), so the transform is
/// cache-friendly and SIMD-accelerated on Apple Silicon.
///
/// ## Thread Safety
/// `fit` and `transform` are individually thread-safe. Do **not** call `fit`
/// concurrently from multiple threads.
public struct StandardScaler: PreprocessingTransformer, @unchecked Sendable {
    /// Per-column means computed during `fit`.
    public private(set) var mean: [Double]?
    /// Per-column standard deviations computed during `fit`.
    public private(set) var std: [Double]?

    private let lock = NSLock()

    /// Creates a new StandardScaler.
    public init() {}

    // MARK: - fit

    /// Computes the per-column mean and standard deviation from `data`.
    ///
    /// - Parameter data: A 2-D matrix in **row-major** format, shape `[rows, cols]`.
    /// - Throws: `PreprocessingError.emptyInput` or `dimensionMismatch`.
    public mutating func fit(_ data: [[Double]]) throws {
        guard !data.isEmpty, !data[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
        let rows = data.count
        let cols = data[0].count

        // Validate row lengths
        for row in data where row.count != cols {
            throw PreprocessingError.dimensionMismatch(expected: cols, got: row.count)
        }

        var computedMean = [Double](repeating: 0.0, count: cols)
        var computedStd  = [Double](repeating: 0.0, count: cols)

        // Extract each column into a contiguous buffer for vDSP
        var colBuf = [Double](repeating: 0.0, count: rows)

        for c in 0..<cols {
            for r in 0..<rows { colBuf[r] = data[r][c] }

            // vDSP mean
            var m = 0.0
            vDSP_meanvD(colBuf, 1, &m, vDSP_Length(rows))

            // vDSP variance: E[(x-μ)²]
            var shifted = [Double](repeating: 0.0, count: rows)
            var neg = -m
            vDSP_vsaddD(colBuf, 1, &neg, &shifted, 1, vDSP_Length(rows))
            var sumSq = 0.0
            vDSP_svesqD(shifted, 1, &sumSq, vDSP_Length(rows))
            let variance = sumSq / Double(rows)
            let sd = variance.squareRoot()

            computedMean[c] = m
            computedStd[c]  = sd < 1e-12 ? 1.0 : sd
        }

        lock.lock()
        self.mean = computedMean
        self.std  = computedStd
        lock.unlock()
    }

    // MARK: - transform

    /// Standardizes `data` using the fitted mean and std.
    ///
    /// - Parameter data: A 2-D matrix, same column count as used in `fit`.
    /// - Returns: Standardized matrix, same shape as `data`.
    /// - Throws: `PreprocessingError.fitNotCalled` if `fit` was not called first.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        lock.lock()
        let mean = self.mean
        let std  = self.std
        lock.unlock()

        guard let mean, let std else {
            throw PreprocessingError.fitNotCalled
        }
        guard !data.isEmpty else { return [] }

        let cols = mean.count
        var transformed = [[Double]](repeating: [Double](repeating: 0.0, count: cols), count: data.count)
        let negMean = mean.map { -$0 }

        for (r, row) in data.enumerated() {
            guard row.count == cols else {
                throw PreprocessingError.dimensionMismatch(expected: cols, got: row.count)
            }
            // vDSP: (row - mean) / std  element-wise
            var shifted = [Double](repeating: 0.0, count: cols)
            vDSP_vaddD(row, 1, negMean, 1, &shifted, 1, vDSP_Length(cols))
            vDSP_vdivD(std, 1, shifted, 1, &transformed[r], 1, vDSP_Length(cols))
        }

        return transformed
    }

    /// Fits to `data` then returns the standardized result.
    public mutating func fitTransform(_ data: [[Double]]) throws -> [[Double]] {
        try fit(data)
        return try transform(data)
    }
}
