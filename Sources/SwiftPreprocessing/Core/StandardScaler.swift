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
    /// Uses native compact-column fitting and transformation in pipelines.
    public var supportsNativePreparedBatches: Bool { true }

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
        var shifted = [Double](repeating: 0.0, count: rows)

        for c in 0..<cols {
            for r in 0..<rows { colBuf[r] = data[r][c] }

            // vDSP mean
            var m = 0.0
            vDSP_meanvD(colBuf, 1, &m, vDSP_Length(rows))

            // vDSP variance: E[(x-μ)²]
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
        var shifted = [Double](repeating: 0.0, count: cols)

        for (r, row) in data.enumerated() {
            guard row.count == cols else {
                throw PreprocessingError.dimensionMismatch(expected: cols, got: row.count)
            }
            // vDSP: (row - mean) / std  element-wise
            vDSP_vaddD(row, 1, negMean, 1, &shifted, 1, vDSP_Length(cols))
            vDSP_vdivD(std, 1, shifted, 1, &transformed[r], 1, vDSP_Length(cols))
        }

        return transformed
    }

    /// Fits to `data` then returns the standardized result.
    /// - Parameters:
    ///   - data: Raw input data array or matrix for transformation.
    /// - Throws: `PreprocessingError` or `SwiftMLError` if columns are missing, types are invalid, or arrays are empty.
    /// - Returns: 2D numerical matrix of shape `[N, P]`.
    public mutating func fitTransform(_ data: [[Double]]) throws -> [[Double]] {
        try fit(data)
        return try transform(data)
    }
}


extension StandardScaler {
    /// Fits directly from compact columns without constructing nested rows.
    public mutating func fit(_ batch: PreparedNumericBatch) throws {
        guard batch.rowCount > 0, batch.columnCount > 0 else { throw PreprocessingError.emptyInput }
        var means = [Double](), deviations = [Double]()
        means.reserveCapacity(batch.columnCount); deviations.reserveCapacity(batch.columnCount)
        var shifted = [Double](repeating: 0, count: batch.rowCount)
        for column in batch.columns {
            var mean = 0.0
            vDSP_meanvD(column.values, 1, &mean, vDSP_Length(batch.rowCount))
            var negativeMean = -mean
            vDSP_vsaddD(column.values, 1, &negativeMean, &shifted, 1, vDSP_Length(batch.rowCount))
            var sumSquares = 0.0
            vDSP_svesqD(shifted, 1, &sumSquares, vDSP_Length(batch.rowCount))
            let deviation = (sumSquares / Double(batch.rowCount)).squareRoot()
            means.append(mean); deviations.append(deviation < 1e-12 ? 1 : deviation)
        }
        lock.lock()
        mean = means; std = deviations
        lock.unlock()
    }

    /// Transforms compact columns while preserving the input snapshot.
    /// Missing inputs participate as NaN and every output is a numeric value.
    public func transform(_ batch: PreparedNumericBatch) throws -> PreparedNumericBatch {
        try transform(consuming: batch)
    }

    /// Transforms a batch, reusing its column storage when uniquely owned.
    /// Pass `consume batch` when the caller no longer needs the input. Shared
    /// arrays detach through copy-on-write, preserving other batch and matrix
    /// snapshots. Reuse depends on ownership and is not guaranteed for aliases.
    /// Column names and original row indices are preserved. Missing inputs
    /// become numeric NaNs, matching the nonconsuming transform.
    public func transform(consuming batch: consuming PreparedNumericBatch) throws -> PreparedNumericBatch {
        lock.lock()
        let means = mean, deviations = std
        lock.unlock()
        guard let means, let deviations else { throw PreprocessingError.fitNotCalled }
        guard batch.rowCount > 0 else { return batch.replacingNumericColumns(Array(repeating: [], count: batch.columnCount)) }
        guard batch.columnCount == means.count else {
            throw PreprocessingError.dimensionMismatch(expected: means.count, got: batch.columnCount)
        }
        var output = consume batch
        let width = output.columnCount
        guard width > 0 else { return output }
        // Bound tiled traversal to 16 KiB; short or wide batches use one row. Both
        // paths keep row-width vDSP arithmetic to preserve existing rounding.
        let capacity = output.rowCount >= 32 && width <= 512 ? min(output.rowCount, 2048 / width) : 1
        var tile = [Double](repeating: 0, count: capacity * width)
        var shifted = [Double](repeating: 0, count: width)
        var normalizedRow = shifted
        let negativeMeans = means.map { -$0 }
        tile.withUnsafeMutableBufferPointer { buffer in
            for start in stride(from: 0, to: output.rowCount, by: capacity) {
                let count = min(capacity, output.rowCount - start)
                for c in 0..<width {
                    output.columns[c].values.withUnsafeBufferPointer { input in
                        for r in 0..<count { buffer[r * width + c] = input[start + r] }
                    }
                }
                for r in 0..<count {
                    let row = buffer.baseAddress!.advanced(by: r * width)
                    vDSP_vaddD(row, 1, negativeMeans, 1, &shifted, 1, vDSP_Length(width))
                    // Array-backed output preserves vDSP division rounding;
                    // writing into differently aligned tile rows can change it.
                    vDSP_vdivD(deviations, 1, shifted, 1, &normalizedRow, 1, vDSP_Length(width))
                    for c in 0..<width { row[c] = normalizedRow[c] }
                }
                for c in 0..<width {
                    output.columns[c].values.withUnsafeMutableBufferPointer { values in
                        for r in 0..<count { values[start + r] = buffer[r * width + c] }
                    }
                }
            }
        }
        for c in 0..<width {
            output.columns[c].validity = nil
            output.columns[c].nullCount = 0
        }
        return output
    }

    /// Fits and transforms a prepared batch, retaining its column order.
    public mutating func fitTransform(_ batch: PreparedNumericBatch) throws -> PreparedNumericBatch {
        try fit(batch)
        return try transform(batch)
    }

    /// Fits dataframe columns directly, preserving the existing conversion rules.
    public mutating func fit(_ frame: DataFrame, columns: [String]) throws {
        try fit(frame.prepareNumericBatch(columns))
    }

    /// Replaces only the requested dataframe columns with their scaled values.
    public func transform(_ frame: DataFrame, columns: [String]) throws -> DataFrame {
        try transform(frame.prepareNumericBatch(columns)).replacingColumns(in: frame)
    }

    /// Prepares dataframe columns once, then fits and transforms them directly.
    public mutating func fitTransform(_ frame: DataFrame, columns: [String]) throws -> DataFrame {
        try fitTransform(frame.prepareNumericBatch(columns)).replacingColumns(in: frame)
    }
}
