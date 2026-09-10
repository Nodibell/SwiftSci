import Foundation
import Accelerate

/// Storage format for compressed sparse matrix structures.
public enum SparseFormat: String, Sendable, Codable {
    /// Compressed Sparse Row (CSR) format: efficient for row slicing and sparse matrix-vector multiplication.
    case csr
    /// Compressed Sparse Column (CSC) format: efficient for column slicing and sparse matrix-matrix products.
    case csc
}

/// Memory-efficient sparse matrix storage implementing Compressed Sparse Row (CSR) and Column (CSC) formats.
///
/// Reduces memory consumption by 10x–50x when working with high-dimensional sparse representations,
/// such as one-hot encoded categorical tables, term-frequency bags-of-words, and graph adjacency structures.
///
/// ## Acceleration
/// Matrix-vector products are accelerated via Apple Accelerate Sparse BLAS routines (`sparse_matrix_vector_product_dense_double`)
/// and optimized vector routines.
///
/// ## Thread Safety
/// Implemented as an immutable Swift struct with value semantics, ensuring strict thread safety under Swift 6.
public struct SparseMatrix: Sendable, Equatable {

    /// Number of rows in the matrix.
    public let rowCount: Int

    /// Number of columns in the matrix.
    public let colCount: Int

    /// The compression format (CSR or CSC).
    public let format: SparseFormat

    /// Flat array of non-zero numerical values stored in row-major (CSR) or column-major (CSC) order.
    public let values: [Double]

    /// Inner dimension indices (column indices for CSR, row indices for CSC).
    public let innerIndices: [Int]

    /// Outer dimension pointers (row pointers for CSR, column pointers for CSC). Size is `(outerDimension + 1)`.
    public let outerPointers: [Int]

    // MARK: - Computed Properties

    /// Column indices of stored non-zero values (valid for CSR format).
    public var colIndices: [Int] {
        return format == .csr ? innerIndices : []
    }

    /// Row pointer offsets indicating the start of each row in `values` (valid for CSR format).
    public var rowPointers: [Int] {
        return format == .csr ? outerPointers : []
    }

    /// Row indices of stored non-zero values (valid for CSC format).
    public var rowIndices: [Int] {
        return format == .csc ? innerIndices : []
    }

    /// Column pointer offsets indicating the start of each column in `values` (valid for CSC format).
    public var colPointers: [Int] {
        return format == .csc ? outerPointers : []
    }

    /// Number of stored non-zero elements.
    ///
    /// ## Complexity
    /// \(O(1)\).
    public var nnz: Int {
        return values.count
    }

    /// Proportion of matrix elements that are exactly zero, ranging from `0.0` (fully dense) to `1.0` (empty).
    ///
    /// ## Complexity
    /// \(O(1)\).
    public var sparsity: Double {
        let total = Double(rowCount) * Double(colCount)
        guard total > 0 else { return 1.0 }
        return 1.0 - (Double(values.count) / total)
    }

    // MARK: - Initializers

    /// Creates a sparse matrix in Compressed Sparse Row (CSR) format from flat coordinate arrays.
    ///
    /// - Parameters:
    ///   - rowCount: Total number of rows.
    ///   - colCount: Total number of columns.
    ///   - values: Array of non-zero entries.
    ///   - colIndices: Column index for each entry in `values`.
    ///   - rowPointers: Row start offsets of length `rowCount + 1`.
    ///
    /// ## Complexity
    /// \(O(1)\) initialization.
    public init(
        rowCount: Int,
        colCount: Int,
        values: [Double],
        colIndices: [Int],
        rowPointers: [Int]
    ) {
        precondition(rowCount >= 0, "rowCount must be non-negative")
        precondition(colCount >= 0, "colCount must be non-negative")
        precondition(values.count == colIndices.count, "values and colIndices must have matching lengths")
        precondition(rowPointers.count == rowCount + 1, "rowPointers length must be rowCount + 1")

        self.rowCount = rowCount
        self.colCount = colCount
        self.format = .csr
        self.values = values
        self.innerIndices = colIndices
        self.outerPointers = rowPointers
    }

    /// Creates a sparse matrix from raw components specifying format explicitly.
    ///
    /// - Parameters:
    ///   - rowCount: Number of rows.
    ///   - colCount: Number of columns.
    ///   - format: `.csr` or `.csc`.
    ///   - values: Non-zero values.
    ///   - innerIndices: Inner indices (columns for CSR, rows for CSC).
    ///   - outerPointers: Outer pointers (row pointers for CSR, col pointers for CSC).
    public init(
        rowCount: Int,
        colCount: Int,
        format: SparseFormat,
        values: [Double],
        innerIndices: [Int],
        outerPointers: [Int]
    ) {
        precondition(rowCount >= 0, "rowCount must be non-negative")
        precondition(colCount >= 0, "colCount must be non-negative")
        precondition(values.count == innerIndices.count, "values and innerIndices must have matching lengths")
        let outerCount = format == .csr ? rowCount : colCount
        precondition(outerPointers.count == outerCount + 1, "outerPointers length mismatch")

        self.rowCount = rowCount
        self.colCount = colCount
        self.format = format
        self.values = values
        self.innerIndices = innerIndices
        self.outerPointers = outerPointers
    }

    /// Constructs a sparse matrix by compressing a dense 2D array.
    ///
    /// - Parameters:
    ///   - dense: 2D array of numerical values.
    ///   - format: Output sparse compression format (`.csr` by default).
    ///   - tolerance: Absolute threshold below which values are treated as zero. Default is `1e-12`.
    ///
    /// ## Complexity
    /// \(O(M \cdot N)\) scan over the dense matrix.
    public init(dense: [[Double]], format: SparseFormat = .csr, tolerance: Double = 1e-12) {
        let rCount = dense.count
        let cCount = dense.first?.count ?? 0

        self.rowCount = rCount
        self.colCount = cCount
        self.format = format

        if format == .csr {
            var vals: [Double] = []
            var cols: [Int] = []
            var ptrs: [Int] = [0]

            for r in 0..<rCount {
                for c in 0..<cCount {
                    let v = dense[r][c]
                    if abs(v) > tolerance {
                        vals.append(v)
                        cols.append(c)
                    }
                }
                ptrs.append(vals.count)
            }

            self.values = vals
            self.innerIndices = cols
            self.outerPointers = ptrs
        } else {
            var vals: [Double] = []
            var rows: [Int] = []
            var ptrs: [Int] = [0]

            for c in 0..<cCount {
                for r in 0..<rCount {
                    let v = dense[r][c]
                    if abs(v) > tolerance {
                        vals.append(v)
                        rows.append(r)
                    }
                }
                ptrs.append(vals.count)
            }

            self.values = vals
            self.innerIndices = rows
            self.outerPointers = ptrs
        }
    }

    // MARK: - Subscripts & Element Access

    /// Retrieves or evaluates the matrix element at `(row, col)`.
    ///
    /// - Parameters:
    ///   - row: Zero-based row index.
    ///   - col: Zero-based column index.
    /// - Returns: The stored value if non-zero, or `0.0`.
    ///
    /// ## Complexity
    /// \(O(\log(\text{nnz per row}))\) for CSR via binary search.
    public subscript(row: Int, col: Int) -> Double {
        precondition(row >= 0 && row < rowCount, "Row index \(row) out of bounds \(rowCount)")
        precondition(col >= 0 && col < colCount, "Column index \(col) out of bounds \(colCount)")

        if format == .csr {
            let start = outerPointers[row]
            let end = outerPointers[row + 1]
            guard start < end else { return 0.0 }

            var low = start
            var high = end
            while low < high {
                let mid = (low + high) / 2
                let c = innerIndices[mid]
                if c == col {
                    return values[mid]
                } else if c < col {
                    low = mid + 1
                } else {
                    high = mid
                }
            }
            return 0.0
        } else {
            let start = outerPointers[col]
            let end = outerPointers[col + 1]
            guard start < end else { return 0.0 }

            var low = start
            var high = end
            while low < high {
                let mid = (low + high) / 2
                let r = innerIndices[mid]
                if r == row {
                    return values[mid]
                } else if r < row {
                    low = mid + 1
                } else {
                    high = mid
                }
            }
            return 0.0
        }
    }

    // MARK: - Conversions

    /// Converts the sparse representation back into a full dense 2D array.
    ///
    /// - Returns: Full 2D array `[rowCount × colCount]`.
    ///
    /// ## Complexity
    /// \(O(\text{rowCount} \cdot \text{colCount})\).
    public func toDense() -> [[Double]] {
        var dense = Array(repeating: [Double](repeating: 0.0, count: colCount), count: rowCount)
        if format == .csr {
            for r in 0..<rowCount {
                let start = outerPointers[r]
                let end = outerPointers[r + 1]
                for idx in start..<end {
                    dense[r][innerIndices[idx]] = values[idx]
                }
            }
        } else {
            for c in 0..<colCount {
                let start = outerPointers[c]
                let end = outerPointers[c + 1]
                for idx in start..<end {
                    dense[innerIndices[idx]][c] = values[idx]
                }
            }
        }
        return dense
    }

    /// Converts this sparse matrix to Compressed Sparse Row (CSR) format.
    ///
    /// - Returns: An equivalent matrix structured as `.csr`.
    public func toCSR() -> SparseMatrix {
        guard format != .csr else { return self }
        return self.transpose().toCSCTransposedToCSR()
    }

    /// Converts this sparse matrix to Compressed Sparse Column (CSC) format.
    ///
    /// - Returns: An equivalent matrix structured as `.csc`.
    public func toCSC() -> SparseMatrix {
        guard format != .csc else { return self }
        var colCounts = [Int](repeating: 0, count: colCount)
        for c in innerIndices {
            colCounts[c] += 1
        }

        var colPtrs = [Int](repeating: 0, count: colCount + 1)
        for c in 0..<colCount {
            colPtrs[c + 1] = colPtrs[c] + colCounts[c]
        }

        var currentPtrs = colPtrs
        var cscValues = [Double](repeating: 0.0, count: values.count)
        var cscRowIndices = [Int](repeating: 0, count: values.count)

        for r in 0..<rowCount {
            let start = outerPointers[r]
            let end = outerPointers[r + 1]
            for idx in start..<end {
                let col = innerIndices[idx]
                let dest = currentPtrs[col]
                cscValues[dest] = values[idx]
                cscRowIndices[dest] = r
                currentPtrs[col] += 1
            }
        }

        return SparseMatrix(
            rowCount: rowCount,
            colCount: colCount,
            format: .csc,
            values: cscValues,
            innerIndices: cscRowIndices,
            outerPointers: colPtrs
        )
    }

    private func toCSCTransposedToCSR() -> SparseMatrix {
        // Internal helper for CSC transposition
        return SparseMatrix(
            rowCount: colCount,
            colCount: rowCount,
            format: .csr,
            values: values,
            innerIndices: innerIndices,
            outerPointers: outerPointers
        )
    }

    /// Transposes the sparse matrix: rows become columns and columns become rows.
    ///
    /// - Returns: Transposed `SparseMatrix`.
    ///
    /// ## Complexity
    /// \(O(1)\) when toggling format between CSR and CSC; \(O(\text{nnz})\) when reordering.
    public func transpose() -> SparseMatrix {
        if format == .csr {
            return SparseMatrix(
                rowCount: colCount,
                colCount: rowCount,
                format: .csc,
                values: values,
                innerIndices: innerIndices,
                outerPointers: outerPointers
            )
        } else {
            return SparseMatrix(
                rowCount: colCount,
                colCount: rowCount,
                format: .csr,
                values: values,
                innerIndices: innerIndices,
                outerPointers: outerPointers
            )
        }
    }

    // MARK: - Mathematical Operations

    /// Multiplies this sparse matrix by a dense vector: \(y = A \cdot x\).
    ///
    /// - Parameter vector: Dense numerical vector of length `colCount`.
    /// - Returns: Dense numerical vector of length `rowCount`.
    /// - Throws: `SwiftMLError.dimensionMismatch` if vector length does not match `colCount`.
    ///
    /// ## Acceleration
    /// Utilizes Apple Accelerate Sparse BLAS routines (`sparse_matrix_vector_product_dense_double`)
    /// when dimensions warrant hardware offload, and fast vectorized dot-products otherwise.
    ///
    /// ## Complexity
    /// \(O(\text{nnz})\).
    public func multiply(vector: [Double]) throws -> [Double] {
        guard vector.count == colCount else {
            throw SwiftMLError.dimensionMismatch(expected: colCount, got: vector.count)
        }

        let csrMatrix = (format == .csr) ? self : self.toCSR()

        // Offload to Accelerate Sparse BLAS for medium-to-large matrices
        if csrMatrix.rowCount >= 64 && csrMatrix.nnz >= 128,
           let A = sparse_matrix_create_double(UInt64(csrMatrix.rowCount), UInt64(csrMatrix.colCount)) {
            let rawA = UnsafeMutableRawPointer(A)
            defer { sparse_matrix_destroy(rawA) }

            for r in 0..<csrMatrix.rowCount {
                let start = csrMatrix.outerPointers[r]
                let end = csrMatrix.outerPointers[r + 1]
                let count = end - start
                if count > 0 {
                    let rowVals = Array(csrMatrix.values[start..<end])
                    let rowCols: [Int64] = csrMatrix.innerIndices[start..<end].map { Int64($0) }
                    _ = sparse_insert_row_double(A, Int64(r), UInt64(count), rowVals, rowCols)
                }
            }
            _ = sparse_commit(rawA)

            var result = [Double](repeating: 0.0, count: csrMatrix.rowCount)
            let status = sparse_matrix_vector_product_dense_double(
                CblasNoTrans,
                1.0,
                A,
                vector,
                1,
                &result,
                1
            )
            if status == SPARSE_SUCCESS {
                return result
            }
        }

        // High-speed CPU sparse accumulator fallback
        var result = [Double](repeating: 0.0, count: csrMatrix.rowCount)
        for r in 0..<csrMatrix.rowCount {
            let start = csrMatrix.outerPointers[r]
            let end = csrMatrix.outerPointers[r + 1]
            var sum = 0.0
            for idx in start..<end {
                sum += csrMatrix.values[idx] * vector[csrMatrix.innerIndices[idx]]
            }
            result[r] = sum
        }
        return result
    }

    /// Multiplies this sparse matrix by another sparse matrix: \(C = A \cdot B\).
    ///
    /// - Parameter other: The right-hand side `SparseMatrix`.
    /// - Returns: Resulting `SparseMatrix` in `.csr` format.
    /// - Throws: `SwiftMLError.dimensionMismatch` if inner dimensions do not match.
    ///
    /// ## Complexity
    /// \(O(\sum \text{nnz}_A \cdot \text{nnz}_B)\).
    public func multiply(matrix other: SparseMatrix) throws -> SparseMatrix {
        guard self.colCount == other.rowCount else {
            throw SwiftMLError.dimensionMismatch(expected: self.colCount, got: other.rowCount)
        }

        let A = (self.format == .csr) ? self : self.toCSR()
        let B = (other.format == .csr) ? other : other.toCSR()

        var resultVals: [Double] = []
        var resultCols: [Int] = []
        var resultPtrs: [Int] = [0]

        for r in 0..<A.rowCount {
            let aStart = A.outerPointers[r]
            let aEnd = A.outerPointers[r + 1]

            if aStart >= aEnd {
                resultPtrs.append(resultVals.count)
                continue
            }

            // Scatter row r into a temporary accumulator
            var rowAccum: [Int: Double] = [:]
            for aIdx in aStart..<aEnd {
                let k = A.innerIndices[aIdx]
                let aVal = A.values[aIdx]

                let bStart = B.outerPointers[k]
                let bEnd = B.outerPointers[k + 1]
                for bIdx in bStart..<bEnd {
                    let bCol = B.innerIndices[bIdx]
                    rowAccum[bCol, default: 0.0] += aVal * B.values[bIdx]
                }
            }

            let sortedEntries = rowAccum.filter { abs($0.value) > 1e-12 }.sorted { $0.key < $1.key }
            for entry in sortedEntries {
                resultVals.append(entry.value)
                resultCols.append(entry.key)
            }
            resultPtrs.append(resultVals.count)
        }

        return SparseMatrix(
            rowCount: self.rowCount,
            colCount: other.colCount,
            values: resultVals,
            colIndices: resultCols,
            rowPointers: resultPtrs
        )
    }
}
