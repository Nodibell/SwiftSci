import Accelerate
import SwiftDataFrame

extension StandardScaler {
    /// Scales a matrix using fitted statistics while preserving the input snapshot.
    /// Feature order must match fitting. Layout, schema and row provenance are preserved.
    /// Throws if fitting has not occurred or a nonempty matrix has the wrong width.
    public func transform(_ matrix: PreparedNumericMatrix) throws -> PreparedNumericMatrix {
        try transform(consuming: matrix)
    }

    /// Reuses a uniquely owned matrix buffer and row-sized scratch for standardization.
    /// Pass `consume matrix` to surrender the input. Shared snapshots detach on mutation.
    /// Uses the same row-width vDSP operations as existing transforms. Statistics are
    /// never refitted here. NaNs participate in arithmetic, matching the row interface.
    /// Throws if fitting has not occurred or a nonempty matrix has the wrong width.
    public func transform(consuming matrix: consuming PreparedNumericMatrix) throws -> PreparedNumericMatrix {
        let (means, deviations) = try fittedParameters()
        guard matrix.rowCount > 0 else { return matrix }
        guard matrix.columnCount == means.count else {
            throw PreprocessingError.dimensionMismatch(expected: means.count, got: matrix.columnCount)
        }
        var output = consume matrix
        let width = output.columnCount, rows = output.rowCount, order = output.order
        let negativeMeans = means.map { -$0 }
        var row = [Double](repeating: 0, count: width)
        var shifted = row, normalized = row
        output.withMutableValues { values in
            for r in 0..<rows {
                for c in 0..<width { row[c] = values[order == .rowMajor ? r * width + c : c * rows + r] }
                row.withUnsafeBufferPointer { input in
                    standardizeRow(input.baseAddress!, negativeMeans: negativeMeans,
                                   deviations: deviations, shifted: &shifted, output: &normalized)
                }
                for c in 0..<width { values[order == .rowMajor ? r * width + c : c * rows + r] = normalized[c] }
            }
        }
        return output
    }
}

func standardizeRow(_ input: UnsafePointer<Double>, negativeMeans: [Double], deviations: [Double],
                    shifted: inout [Double], output: inout [Double]) {
    let width = vDSP_Length(negativeMeans.count)
    vDSP_vaddD(input, 1, negativeMeans, 1, &shifted, 1, width)
    // Array-backed output preserves the existing division rounding; a differently
    // aligned pointer into a matrix or tile can select different vDSP arithmetic.
    vDSP_vdivD(deviations, 1, shifted, 1, &output, 1, width)
}
