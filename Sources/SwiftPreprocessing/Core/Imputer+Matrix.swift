import SwiftDataFrame

extension Imputer {
    /// Replaces matrix NaNs with fitted statistics, preserving the input snapshot.
    /// Feature order must match fitting. Layout, schema and row provenance are preserved.
    /// Throws if fitting has not occurred or a nonempty matrix has the wrong width.
    public func transform(_ matrix: PreparedNumericMatrix) throws -> PreparedNumericMatrix {
        try transform(consuming: matrix)
    }

    /// Reuses a uniquely owned matrix buffer when replacing NaNs with fitted statistics.
    /// Pass `consume matrix` to surrender the input. Shared snapshots detach on mutation;
    /// clean matrices keep their storage. Nil values converted to NaN during extraction
    /// receive the same treatment as numeric NaNs. Statistics are never refitted here.
    /// Throws if fitting has not occurred or a nonempty matrix has the wrong width.
    public func transform(consuming matrix: consuming PreparedNumericMatrix) throws -> PreparedNumericMatrix {
        guard let statistics else { throw PreprocessingError.fitNotCalled }
        guard matrix.rowCount > 0 else { return matrix }
        guard matrix.columnCount == statistics.count else {
            throw PreprocessingError.dimensionMismatch(expected: statistics.count, got: matrix.columnCount)
        }
        guard matrix.values.contains(where: { $0.isNaN }) else { return matrix }
        var output = consume matrix
        let rows = output.rowCount, width = output.columnCount, order = output.order
        output.withMutableValues { values in
            for column in 0..<width {
                let start = order == .rowMajor ? column : column * rows
                let stride = order == .rowMajor ? width : 1
                imputeMissing(values, start: start, stride: stride, count: rows, replacement: statistics[column])
            }
        }
        return output
    }
}

func imputeMissing(_ values: UnsafeMutableBufferPointer<Double>, start: Int,
                   stride: Int, count: Int, replacement: Double) {
    for row in 0..<count {
        let index = start + row * stride
        if values[index].isNaN { values[index] = replacement }
    }
}
