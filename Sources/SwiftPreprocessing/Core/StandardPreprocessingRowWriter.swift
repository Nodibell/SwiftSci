import Accelerate
import SwiftDataFrame

/// Per-prediction workspace for filling bounded, strided predictor input buffers.
package struct StandardPreprocessingRowWriter {
    private let input: PreparedNumericBatch
    private let replacements: [Double]
    private let negativeMeans: [Double]
    private let deviations: [Double]
    private let maximumRows: Int
    private var tile: [Double]
    private var shifted: [Double]
    private var normalized: [Double]

    init(input: PreparedNumericBatch, replacements: [Double], negativeMeans: [Double],
         deviations: [Double], maximumRows: Int, tileRows: Int) {
        self.input = input
        self.replacements = replacements
        self.negativeMeans = negativeMeans
        self.deviations = deviations
        self.maximumRows = maximumRows
        tile = [Double](repeating: 0, count: tileRows * input.columnCount)
        shifted = [Double](repeating: 0, count: input.columnCount)
        normalized = [Double](repeating: 0, count: input.columnCount)
    }

    package mutating func fillDouble(rows: Range<Int>, into output: UnsafeMutableBufferPointer<Double>,
                                     stride: Int) throws {
        try fill(rows: rows, into: output, stride: stride)
    }

    mutating func fill<T: BinaryFloatingPoint>(rows: Range<Int>,
        into output: UnsafeMutableBufferPointer<T>, stride: Int) throws {
        try tile.withUnsafeMutableBufferPointer { buffer in
            try fillRows(input: input, rows: rows, maximumRows: maximumRows, output: output,
                stride: stride, buffer: buffer, replacements: replacements,
                negativeMeans: negativeMeans, deviations: deviations,
                shifted: &shifted, normalized: &normalized)
        }
    }
}

private func fillRows<T: BinaryFloatingPoint>(input: PreparedNumericBatch, rows: Range<Int>,
    maximumRows: Int, output: UnsafeMutableBufferPointer<T>, stride: Int,
    buffer: UnsafeMutableBufferPointer<Double>, replacements: [Double],
    negativeMeans: [Double], deviations: [Double], shifted: inout [Double], normalized: inout [Double]
) throws {
    let width = input.columnCount
    guard rows.lowerBound >= 0, rows.upperBound <= input.rowCount else {
        throw SwiftMLError.invalidParameter("Preprocessing row range differs")
    }
    let (required, overflow) = rows.count.multipliedReportingOverflow(by: stride)
    guard rows.count <= maximumRows, stride >= width, !overflow, required <= output.count else {
        throw SwiftMLError.invalidParameter("Preprocessing row range, stride, or capacity differs")
    }
    guard !rows.isEmpty, width > 0 else { return }
    let capacity = buffer.count / width
    for start in Swift.stride(from: 0, to: rows.count, by: capacity) {
        try Task.checkCancellation()
        let count = min(capacity, rows.count - start)
        for c in 0..<width {
            input.columns[c].values.withUnsafeBufferPointer { column in
                for r in 0..<count {
                    let value = column[rows.lowerBound + start + r]
                    buffer[r * width + c] = value.isNaN ? replacements[c] : value
                }
            }
        }
        normalizePreprocessingTile(buffer: buffer, count: count, width: width, output: output,
            destinationRow: start, stride: stride, negativeMeans: negativeMeans, deviations: deviations,
            shifted: &shifted, normalized: &normalized)
    }
}
