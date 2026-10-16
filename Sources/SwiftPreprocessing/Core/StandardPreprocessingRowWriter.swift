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
        gatherPreprocessingTile(input, count: count, start: rows.lowerBound + start,
            output: buffer, replacements: replacements)
        normalizePreprocessingTile(buffer: buffer, count: count, width: width, output: output,
            destinationRow: start, stride: stride, negativeMeans: negativeMeans, deviations: deviations,
            shifted: &shifted, normalized: &normalized)
    }
}

private func gatherPreprocessingTile(_ input: PreparedNumericBatch, count: Int, start: Int,
    output: UnsafeMutableBufferPointer<Double>, replacements: [Double]) {
    let width = input.columnCount
    // The caller bounds count by output.count / width, so row offsets cannot overflow.
    let grouped = width / 4 * 4
    for c in stride(from: 0, to: grouped, by: 4) {
        input.columns[c].values.withUnsafeBufferPointer { column0 in
            input.columns[c + 1].values.withUnsafeBufferPointer { column1 in
                input.columns[c + 2].values.withUnsafeBufferPointer { column2 in
                    input.columns[c + 3].values.withUnsafeBufferPointer { column3 in
                        let p0 = column0.baseAddress!.advanced(by: start)
                        let p1 = column1.baseAddress!.advanced(by: start)
                        let p2 = column2.baseAddress!.advanced(by: start)
                        let p3 = column3.baseAddress!.advanced(by: start)
                        for r in 0..<count {
                            let destination = output.baseAddress!.advanced(by: r &* width &+ c)
                            let v0 = p0[r]
                            let v1 = p1[r]
                            let v2 = p2[r]
                            let v3 = p3[r]
                            destination[0] = v0.isNaN ? replacements[c] : v0
                            destination[1] = v1.isNaN ? replacements[c + 1] : v1
                            destination[2] = v2.isNaN ? replacements[c + 2] : v2
                            destination[3] = v3.isNaN ? replacements[c + 3] : v3
                        }
                    }
                }
            }
        }
    }
    for c in grouped..<width {
        input.columns[c].values.withUnsafeBufferPointer { column in
            let source = column.baseAddress!.advanced(by: start)
            for r in 0..<count {
                let v = source[r]
                output[r &* width &+ c] = v.isNaN ? replacements[c] : v
            }
        }
    }
}
