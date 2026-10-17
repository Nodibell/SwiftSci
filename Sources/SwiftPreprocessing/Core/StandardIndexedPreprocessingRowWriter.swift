import SwiftDataFrame

package struct StandardIndexedPreprocessingRowWriter {
    private let input: IndexedNumericBatch
    private let replacements: [Double]
    private let negativeMeans: [Double]
    private let deviations: [Double]
    private let maximumRows: Int
    private var tile: [Double]
    private var shifted: [Double]
    private var normalized: [Double]

    init(input: IndexedNumericBatch, plan: StandardPreprocessingPlan, maximumRows: Int, tileRows: Int) {
        self.input = input
        replacements = plan.replacements
        negativeMeans = plan.negativeMeans
        deviations = plan.deviations
        self.maximumRows = maximumRows
        tile = [Double](repeating: 0, count: tileRows * input.columnCount)
        shifted = [Double](repeating: 0, count: input.columnCount)
        normalized = [Double](repeating: 0, count: input.columnCount)
    }

    package mutating func fillDouble(rows: Range<Int>, into output: UnsafeMutableBufferPointer<Double>,
                                     stride: Int) throws {
        guard rows.lowerBound >= 0, rows.upperBound <= input.rowCount else {
            throw SwiftMLError.invalidParameter("Indexed preprocessing row range differs")
        }
        let width = input.columnCount
        let (required, overflow) = rows.count.multipliedReportingOverflow(by: stride)
        guard rows.count <= maximumRows, stride >= width, !overflow, required <= output.count else {
            throw SwiftMLError.invalidParameter("Indexed preprocessing stride or capacity differs")
        }
        guard !rows.isEmpty, width > 0 else { return }
        let capacity = tile.count / width
        try tile.withUnsafeMutableBufferPointer { buffer in
            for start in Swift.stride(from: 0, to: rows.count, by: capacity) {
                try Task.checkCancellation()
                let count = min(capacity, rows.count - start)
                var edited = 0
                for c in 0..<width {
                    if edited < input.editedColumns.count, input.editedColumns[edited].index == c {
                        input.editedColumns[edited].data.values.withUnsafeBufferPointer { column in
                            for r in 0..<count {
                                let value = column[rows.lowerBound + start + r]
                                buffer[r * width + c] = value.isNaN ? replacements[c] : value
                            }
                        }
                        edited += 1
                    } else {
                        input.source.columns[c].values.withUnsafeBufferPointer { column in
                            for r in 0..<count {
                                let sourceRow = input.sourcePositions[rows.lowerBound + start + r]
                                let value = column[sourceRow]
                                buffer[r * width + c] = value.isNaN ? replacements[c] : value
                            }
                        }
                    }
                }
                normalizePreprocessingTile(buffer: buffer, count: count, width: width, output: output,
                    destinationRow: start, stride: stride, negativeMeans: negativeMeans,
                    deviations: deviations, shifted: &shifted, normalized: &normalized)
            }
        }
    }
}
