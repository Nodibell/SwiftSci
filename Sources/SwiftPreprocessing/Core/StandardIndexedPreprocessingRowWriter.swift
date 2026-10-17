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
                            if count < 4 {
                                for r in 0..<count {
                                    let sourceRow = input.sourcePositions[rows.lowerBound + start + r]
                                    let value = column[sourceRow]
                                    buffer[r * width + c] = value.isNaN ? replacements[c] : value
                                }
                                return
                            }
                            let first = rows.lowerBound + start
                            var r = 0
                            while r + 4 <= count {
                                let value0 = column[input.sourcePositions[first + r]]
                                let value1 = column[input.sourcePositions[first + r + 1]]
                                let value2 = column[input.sourcePositions[first + r + 2]]
                                let value3 = column[input.sourcePositions[first + r + 3]]
                                let destination = r * width + c
                                buffer[destination] = value0.isNaN ? replacements[c] : value0
                                buffer[destination + width] = value1.isNaN ? replacements[c] : value1
                                buffer[destination + 2 * width] = value2.isNaN ? replacements[c] : value2
                                buffer[destination + 3 * width] = value3.isNaN ? replacements[c] : value3
                                r += 4
                            }
                            while r < count {
                                let value = column[input.sourcePositions[first + r]]
                                buffer[r * width + c] = value.isNaN ? replacements[c] : value
                                r += 1
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
