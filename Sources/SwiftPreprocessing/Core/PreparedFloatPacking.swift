import SwiftDataFrame

/// Bulk conversion for validated prepared batches. Preserves scalar Float
/// conversion and row-major output. Missing/finite policy belongs to the calling boundary.
package enum PreparedFloatPacking {
    package static func initialize(_ batch: PreparedNumericBatch,
                                    into destination: UnsafeMutableBufferPointer<Float>) {
        let rows = batch.rowCount, columns = batch.columnCount
        let (count, overflow) = rows.multipliedReportingOverflow(by: columns)
        precondition(!overflow && destination.count == count)
        guard count > 0 else { return }
        let output = destination.baseAddress!
        // Bound strided writes while reading contiguous column segments.
        let tile = 16
        for rowStart in stride(from: 0, to: rows, by: tile) {
            let rowEnd = min(rowStart + tile, rows)
            for columnStart in stride(from: 0, to: columns, by: tile) {
                let columnEnd = min(columnStart + tile, columns)
                for column in columnStart..<columnEnd {
                    batch.columns[column].values.withUnsafeBufferPointer { values in
                        for row in rowStart..<rowEnd {
                            output.advanced(by: row * columns + column).initialize(to: Float(values[row]))
                        }
                    }
                }
            }
        }
    }
}
