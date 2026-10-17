import SwiftDataFrame

extension StandardPreprocessingPlan {
    package func makeRowWriter(_ input: IndexedNumericBatch, maximumRows: Int,
                               tiled: Bool = true) throws -> StandardIndexedPreprocessingRowWriter {
        try validate(input.source)
        guard maximumRows >= 0 else { throw SwiftMLError.invalidParameter("Negative indexed row capacity") }
        let capacity = min(max(1, maximumRows), tileRows(rowCount: input.rowCount,
            width: input.columnCount, tiled: tiled))
        return StandardIndexedPreprocessingRowWriter(input: input, plan: self,
            maximumRows: maximumRows, tileRows: capacity)
    }

    // Materialization is for independent qualification, not indexed prediction.
    package func preparedMatrix(_ input: IndexedNumericBatch) throws -> PreparedNumericMatrix {
        var writer = try makeRowWriter(input, maximumRows: input.rowCount)
        var values = [Double](repeating: 0, count: input.rowCount * input.columnCount)
        try values.withUnsafeMutableBufferPointer {
            try writer.fillDouble(rows: 0..<input.rowCount, into: $0, stride: input.columnCount)
        }
        return PreparedNumericMatrix(values: values, columnNames: input.columnNames,
            rowCount: input.rowCount, order: .rowMajor, sourceRows: input.originalRowIndices)
    }
}
