import SwiftDataFrame

extension StandardPreprocessingPlan {
    package func preparedMatrix(_ input: PreparedNumericBatch) throws -> PreparedNumericMatrix {
        try validate(input)
        var values = [Double](repeating: 0, count: input.rowCount * input.columnCount)
        try values.withUnsafeMutableBufferPointer { try fillDouble(input, into: $0) }
        return PreparedNumericMatrix(values: values, columnNames: input.columnNames,
            rowCount: input.rowCount, order: .rowMajor, sourceRows: input.originalRowIndices)
    }
}
