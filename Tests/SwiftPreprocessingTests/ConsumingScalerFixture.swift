import SwiftDataFrame

func makeConsumingScalerFixture(rows: Int, width: Int) throws -> PreparedNumericBatch {
    var names: [String] = []
    var columns: [[Double]] = []
    names.reserveCapacity(width)
    columns.reserveCapacity(width)
    for columnIndex in 0..<width {
        names.append("x\(columnIndex)")
        var values: [Double] = []
        values.reserveCapacity(rows)
        for rowIndex in 0..<rows {
            let numerator: Int = (rowIndex * 19 + columnIndex * 11) % 113
            let value: Double = Double(numerator) / 13.0 - 4.0
            values.append(value)
        }
        columns.append(values)
    }
    return try PreparedNumericBatch(columnNames: names, columns: columns)
}
