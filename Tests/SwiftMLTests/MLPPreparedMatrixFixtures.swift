import Testing
import SwiftDataFrame

func matrixMLPFeatures(_ count: Int, width: Int) -> [[Double]] {
    var rows: [[Double]] = []
    for row in 0..<count {
        var values: [Double] = []
        for column in 0..<width {
            let value: Int = (row * 13 + column * 7) % 31 - 15
            values.append(Double(value) / 17)
        }
        rows.append(values)
    }
    return rows
}

func matrixMLPBatch(_ rows: [[Double]], width: Int) throws -> PreparedNumericBatch {
    let names: [String] = (0..<width).map { "x\($0)" }
    let columns: [[Double]] = (0..<width).map { column in rows.map { $0[column] } }
    return try PreparedNumericBatch(columnNames: names, columns: columns)
}

func matrixMLPInput(_ rows: [[Double]], width: Int,
                    order: NumericMatrixOrder = .rowMajor) throws -> PreparedNumericMatrix {
    try matrixMLPBatch(rows, width: width).matrix(order: order, missing: .reject)
}

func matrixMLPExpectClose(_ actual: [Double], _ expected: [Double]) {
    #expect(actual.count == expected.count)
    for (a, b) in zip(actual, expected) {
        #expect(a.isFinite && b.isFinite)
        #expect(abs(a - b) <= 1e-10 + 1e-12 * abs(b))
    }
}
