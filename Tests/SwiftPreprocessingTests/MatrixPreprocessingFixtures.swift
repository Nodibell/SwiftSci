import Testing
import SwiftDataFrame

func preprocessingRows(_ count: Int, width: Int, offset: Int = 0, missing: Bool = false) -> [[Double]] {
    var rows: [[Double]] = []
    for r in 0..<count {
        var row: [Double] = []
        for c in 0..<width {
            let value: Int = ((r + offset) * 13 + c * 7) % 37 - 18
            row.append(missing && (r + c) % 7 == 0 ? .nan : Double(value) / 11)
        }
        rows.append(row)
    }
    return rows
}

func preprocessingMatrix(_ rows: [[Double]], width: Int,
                         order: NumericMatrixOrder = .rowMajor) throws -> PreparedNumericMatrix {
    let columns: [TypedColumn<Double>] = (0..<width).map { c in
        let values: [Double] = rows.map { $0[c] }
        return TypedColumn<Double>(name: "x\(c)", values: values)
    }
    let frame = try DataFrame(columns: columns)
    let names: [String] = (0..<width).map { "x\($0)" }
    return try frame.prepareNumericMatrix(names, order: order)
}

func expectMatrixRows(_ matrix: PreparedNumericMatrix, _ rows: [[Double]]) {
    #expect(matrix.rowCount == rows.count)
    for r in rows.indices {
        #expect(matrix.columnCount == rows[r].count)
        for c in rows[r].indices {
            let actual = matrix[r, c], expected = rows[r][c]
            #expect(actual.bitPattern == expected.bitPattern || (actual.isNaN && expected.isNaN))
        }
    }
}

func preprocessingAddress(_ matrix: borrowing PreparedNumericMatrix) -> UInt {
    matrix.values.withUnsafeBufferPointer { UInt(bitPattern: $0.baseAddress!) }
}
