import Testing
import SwiftDataFrame

@Suite("Matrix extraction across strided destinations")
struct MatrixExtractionBorrowTests {
    @Test(arguments: [1, 7, 54, 129])
    func matchesColumnPreparation(width: Int) throws {
        for count in [0, 1, 33, 257] {
            let frame = try extractionFrame(rows: count, width: width)
            var names = (0..<width).map { "x\($0)" }
            names.reverse()
            names.append(names[0])
            let repeated: [Int] = count == 0 ? [] : [count - 1, 0, count / 2, count - 1]
            let selections: [[Int]?] = [nil, [], repeated, Array(0..<count), Array((0..<count).reversed())]
            for selection in selections {
                var batch = try frame.prepareNumericBatch(names)
                if let selection { batch = try batch.selectingRows(selection) }
                for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
                    let actual = try frame.prepareNumericMatrix(names, rows: selection, order: order)
                    let expected = try batch.matrix(order: order)
                    #expect(actual.values.map(\.bitPattern) == expected.values.map(\.bitPattern))
                    #expect(actual.columnNames == expected.columnNames)
                    #expect(actual.rowCount == expected.rowCount && actual.order == order)
                    #expect(actual.originalRowIndices == expected.originalRowIndices)
                }
            }
        }
    }
}

private func extractionFrame(rows: Int, width: Int) throws -> DataFrame {
    var columns: [any AnyColumn] = []
    for column in 0..<width {
        let name = "x\(column)"
        switch column % 3 {
        case 0:
            let values: [Double?] = (0..<rows).map { row -> Double? in
                if row % 17 == 0 { return nil }
                if row % 13 == 0 { return Double(bitPattern: 0x7ff800000000002a) }
                let difference: Int = row - column
                let numeric: Double = Double(difference) / 7
                return row % 11 == 0 ? -0.0 : numeric
            }
            columns.append(TypedColumn<Double>(name: name, values: values))
        case 1:
            let values: [Int64?] = (0..<rows).map { row -> Int64? in
                let offset: Int64 = Int64(row + column)
                let numeric: Int64 = Int64.max - offset
                return row % 19 == 0 ? nil : numeric
            }
            columns.append(TypedColumn<Int64>(name: name, values: values))
        default:
            let values: [Bool?] = (0..<rows).map { row -> Bool? in
                row % 23 == 0 ? nil : row % 2 == 0
            }
            columns.append(TypedColumn<Bool>(name: name, values: values))
        }
    }
    return try DataFrame(columns: columns)
}
