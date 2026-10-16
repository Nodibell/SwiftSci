import Testing
@testable import SwiftDataFrame

@Suite("Compact matrix packing boundaries")
struct CompactMatrixPackingTests {
    @Test(arguments: [1, 7, 64, 128, 513, 2049], [0, 1, 31, 32, 65])
    func layoutsPreserveBitsAndProvenance(width: Int, rows: Int) throws {
        let names: [String] = (0..<width).map { "x\($0)" }
        var columns: [[Double]] = []
        for c in 0..<width {
            let values: [Double] = (0..<rows).map { r -> Double in
                if (r + c) % 13 == 0 { return Double(bitPattern: 0x7ff8_0000_0000_0042) }
                if (r + c) % 11 == 0 { return -0.0 }
                return Double((r * 13 + c * 17) % 97 - 48) / 16
            }
            columns.append(values)
        }
        var original = try PreparedNumericBatch(columnNames: names, columns: columns)
        if rows > 0 { try original.updateColumn(at: 0, rows: [0], values: [nil]) }
        let selection: [Int] = rows == 0 ? [] : Array((0..<rows).reversed()) + [0]
        let batch = try original.selectingRows(selection)
        for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
            let matrix = try batch.matrix(order: order)
            #expect(matrix.columnNames == names)
            #expect(matrix.originalRowIndices == selection)
            #expect(matrix.rowCount == selection.count)
            var bitsMatch = true
            for r in selection.indices {
                for c in 0..<width {
                    let expected = original[selection[r], c] ?? Double.nan
                    if matrix[r, c].bitPattern != expected.bitPattern { bitsMatch = false }
                }
            }
            #expect(bitsMatch)
            if rows > 0 {
                #expect(throws: (any Error).self) { try batch.matrix(order: order, missing: .reject) }
            }
        }
    }

    @Test func exportedMatricesAndBatchCopiesRemainSnapshots() throws {
        let columns: [[Double]] = (0..<64).map { c -> [Double] in
            (0..<129).map { r -> Double in Double(r + c) }
        }
        var batch = try PreparedNumericBatch(columnNames: (0..<64).map { "x\($0)" }, columns: columns)
        let snapshot = batch
        let row = try batch.matrix()
        let column = try batch.matrix(order: .columnMajor)
        try batch.updateColumn(at: 0, rows: [0, 64, 128], values: [nil, .nan, -0.0])
        #expect(snapshot[64, 0] == 64)
        #expect(row[0, 0] == 0 && row[64, 0] == 64 && row[128, 0] == 128)
        #expect(column[0, 0] == 0 && column[64, 0] == 64 && column[128, 0] == 128)
        #expect(batch[0, 0] == nil && batch[64, 0]?.isNaN == true)
        #expect(batch[128, 0]?.bitPattern == (-0.0).bitPattern)
    }
}
