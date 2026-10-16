import Testing
import SwiftDataFrame
import SwiftPreprocessing

@Suite("Matrix scaler row alignment compatibility")
struct MatrixScalerRowBorrowTests {
    @Test(arguments: [1, 2, 3, 7, 8, 17, 31, 54, 64, 129, 511, 513, 2049])
    func exactValuesAcrossRowOffsets(width: Int) throws {
        var scaler = StandardScaler()
        try scaler.fit(preprocessingRows(19, width: width))
        let means = scaler.mean, deviations = scaler.std
        for count in [1, 2, 3, 31, 33, 257] {
            let rows = preprocessingRows(count, width: width, offset: 71, missing: true)
            let expected = try scaler.transform(rows)
            for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
                let source = try preprocessingMatrix(rows, width: width, order: order)
                let original = source.values.map(\.bitPattern)
                for result in [try scaler.transform(source), try scaler.transform(consuming: source)] {
                    #expect(result.order == order && result.columnNames == source.columnNames)
                    #expect(result.originalRowIndices == source.originalRowIndices)
                    #expect(rows.indices.allSatisfy { r in
                        rows[r].indices.allSatisfy { c in
                            let actual = result[r, c], reference = expected[r][c]
                            return actual.bitPattern == reference.bitPattern || (actual.isNaN && reference.isNaN)
                        }
                    })
                }
                #expect(source.values.map(\.bitPattern) == original)
            }
        }
        #expect(scaler.mean == means && scaler.std == deviations)
    }
}
