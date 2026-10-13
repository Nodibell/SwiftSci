import Testing
import SwiftDataFrame
import SwiftPreprocessing

@Suite("Matrix scaler nonfinite fitted statistics")
struct MatrixScalerNonfiniteTests {
    @Test func matchesRowsWithNonfiniteFittedStatistics() throws {
        let training: [[Double]] = [[.nan, .infinity, -1e300, 1], [2, 3, 1e300, 3]]
        var scaler = StandardScaler()
        try scaler.fit(training)
        let means = try #require(scaler.mean)
        let deviations = try #require(scaler.std)
        #expect(means[0].isNaN && means[1].isInfinite)
        #expect(deviations[0].isNaN && deviations[1].isNaN && deviations[2].isInfinite)
        let rows: [[Double]] = [[0, 1, -1, 2], [.infinity, .nan, 1e300, 5]]
        let expected = try scaler.transform(rows)
        for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
            let input = try preprocessingMatrix(rows, width: 4, order: order)
            expectMatrixRows(try scaler.transform(input), expected)
            let consumed = try scaler.transform(consuming: input)
            expectMatrixRows(consumed, expected)
            expectMatrixRows(input, rows)
            #expect(consumed.order == order && consumed.columnNames == input.columnNames)
        }
    }
}
