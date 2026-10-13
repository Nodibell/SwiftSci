import Testing
import SwiftDataFrame
import SwiftPreprocessing

@Suite("Matrix standard scaler")
struct MatrixScalerTests {
    @Test func matchesRowsAcrossShapesAndLayouts() throws {
        for width in [1, 7, 64, 513] {
            var scaler = StandardScaler()
            try scaler.fit(preprocessingRows(19, width: width))
            for count in [0, 1, 31, 32, 33, 65] {
                let rows = preprocessingRows(count, width: width, offset: 71, missing: true)
                let expected = try scaler.transform(rows)
                for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
                    let input = try preprocessingMatrix(rows, width: width, order: order)
                    let actual = try scaler.transform(input)
                    expectMatrixRows(actual, expected)
                    expectMatrixRows(input, rows)
                    #expect(actual.order == order && actual.columnNames == input.columnNames)
                }
            }
        }
    }

    @Test func reusesUniqueStorageAndPreservesSharedSnapshots() throws {
        var scaler = StandardScaler(); try scaler.fit([[1, 2], [3, 6]])
        for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
            let input = try preprocessingMatrix([[2, 4], [5, 8]], width: 2, order: order)
            let address = preprocessingAddress(input)
            let output = try scaler.transform(consuming: consume input)
            #expect(preprocessingAddress(output) == address)
            expectMatrixRows(output, [[0, 0], [3, 2]])
            let source = try preprocessingMatrix([[2, 4], [5, 8]], width: 2, order: order)
            let savedValues = source.values
            let copy = try scaler.transform(consuming: source)
            expectMatrixRows(source, [[2, 4], [5, 8]])
            expectMatrixRows(copy, [[0, 0], [3, 2]])
            #expect(source.values == savedValues)
            #expect(preprocessingAddress(copy) != preprocessingAddress(source))
            withExtendedLifetime((source, savedValues)) {}
        }
    }

    @Test func edgeArithmeticMatchesRows() throws {
        var scaler = StandardScaler(); try scaler.fit([[0, 3, -2], [0, 3, 2]])
        let rows: [[Double]] = [[-0.0, .infinity, .nan], [Double.leastNonzeroMagnitude, -.infinity, 1e300]]
        let expected = try scaler.transform(rows)
        for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
            expectMatrixRows(try scaler.transform(preprocessingMatrix(rows, width: 3, order: order)), expected)
        }
    }
}
