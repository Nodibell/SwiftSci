import Testing
import SwiftDataFrame
import SwiftPreprocessing

@Suite("Matrix imputer")
struct MatrixImputerTests {
    @Test func strategiesMatchFittedRows() throws {
        let training: [[Double]] = [[1, 4, .nan], [3, 2, .nan], [3, 2, .nan]]
        let heldOut: [[Double]] = [[.nan, -0.0, .nan], [.infinity, .nan, 7], [9, .nan, .nan]]
        let strategies: [Imputer.Strategy] = [.mean, .median, .mostFrequent, .constant(-9), .constant(.nan), .constant(.infinity)]
        for strategy in strategies {
            var imputer = Imputer(strategy: strategy)
            try imputer.fit(training)
            let expected = try imputer.transform(heldOut)
            for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
                let input = try preprocessingMatrix(heldOut, width: 3, order: order)
                let actual = try imputer.transform(input)
                expectMatrixRows(actual, expected)
                expectMatrixRows(input, heldOut)
                #expect(actual.columnNames == input.columnNames && actual.order == order)
            }
        }
    }

    @Test func reusesUniqueAndCleanSharedStorage() throws {
        var imputer = Imputer(strategy: .constant(7)); try imputer.fit([[1, 2]])
        for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
            let input = try preprocessingMatrix([[.nan, 2], [3, .nan]], width: 2, order: order)
            let address = preprocessingAddress(input)
            let output = try imputer.transform(consuming: consume input)
            #expect(preprocessingAddress(output) == address)
            expectMatrixRows(output, [[7, 2], [3, 7]])
            let clean = try preprocessingMatrix([[1, 2]], width: 2, order: order)
            let shared = try imputer.transform(clean)
            #expect(preprocessingAddress(shared) == preprocessingAddress(clean))
            withExtendedLifetime(clean) {}
        }
    }

    @Test func detachesSharedBufferAndRetainsProvenance() throws {
        let values: [Double?] = [nil, 2, .nan]
        let frame = try DataFrame(columns: [TypedColumn<Double>(name: "x", values: values)])
        let input = try frame.prepareNumericMatrix(["x"], rows: [2, 0, 2, 1])
        let retainedValues = input.values
        var imputer = Imputer(strategy: .constant(8)); try imputer.fit([[1]])
        let output = try imputer.transform(consuming: input)
        #expect(output.values == [8, 8, 8, 2])
        #expect(output.originalRowIndices == [2, 0, 2, 1])
        #expect(input.values[0].isNaN && retainedValues[1].isNaN)
        #expect(preprocessingAddress(input) != preprocessingAddress(output))
        withExtendedLifetime((input, retainedValues)) {}
    }
}
