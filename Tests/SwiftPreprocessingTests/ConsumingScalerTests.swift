import Testing
import SwiftDataFrame
import SwiftPreprocessing

@Suite("Consuming prepared scaler")
struct ConsumingScalerTests {
    private func fixture(rows: Int, width: Int) throws -> PreparedNumericBatch {
        try makeConsumingScalerFixture(rows: rows, width: width)
    }

    private func addresses(_ batch: PreparedNumericBatch) -> [UInt] {
        batch.columns.map { $0.values.withUnsafeBufferPointer { UInt(bitPattern: $0.baseAddress!) } }
    }

    @Test(arguments: [17, 513], [3, 65])
    func uniqueColumnsReuseStorage(width: Int, rows: Int) throws {
        var scaler = StandardScaler()
        try scaler.fit(fixture(rows: 19, width: width))
        let input = try fixture(rows: rows, width: width)
        let before = addresses(input)
        let expected = try scaler.transform(input.rowValues()).map { $0.map(\.bitPattern) }
        let actual = try scaler.transform(consuming: consume input)
        #expect(addresses(actual) == before)
        #expect(actual.rowValues().map { $0.map(\.bitPattern) } == expected)
    }

    @Test func sharedColumnsAndExternalArraysKeepTheirValues() throws {
        let values = (0..<65).map(Double.init)
        let input = try PreparedNumericBatch(columnNames: ["a", "b"], columns: [values, values])
        var scaler = StandardScaler()
        try scaler.fit([[0, 20], [2, 40], [4, 60]])
        let expected = try scaler.transform(input.rowValues()).map { $0.map(\.bitPattern) }
        let output = try scaler.transform(consuming: input)
        #expect(output.rowValues().map { $0.map(\.bitPattern) } == expected)
        #expect(input.columnValues(at: 0) == values.map(Optional.some))
        #expect(input.columnValues(at: 1) == values.map(Optional.some))
        #expect(zip(addresses(input), addresses(output)).allSatisfy { $0 != $1 })
        #expect(values == (0..<65).map(Double.init))
        withExtendedLifetime((input, values)) {}
    }

    @Test func uniqueBatchWithInternallySharedColumns() throws {
        func input() throws -> PreparedNumericBatch {
            let values = (0..<65).map(Double.init)
            return try PreparedNumericBatch(columnNames: ["a", "b"], columns: [values, values])
        }
        var scaler = StandardScaler()
        try scaler.fit([[0, 20], [2, 40], [4, 60]])
        let expected = try scaler.transform(input().rowValues()).map { $0.map(\.bitPattern) }
        let actual = try scaler.transform(consuming: input())
        #expect(actual.rowValues().map { $0.map(\.bitPattern) } == expected)
    }

    @Test func retainedMatrixKeepsItsSnapshot() throws {
        let input = try fixture(rows: 65, width: 1)
        let matrix = try input.matrix()
        let before = input.rowValues()
        var scaler = StandardScaler()
        try scaler.fit([[0], [2], [4]])
        let output = try scaler.transform(consuming: consume input)
        #expect(matrix.values == before.map { $0[0] })
        #expect(output.rowValues() == (try scaler.transform(before)))
        withExtendedLifetime(matrix) {}
    }

    @Test func errorsAndEmptyInputsKeepExistingContracts() throws {
        let input = try fixture(rows: 3, width: 2)
        let original = input.rowValues()
        let empty = try fixture(rows: 0, width: 2)
        var scaler = StandardScaler()
        #expect(throws: PreprocessingError.fitNotCalled) { try scaler.transform(consuming: input) }
        #expect(throws: PreprocessingError.fitNotCalled) { try scaler.transform(consuming: empty) }
        try scaler.fit([[0], [2], [4]])
        #expect(throws: PreprocessingError.dimensionMismatch(expected: 1, got: 2)) {
            try scaler.transform(consuming: input)
        }
        #expect(input.rowValues() == original)
        let output = try scaler.transform(consuming: empty)
        #expect(output.rowCount == 0 && output.columnNames == empty.columnNames)
        #expect(output.originalRowIndices.isEmpty)
        #expect(output.nullCount(inColumn: 0) == 0 && output.nullCount(inColumn: 1) == 0)
    }

}
