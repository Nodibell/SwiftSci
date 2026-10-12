import Testing
import SwiftDataFrame
import SwiftPreprocessing

@Suite("Matrix preprocessing validation and concurrency")
struct MatrixPreprocessingValidationTests {
    @Test func fittingEmptyAndWidthChecksMatchColumns() throws {
        var imputer = Imputer(), scaler = StandardScaler()
        let empty = try preprocessingMatrix([], width: 1)
        #expect(throws: PreprocessingError.fitNotCalled) { try imputer.transform(empty) }
        #expect(throws: PreprocessingError.fitNotCalled) { try scaler.transform(empty) }
        try imputer.fit([[0, 2], [2, 4]]); try scaler.fit([[0, 2], [2, 4]])
        #expect(try imputer.transform(empty).columnNames == ["x0"])
        #expect(try scaler.transform(empty).rowCount == 0)
        let mismatch = try preprocessingMatrix([[1]], width: 1)
        let error = PreprocessingError.dimensionMismatch(expected: 2, got: 1)
        #expect(throws: error) { try imputer.transform(mismatch) }
        #expect(throws: error) { try scaler.transform(mismatch) }
        let frame = try DataFrame(columns: [TypedColumn<Double>(name: "x", values: [1])])
        let zeroColumns = try frame.prepareNumericMatrix([])
        let zeroError = PreprocessingError.dimensionMismatch(expected: 2, got: 0)
        #expect(throws: zeroError) { try imputer.transform(zeroColumns) }
        #expect(throws: zeroError) { try scaler.transform(zeroColumns) }
    }

    @Test func concurrentSharedTransformsAndRefit() async throws {
        var imputer = Imputer(), scaler = StandardScaler()
        let training: [[Double]] = [[1, 2], [3, 6]]
        try imputer.fit(training); try scaler.fit(training)
        let fittedImputer = imputer, fittedScaler = scaler
        let rows: [[Double]] = [[.nan, 4], [3, .nan]]
        let input = try preprocessingMatrix(rows, width: 2)
        let expected = try scaler.transform(imputer.transform(rows))
        try await withThrowingTaskGroup(of: PreparedNumericMatrix.self) { group in
            for _ in 0..<4 {
                group.addTask {
                    let imputed = try fittedImputer.transform(input)
                    return try fittedScaler.transform(consuming: consume imputed)
                }
            }
            for try await output in group { expectMatrixRows(output, expected) }
        }
        expectMatrixRows(input, rows)
        try imputer.fit([[10, 20], [30, 40]])
        try scaler.fit([[10, 20], [30, 40]])
        let refitted = try scaler.transform(imputer.transform(input))
        expectMatrixRows(refitted, try scaler.transform(imputer.transform(rows)))
        #expect(refitted.values != expected.flatMap { $0 })
    }
}
