import Testing
import SwiftDataFrame
import SwiftPreprocessing

@Suite("Fused matrix preparation", .serialized)
struct StandardPreprocessingMatrixTests {
    @Test(arguments: [1, 7, 64, 128, 513, 2049])
    func matchesSeparateTransformsAcrossShapes(width: Int) throws {
        let names: [String] = (0..<width).map { c -> String in "x\(c)" }
        let trainingColumns: [[Double]] = preprocessingTrainingColumns(width: width)
        let training = try PreparedNumericBatch(columnNames: names, columns: trainingColumns)
        for strategy in [Imputer.Strategy.mean, .median, .mostFrequent, .constant(7)] {
            let plan = try StandardPreprocessingPlan(training: training, strategy: strategy)
            for rows in [0, 1, 31, 32, 257] {
                let columns: [[Double]] = preprocessingIncomingColumns(width: width, rows: rows)
                let input = try PreparedNumericBatch(columnNames: names, columns: columns)
                let expected = try plan.scaler.transform(plan.imputer.transform(input)).matrix()
                let actual = try plan.preparedMatrix(input)
                #expect(actual.values.map(\.bitPattern) == expected.values.map(\.bitPattern))
                #expect(actual.rowCount == rows)
                #expect(actual.columnNames == names)
                #expect(actual.originalRowIndices == expected.originalRowIndices)
            }
        }
    }

    @Test func preservesInputProvenanceAndIndependentMatrixSnapshots() throws {
        let training = try PreparedNumericBatch(columnNames: ["x", "constant", "absent"],
            columns: [[1, 2, 3, 4], [7, 7, 7, 7], [.nan, .nan, .nan, .nan]])
        let plan = try StandardPreprocessingPlan(training: training)
        let input = try training.selectingRows([3, 1, 1, 0])
        let sourceBits = input.columns.map { $0.values.map(\.bitPattern) }
        var matrix = try plan.preparedMatrix(input)
        let snapshot = matrix
        let bits = snapshot.values.map(\.bitPattern)
        matrix.withMutableValues { $0[0] = 42 }
        #expect(snapshot.values.map(\.bitPattern) == bits)
        #expect(input.columns.map { $0.values.map(\.bitPattern) } == sourceBits)
        #expect(snapshot.originalRowIndices == [3, 1, 1, 0])
        #expect(snapshot.columnNames == training.columnNames)
    }

    @Test func rejectsChangedFeatureOrder() throws {
        let training = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[1, 2], [3, 4]])
        let plan = try StandardPreprocessingPlan(training: training)
        let reordered = try PreparedNumericBatch(columnNames: ["y", "x"], columns: [[3, 4], [1, 2]])
        #expect(throws: SwiftMLError.self) { try plan.preparedMatrix(reordered) }
    }
}
