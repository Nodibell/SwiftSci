import Testing
import SwiftPreprocessing
import SwiftDataFrame

@Suite("Immutable fitted standard preprocessing")
struct StandardPreprocessingPlanTests {
    @Test func fitUsesTrainingOnlyAndKeepsMissingValueRules() throws {
        let training = try PreparedNumericBatch(columnNames: ["x", "constant", "absent"],
            columns: [[1, .nan, 3], [4, 4, 4], [.nan, .nan, .nan]])
        let input = try PreparedNumericBatch(columnNames: training.columnNames,
            columns: [[1000, .nan], [5, 4], [.nan, 2]])
        for strategy in [Imputer.Strategy.mean, .median, .mostFrequent, .constant(7)] {
            let plan = try StandardPreprocessingPlan(training: training, strategy: strategy)
            var imputer = Imputer(strategy: strategy)
            try imputer.fit(training)
            var scaler = StandardScaler()
            try scaler.fit(imputer.transform(training))
            let expected = try scaler.transform(imputer.transform(input)).matrix().values
            var actual = [Double](repeating: 0, count: expected.count)
            try actual.withUnsafeMutableBufferPointer { try plan.fillDouble(input, into: $0) }
            #expect(actual == expected)
            #expect(plan.columnNames == training.columnNames)
            var single = [Float](repeating: 0, count: expected.count)
            try single.withUnsafeMutableBufferPointer { try plan.fillFloat32(input, into: $0) }
            #expect(single == expected.map(Float.init))
            var half = [Float16](repeating: 0, count: expected.count)
            try half.withUnsafeMutableBufferPointer { try plan.fillFloat16(input, into: $0) }
            #expect(half == expected.map(Float16.init))
        }
    }

    @Test func rejectsEmptyTrainingAndMismatchedDestination() throws {
        let empty = try PreparedNumericBatch(columnNames: ["x"], columns: [[]])
        #expect(throws: PreprocessingError.self) { try StandardPreprocessingPlan(training: empty) }
        let input = try PreparedNumericBatch(columnNames: ["x"], columns: [[1, 2]])
        let plan = try StandardPreprocessingPlan(training: input)
        var destination = [Float](repeating: 0, count: 1)
        #expect(throws: SwiftMLError.self) {
            try destination.withUnsafeMutableBufferPointer { try plan.fillFloat32(input, into: $0) }
        }
        var zero = [Double]()
        try zero.withUnsafeMutableBufferPointer { try plan.fillDouble(empty, into: $0) }
        #expect(zero.isEmpty)
    }
}
