import Testing
import SwiftDataFrame
import SwiftPreprocessing

@Suite("Scoped preprocessing workspace", .serialized)
struct BufferScopeTests {
    @Test func repeatedFillsPreserveStridedOutput() throws {
        let input = try PreparedNumericBatch(columnNames: ["x"], columns: [[1, 2, .nan, -0.0]])
        let plan = try StandardPreprocessingPlan(training: input)
        let expected = try plan.preparedMatrix(input).values
        var writer = try plan.makeRowWriter(input, maximumRows: 2)
        var values = [Double](repeating: -999, count: 4)
        for start in [0, 2] {
            try values.withUnsafeMutableBufferPointer {
                try writer.fillDouble(rows: start..<(start + 2), into: $0, stride: 2)
            }
            #expect(values[0].bitPattern == expected[start].bitPattern)
            #expect(values[2].bitPattern == expected[start + 1].bitPattern)
            #expect(values[1] == -999 && values[3] == -999)
        }
    }
    @Test func extremeInvalidRangeDoesNotWrite() throws {
        let input = try PreparedNumericBatch(columnNames: ["x"], columns: [[1, 2]])
        let plan = try StandardPreprocessingPlan(training: input)
        var writer = try plan.makeRowWriter(input, maximumRows: 2)
        var values = [-999.0]
        #expect(throws: SwiftMLError.self) {
            try values.withUnsafeMutableBufferPointer {
                try writer.fillDouble(rows: Int.min..<Int.max, into: $0, stride: 1)
            }
        }
        #expect(values == [-999])
    }
}
