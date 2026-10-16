import Testing
import SwiftDataFrame
import SwiftPreprocessing

@Suite("Bounded preprocessing row writer", .serialized)
struct StandardPreprocessingRowWriterTests {
    @Test(arguments: [1, 7, 64, 128, 513, 2049])
    func rangesAndPadding(width: Int) throws {
        let names = (0..<width).map { "x\($0)" }
        let columns: [[Double]] = (0..<width).map { c in
            (0..<65).map { r in (r + c) % 13 == 0 ? .nan : Double(r - c) / 16 }
        }
        let input = try PreparedNumericBatch(columnNames: names, columns: columns)
        let plan = try StandardPreprocessingPlan(training: input)
        let expected = try plan.scaler.transform(plan.imputer.transform(input)).matrix()
        let bits = input.columns.map { $0.values.map(\.bitPattern) }
        var writer = try plan.makeRowWriter(input, maximumRows: 32)
        let stride = width + 5
        var buffer = [Double](repeating: -999, count: 32 * stride)
        for range in [0..<32, 17..<49, 64..<65, 32..<32] {
            buffer = Array(repeating: -999, count: buffer.count)
            try buffer.withUnsafeMutableBufferPointer {
                try writer.fillDouble(rows: range, into: $0, stride: stride)
            }
            for r in 0..<32 {
                for c in 0..<stride {
                    let want = r < range.count && c < width
                        ? expected.values[(range.lowerBound + r) * width + c] : -999
                    #expect(buffer[r * stride + c].bitPattern == want.bitPattern)
                }
            }
        }
        #expect(input.columns.map { $0.values.map(\.bitPattern) } == bits)
    }

    @Test func rejectsInvalidRequestsBeforeWriting() throws {
        let input = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[1, 2, 3], [4, 5, 6]])
        let plan = try StandardPreprocessingPlan(training: input)
        var writer = try plan.makeRowWriter(input, maximumRows: 2)
        for (range, stride, size) in [(-1..<1, 2, 4), (2..<4, 2, 4), (0..<3, 2, 6),
                                    (0..<2, 1, 4), (0..<2, 2, 3), (0..<2, Int.max, 4)] {
            var buffer = Array(repeating: -999.0, count: size)
            #expect(throws: SwiftMLError.self) {
                try buffer.withUnsafeMutableBufferPointer {
                    try writer.fillDouble(rows: range, into: $0, stride: stride)
                }
            }
            #expect(buffer.allSatisfy { $0 == -999 })
        }
        #expect(throws: SwiftMLError.self) { try plan.makeRowWriter(input, maximumRows: -1) }
    }
}
