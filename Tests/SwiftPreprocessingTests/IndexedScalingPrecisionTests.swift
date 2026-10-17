import Testing
import SwiftDataFrame
import SwiftPreprocessing

@Suite("Indexed scaling arithmetic and tails")
struct IndexedScalingPrecisionTests {
    @Test func wideDynamicRangeOffsetAndPaddingMatchLegacyBits() throws {
        let source = try indexedScalingPrecisionSource()
        let plan = try StandardPreprocessingPlan(training: source)
        let indices: [Int] = (0..<65).map { ($0 / 3 * 17 + $0 % 2) % 97 }
        var view = try IndexedNumericBatch(source, selectingRows: indices)
        var gathered = try source.selectingRows(indices)
        let edits: [Double?] = [nil, .nan, -0.0, 0x1p-390]
        try view.updateColumn(at: 0, rows: [0, 3, 8, 32], values: edits)
        try gathered.updateColumn(at: 0, rows: [0, 3, 8, 32], values: edits)
        let expected = try plan.scaler.transform(plan.imputer.transform(gathered)).matrix().values
        var writer = try plan.makeRowWriter(view, maximumRows: 33)
        for count in [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33] {
            let width = 9, stride = 14, offset = 1
            var output = [Double](repeating: -999, count: 33 * stride)
            try output.withUnsafeMutableBufferPointer {
                try writer.fillDouble(rows: offset..<(offset + count), into: $0, stride: stride)
            }
            for row in 0..<33 {
                for column in 0..<stride {
                    let want = row < count && column < width
                        ? expected[(row + offset) * width + column] : -999
                    #expect(output[row * stride + column].bitPattern == want.bitPattern)
                }
            }
        }
    }
}
