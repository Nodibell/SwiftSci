import Testing
import SwiftDataFrame
import SwiftPreprocessing

@Suite("Indexed preprocessing", .serialized)
struct IndexedPreprocessingTests {
    @Test(arguments: [1, 7, 64, 128, 512, 513, 2049])
    func normalizationMatchesGatheredLegacyTransforms(width: Int) throws {
        let names = (0..<width).map { "x\($0)" }
        let columns: [[Double]] = (0..<width).map { c in
            (0..<97).map { r in (r + c) % 13 == 0 ? .nan : Double((r * 13 + c) % 47) / 8 }
        }
        let source = try PreparedNumericBatch(columnNames: names, columns: columns)
        for strategy in [Imputer.Strategy.mean, .median, .mostFrequent, .constant(7)] {
            let plan = try StandardPreprocessingPlan(training: source, strategy: strategy)
            for count in [0, 1, 31, 32, 33, 65] {
                let indices = (0..<count).map { ($0 * 17) % 43 }
                var view = try IndexedNumericBatch(source, selectingRows: indices)
                var gathered = try source.selectingRows(indices)
                if count > 0 {
                    let rows = [0, count / 2, count - 1, count / 2]
                    let values: [Double?] = [nil, .infinity, -0.0, .nan]
                    try view.updateColumn(at: width - 1, rows: rows, values: values)
                    try gathered.updateColumn(at: width - 1, rows: rows, values: values)
                }
                let expected = try plan.scaler.transform(plan.imputer.transform(gathered)).matrix()
                let actual = try plan.preparedMatrix(view)
                #expect(actual.values.map(\.bitPattern) == expected.values.map(\.bitPattern))
                #expect(actual.originalRowIndices == expected.originalRowIndices)
            }
        }
    }

    @Test func paddingAndInvalidRanges() throws {
        let source = try PreparedNumericBatch(columnNames: ["x", "y"], columns:
            [(0..<65).map(Double.init), (0..<65).map { Double($0 + 3) }])
        let view = try IndexedNumericBatch(source, selectingRows: Array((0..<65).reversed()))
        let plan = try StandardPreprocessingPlan(training: source)
        let expected = try plan.preparedMatrix(view).values
        var writer = try plan.makeRowWriter(view, maximumRows: 32)
        var output = Array(repeating: -999.0, count: 32 * 7)
        try output.withUnsafeMutableBufferPointer { try writer.fillDouble(rows: 1..<33, into: $0, stride: 7) }
        for r in 0..<32 {
            #expect(output[r * 7].bitPattern == expected[(r + 1) * 2].bitPattern)
            #expect(output[r * 7 + 1].bitPattern == expected[(r + 1) * 2 + 1].bitPattern)
            #expect(output[(r * 7 + 2)..<(r * 7 + 7)].allSatisfy { $0 == -999 })
        }
        for (rows, stride) in [(Int.min..<Int.max, 7), (0..<33, 7), (64..<66, 7), (0..<2, 1), (0..<2, Int.max)] {
            output = Array(repeating: -999, count: output.count)
            #expect(throws: SwiftMLError.self) {
                try output.withUnsafeMutableBufferPointer { try writer.fillDouble(rows: rows, into: $0, stride: stride) }
            }
            #expect(output.allSatisfy { $0 == -999 })
        }
    }
}
