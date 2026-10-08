import Testing
import SwiftDataFrame
import SwiftPreprocessing

@Suite("Prepared scaler traversal contracts")
struct TiledScalerCompatibilityTests {
    @Test(arguments: [1, 3, 8, 31, 64, 129, 511, 512, 513, 2047, 2048, 2049])
    func exactValuesAcrossSizes(width: Int) throws {
        let names = (0..<width).map { "x\($0)" }
        let training = (0..<19).map { r in
            (0..<width).map { c in width > 1 && c == width - 1 ? 7.0 : Double((r * 17 + c * 13) % 101) / 7 - 9 }
        }
        var scaler = StandardScaler()
        try scaler.fit(training)
        let originalMean = scaler.mean, originalStd = scaler.std
        let edges = [max(1, 2048 / width), max(1, 8192 / width)]
        let counts = Set([0, 1, 3, 8, 31, 32, 33, 63, 127, 257] + edges.flatMap { [max(0, $0-1), $0, $0+1, 2*$0+3] })
        for rows in counts.sorted() {
            let columns = (0..<width).map { c in (0..<rows).map { r in
                width > 1 && c == width - 1 ? 7.0 : Double((r * 19 + c * 11) % 113) / 13 - 4
            } }
            let batch = try PreparedNumericBatch(columnNames: names, columns: columns)
            let expected = try scaler.transform(batch.rowValues())
            let actual = try scaler.transform(batch)
            #expect(actual.columnNames == names && actual.rowCount == rows)
            for c in 0..<width {
                #expect(actual.nullCount(inColumn: c) == 0)
                #expect((0..<rows).allSatisfy { actual[$0,c]!.bitPattern == expected[$0][c].bitPattern })
            }
        }
        #expect(scaler.mean == originalMean && scaler.std == originalStd)
    }

    @Test func singleColumnPreparedFitKeepsRowRounding() throws {
        let values = (0..<8).map { Double(($0 * 17) % 101) / 7 - 9 }
        let batch = try PreparedNumericBatch(columnNames: ["x"], columns: [values])
        var scaler = StandardScaler()
        try scaler.fit(batch)
        for count in [8, 64] {
            let input = try PreparedNumericBatch(columnNames: ["x"],
                columns: [(0..<count).map { values[$0 % 8] }])
            let expected = try scaler.transform(input.rowValues())
            let actual = try scaler.transform(input)
            #expect((0..<count).allSatisfy { actual[$0, 0]!.bitPattern == expected[$0][0].bitPattern })
        }
    }

    @Test func missingNonfiniteAndProvenance() throws {
        var source = try PreparedNumericBatch(columnNames: ["a", "b", "c"],
            columns: [(0..<6001).map(Double.init), [Double](repeating: 2, count: 6001), [Double](repeating: -0.0, count: 6001)])
        try source.updateColumn(at: 0, rows: [0, 2047, 2048, 4095, 6000], values: [nil, .nan, .infinity, -.infinity, 1e90])
        let selection = Array((0..<6001).reversed()) + [0, 2048, 2048]
        let snapshot = try source.selectingRows(selection)
        let rows = snapshot.rowValues()
        var scaler = StandardScaler()
        try scaler.fit([[1, 2, 0], [3, 2, 0], [5, 2, 0]])
        let expected = try scaler.transform(rows)
        let actual = try scaler.transform(snapshot)
        try source.updateColumn(at: 0, rows: [1], values: [999])
        #expect(actual.originalRowIndices == selection)
        for c in 0..<3 {
            #expect(actual.nullCount(inColumn: c) == 0)
            #expect((0..<actual.rowCount).allSatisfy { r in
                let value = actual[r,c]!, reference = expected[r][c]
                return value.bitPattern == reference.bitPattern || (value.isNaN && reference.isNaN)
            })
        }
        #expect(snapshot[5999,0] == 1)
    }

    @Test func concurrentTransformsKeepIndependentOutputs() async throws {
        let batch = try PreparedNumericBatch(columnNames: ["a", "b", "c"], columns:
            (0..<3).map { c in (0..<9001).map { Double(($0 * 7 + c * 3) % 97) } })
        var fitting = StandardScaler()
        try fitting.fit(batch)
        let scaler = fitting
        let expected = try scaler.transform(batch.rowValues()).map { $0.map(\.bitPattern) }
        try await withThrowingTaskGroup(of: Bool.self) { group in
            for _ in 0..<4 {
                group.addTask {
                    let actual = try scaler.transform(batch)
                    return actual.rowValues().map { $0.map(\.bitPattern) } == expected
                }
            }
            for try await equal in group { #expect(equal) }
        }
    }
}
