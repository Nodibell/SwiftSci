import Testing
import Accelerate
import SwiftDataFrame
import SwiftPreprocessing

@Suite("Consuming prepared MinMax scaler")
struct ConsumingMinMaxScalerTests {
    private func fixture(rows: Int, width: Int) throws -> PreparedNumericBatch {
        try PreparedNumericBatch(columnNames: (0..<width).map { "x\($0)" }, columns:
            (0..<width).map { c in (0..<rows).map { Double(($0 * 19 + c * 11) % 113) / 13 - 4 } })
    }

    private func addresses(_ batch: PreparedNumericBatch) -> [UInt] {
        batch.columns.map { $0.values.withUnsafeBufferPointer { UInt(bitPattern: $0.baseAddress!) } }
    }

    @Test(arguments: [17, 513], [3, 65])
    func uniqueColumnsReuseStorage(width: Int, rows: Int) throws {
        var scaler = MinMaxScaler(range: (-2, 3))
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
        var scaler = MinMaxScaler(range: (-2, 3))
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
        var scaler = MinMaxScaler(range: (-2, 3))
        try scaler.fit([[0, 20], [2, 40], [4, 60]])
        let expected = try scaler.transform(input().rowValues()).map { $0.map(\.bitPattern) }
        let actual = try scaler.transform(consuming: input())
        #expect(actual.rowValues().map { $0.map(\.bitPattern) } == expected)
    }

    @Test func retainedMatrixKeepsItsSnapshot() throws {
        let input = try fixture(rows: 65, width: 1)
        let matrix = try input.matrix()
        let before = input.rowValues()
        var scaler = MinMaxScaler(range: (-2, 3))
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
        var scaler = MinMaxScaler(range: (-2, 3))
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

    // The previous prepared implementation, retained as an independent oracle.
    private func allocatingReference(_ input: PreparedNumericBatch, scaler: MinMaxScaler) -> [[Double]] {
        let count = vDSP_Length(input.rowCount)
        return input.columns.enumerated().map { c, column in
            var shifted = [Double](repeating: 0, count: input.rowCount)
            var scaled = shifted, output = shifted
            var negative = -scaler.dataMin![c]
            let span = scaler.dataMax![c] - scaler.dataMin![c]
            var scale = span < 1e-12 ? 0 : (scaler.range.max - scaler.range.min) / span
            var offset = scaler.range.min
            vDSP_vsaddD(column.values, 1, &negative, &shifted, 1, count)
            vDSP_vsmulD(shifted, 1, &scale, &scaled, 1, count)
            vDSP_vsaddD(scaled, 1, &offset, &output, 1, count)
            return output
        }
    }

    @Test(arguments: [1, 3, 7, 8, 9, 31, 32, 33, 129, 513, 2049, 8193])
    func preservesExistingArithmetic(rows: Int) throws {
        for range in [(0.0, 1.0), (-2.0, 3.0), (3.0, -2.0), (-0.0, -0.0)] {
            var scaler = MinMaxScaler(range: range)
            try scaler.fit([[0, 0, 0, 0, 7], [1e-12.nextDown, 1e-12, 1e-12.nextUp, 3, 7]])
            let values: [Double] = [-0.0, 0, .leastNonzeroMagnitude, -.leastNormalMagnitude, 1e-12, -1e100, 1e100, .greatestFiniteMagnitude, .infinity, -.infinity, .nan]
            let input = try PreparedNumericBatch(columnNames: ["a", "b", "c", "d", "e"],
                columns: (0..<5).map { c in (0..<rows).map { values[($0 + c) % values.count] } })
            let expected = allocatingReference(input, scaler: scaler)
            let actual = try scaler.transform(consuming: consume input)
            for c in 0..<5 {
                #expect((0..<rows).allSatisfy { r in
                    let a = actual[r,c]!, b = expected[c][r]
                    return a.bitPattern == b.bitPattern || (a.isNaN && b.isNaN)
                })
            }
        }
    }

    @Test func nestedPipelineUsesConsumingWitness() throws {
        var pipeline = Pipeline(steps: [Pipeline(steps: [MinMaxScaler(range: (-2, 3))])])
        try pipeline.fit(fixture(rows: 19, width: 17))
        let input = try fixture(rows: 65, width: 17)
        let before = addresses(input)
        let actual = try pipeline.transform(consuming: consume input)
        #expect(addresses(actual) == before)
    }
    @Test func missingNonfiniteAndProvenance() throws {
        var source = try PreparedNumericBatch(columnNames: ["a", "b", "c"],
            columns: [(0..<6001).map(Double.init), [Double](repeating: 2, count: 6001), [Double](repeating: -0.0, count: 6001)])
        try source.updateColumn(at: 0, rows: [0, 2047, 2048, 4095, 6000], values: [nil, .nan, .infinity, -.infinity, 1e90])
        let selection = Array((0..<6001).reversed()) + [0, 2048, 2048]
        let snapshot = try source.selectingRows(selection)
        let rows = snapshot.rowValues()
        var scaler = MinMaxScaler(range: (-2, 3))
        try scaler.fit([[1, 2, 0], [3, 2, 0], [5, 2, 0]])
        let expected = try scaler.transform(rows)
        let actual = try scaler.transform(snapshot)
        let consumed = try scaler.transform(consuming: snapshot)
        try source.updateColumn(at: 0, rows: [1], values: [999])
        for actual in [actual, consumed] {
            #expect(actual.originalRowIndices == selection)
            for c in 0..<3 {
                #expect(actual.nullCount(inColumn: c) == 0)
                #expect((0..<actual.rowCount).allSatisfy { r in
                    let value = actual[r,c]!, reference = expected[r][c]
                    return value.bitPattern == reference.bitPattern || (value.isNaN && reference.isNaN)
                })
            }
        }
        #expect(snapshot[5999,0] == 1)
        #expect(snapshot[6000,0] == nil && snapshot.nullCount(inColumn: 0) == 2)
    }

    @Test func concurrentTransformsKeepIndependentOutputs() async throws {
        let batch = try PreparedNumericBatch(columnNames: ["a", "b", "c"], columns:
            (0..<3).map { c in (0..<9001).map { Double(($0 * 7 + c * 3) % 97) } })
        var fitting = MinMaxScaler(range: (-2, 3))
        try fitting.fit(batch)
        let scaler = fitting
        let expected = try scaler.transform(batch.rowValues()).map { $0.map(\.bitPattern) }
        try await withThrowingTaskGroup(of: Bool.self) { group in
            for index in 0..<4 {
                group.addTask {
                    let actual = try index.isMultiple(of: 2)
                        ? scaler.transform(batch) : scaler.transform(consuming: batch)
                    return actual.rowValues().map { $0.map(\.bitPattern) } == expected
                }
            }
            for try await equal in group { #expect(equal) }
        }
    }
}
