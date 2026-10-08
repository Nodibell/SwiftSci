import Testing
import SwiftDataFrame
import SwiftPreprocessing

@Suite("Prepared imputer ownership")
struct PreparedImputerTests {
    private func fixture() throws -> PreparedNumericBatch {
        var batch = try PreparedNumericBatch(columnNames: ["a", "b", "c"], columns:
            [[1, .nan, 3, 4], [10, 20, 30, 40], [.nan, .nan, .nan, .nan]])
        try batch.updateColumn(at: 0, rows: [3], values: [nil])
        return batch
    }
    private func addresses(_ batch: PreparedNumericBatch) -> [UInt] {
        batch.columns.map { $0.values.withUnsafeBufferPointer { UInt(bitPattern: $0.baseAddress!) } }
    }
    private func equal(_ a: [[Double]], _ b: [[Double]]) -> Bool {
        a.count == b.count && zip(a,b).allSatisfy { x,y in
            x.count == y.count && zip(x,y).allSatisfy { $0.bitPattern == $1.bitPattern || ($0.isNaN && $1.isNaN) }
        }
    }

    @Test func reusesUniqueColumnsThroughNestedPipeline() throws {
        var pipeline = Pipeline(steps: [Pipeline(steps: [Imputer(), MinMaxScaler()])])
        try pipeline.fit(fixture())
        #expect(pipeline.supportsNativePreparedBatches)
        let input = try PreparedNumericBatch(columnNames: ["a", "b", "c"], columns:
            (0..<3).map { c in (0..<65).map { r in r % 7 == 0 ? Double.nan : Double(r + c * 17) } })
        let expected = try pipeline.transform(input.rowValues())
        let before = addresses(input)
        let output = try pipeline.transform(consuming: consume input)
        #expect(addresses(output) == before)
        #expect(equal(output.rowValues(), expected))
    }

    @Test func preservesSharedInputsAndReusesCleanColumns() throws {
        var imputer = Imputer()
        try imputer.fit(fixture())
        let input = try fixture()
        let before = input.rowValues(), pointers = addresses(input)
        let output = try imputer.transform(consuming: input)
        #expect(equal(input.rowValues(), before))
        #expect(input[3,0] == nil && input.nullCount(inColumn: 0) == 1)
        #expect(output[3,0] == 2 && output.nullCount(inColumn: 0) == 0)
        #expect(addresses(output)[0] != pointers[0])
        #expect(addresses(output)[1] == pointers[1])
        withExtendedLifetime(input) {}
    }

    @Test func strategiesMatchRowsAndKeepProvenance() throws {
        let strategies: [Imputer.Strategy] = [.mean, .median, .mostFrequent, .constant(-9), .constant(.nan), .constant(.infinity)]
        for strategy in strategies {
            var row = Imputer(strategy: strategy), compact = Imputer(strategy: strategy)
            let training = try fixture()
            try row.fit(training.rowValues()); try compact.fit(training)
            #expect(equal([row.statistics!], [compact.statistics!]))
            var combined = Imputer(strategy: strategy)
            #expect(equal(try combined.fitTransform(training).rowValues(), try row.transform(training.rowValues())))
            let selected = try training.selectingRows([3,1,0,3])
            let expected = try row.transform(selected.rowValues())
            for output in [try compact.transform(selected), try compact.transform(consuming: selected)] {
                #expect(equal(output.rowValues(), expected))
                #expect(output.originalRowIndices == [3,1,0,3])
                #expect(output.columnNames == selected.columnNames)
                #expect((0..<3).allSatisfy { output.nullCount(inColumn: $0) == 0 })
            }
        }
    }

    @Test func errorsAndRefitKeepExistingState() throws {
        var imputer = Imputer()
        let input = try fixture()
        let empty = try PreparedNumericBatch(columnNames: ["different"], columns: [[]])
        #expect(throws: PreprocessingError.fitNotCalled) { try imputer.transform(consuming: empty) }
        try imputer.fit(input)
        let stats = imputer.statistics
        #expect(throws: PreprocessingError.emptyInput) { try imputer.fit(empty) }
        #expect(imputer.statistics == stats)
        #expect(throws: PreprocessingError.dimensionMismatch(expected: 2, got: 1)) { try imputer.fit([[1,2],[3]]) }
        #expect(imputer.statistics == stats)
        let mismatch = try PreparedNumericBatch(columnNames: ["x"], columns: [[1]])
        #expect(throws: PreprocessingError.dimensionMismatch(expected: 3, got: 1)) { try imputer.transform(consuming: mismatch) }
        let result = try imputer.transform(consuming: empty)
        #expect(result.rowCount == 0 && result.columnNames == ["different"])
    }

    @Test func retainedMatrixAndInternalAliasesRemainSnapshots() throws {
        let values: [Double] = [1, .nan, 3]
        let input = try PreparedNumericBatch(columnNames: ["a", "b"], columns: [values, values])
        var imputer = Imputer(); try imputer.fit([[0,10],[2,20]])
        let output = try imputer.transform(consuming: consume input)
        #expect(output.rowValues() == [[1,1],[1,15],[3,3]])
        #expect(values[1].isNaN)
        let single = try PreparedNumericBatch(columnNames: ["a"], columns: [values])
        let matrix = try single.matrix()
        var one = Imputer(strategy: .constant(7)); try one.fit([[0]])
        #expect(try one.transform(consuming: consume single)[1,0] == 7)
        #expect(matrix.values[1].isNaN)
        withExtendedLifetime((values,matrix)) {}
    }

    @Test func edgeStatisticsMatchRows() throws {
        let columns: [[Double]] = [[.nan,.nan,.nan,.nan], [2,1,2,1], [.infinity,-.infinity,1,.nan], [-1e16,1,1e16,.nan], [Double.leastNonzeroMagnitude,0,1,.nan]]
        let input = try PreparedNumericBatch(columnNames: ["a","b","c","d","e"], columns: columns)
        for strategy in [Imputer.Strategy.mean,.median,.mostFrequent,.constant(-0.0)] {
            var a = Imputer(strategy: strategy), b = Imputer(strategy: strategy)
            try a.fit(input.rowValues()); try b.fit(input)
            #expect(equal([a.statistics!],[b.statistics!]))
            #expect(equal(try a.transform(input.rowValues()),try b.transform(input).rowValues()))
        }
    }

    @Test func concurrentTransformsPreserveSource() async throws {
        let input = try fixture(); let before = input.rowValues()
        var fitting = Imputer(); try fitting.fit(input)
        let imputer = fitting
        let expected = try imputer.transform(before)
        try await withThrowingTaskGroup(of: Bool.self) { group in
            for _ in 0..<4 { group.addTask { equal(try imputer.transform(consuming: input).rowValues(),expected) } }
            for try await same in group { #expect(same) }
        }
        #expect(equal(input.rowValues(),before))
    }
}
