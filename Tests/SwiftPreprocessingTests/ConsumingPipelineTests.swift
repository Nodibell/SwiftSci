import Testing
import SwiftDataFrame
import SwiftPreprocessing

private enum RowMaterialization: Error { case forbidden }
private struct CompactIdentity: PreprocessingTransformer {
    var supportsNativePreparedBatches: Bool { true }
    mutating func fit(_ data: [[Double]]) throws { throw RowMaterialization.forbidden }
    func transform(_ data: [[Double]]) throws -> [[Double]] { throw RowMaterialization.forbidden }
    mutating func fit(_ data: PreparedNumericBatch) throws {}
    func transform(consuming data: consuming PreparedNumericBatch) throws -> PreparedNumericBatch { data }
}
private struct LegacyOffset: PreprocessingTransformer {
    mutating func fit(_ data: [[Double]]) throws {}
    func transform(_ data: [[Double]]) throws -> [[Double]] { data.map { $0.map { $0 + 1 } } }
}

@Suite("Consuming prepared pipeline")
struct ConsumingPipelineTests {
    private func fixture() throws -> PreparedNumericBatch {
        let first: [Double] = (0..<65).map(Double.init)
        let second: [Double] = (0..<65).map { rowIndex -> Double in
            let value: Int = (rowIndex * 7) % 19
            return Double(value)
        }
        let columns: [[Double]] = [first, second]
        return try PreparedNumericBatch(columnNames: ["a", "b"], columns: columns)
    }
    private func addresses(_ batch: PreparedNumericBatch) -> [UInt] {
        batch.columns.map { $0.values.withUnsafeBufferPointer { UInt(bitPattern: $0.baseAddress!) } }
    }

    @Test func nestedPipelineKeepsUniqueColumnStorage() throws {
        var pipeline = Pipeline(steps: [StandardScaler(), Pipeline(steps: [StandardScaler()])])
        try pipeline.fit(fixture())
        let input = try fixture()
        let original = addresses(input)
        let expected = try pipeline.transform(input.rowValues())
        let erased: any PreprocessingTransformer = pipeline
        let result = try erased.transform(consuming: consume input)
        #expect(addresses(result) == original)
        #expect(result.rowValues().map { $0.map(\.bitPattern) } == expected.map { $0.map(\.bitPattern) })
    }

    @Test func preparedOnlyStepsNeverMaterializeRows() throws {
        var pipeline = Pipeline(steps: [CompactIdentity(), Pipeline(steps: [CompactIdentity()])])
        let batch = try fixture()
        try pipeline.fit(batch)
        #expect(try pipeline.transform(consuming: batch).rowValues() == batch.rowValues())
    }

    @Test func legacyDefaultRetainsProvenanceAndSnapshots() throws {
        let source = try fixture().selectingRows([4, 1, 4])
        let erased: any PreprocessingTransformer = Pipeline(steps: [LegacyOffset(), LegacyOffset()])
        let result = try erased.transform(consuming: source)
        #expect(result.originalRowIndices == [4, 1, 4])
        #expect(result.columnNames == source.columnNames)
        #expect(result.rowValues() == source.rowValues().map { $0.map { $0 + 2 } })
        #expect(source[0,0] == 4)
    }

    @Test func emptyPipelineTransfersStorageAndEmptyInputs() throws {
        let input = try fixture()
        let before = addresses(input)
        let pipeline: any PreprocessingTransformer = Pipeline(steps: [])
        let output = try pipeline.transform(consuming: consume input)
        #expect(addresses(output) == before)
        let empty = try PreparedNumericBatch(columnNames: ["a"], columns: [[]])
        #expect(try pipeline.transform(consuming: empty).columnNames == ["a"])
        #expect(try pipeline.transform(consuming: empty).rowCount == 0)
    }

    @Test func concurrentSharedInputsKeepTheirValues() async throws {
        var fitting = Pipeline(steps: [StandardScaler(), Pipeline(steps: [StandardScaler()])])
        try fitting.fit(fixture())
        let pipeline = fitting, input = try fixture()
        let original = input.rowValues()
        let expected = try pipeline.transform(original)
        try await withThrowingTaskGroup(of: [[Double]].self) { group in
            for _ in 0..<4 { group.addTask { try pipeline.transform(consuming: input).rowValues() } }
            for try await result in group { #expect(result == expected) }
        }
        #expect(input.rowValues() == original)
    }


    @Test(arguments: [false, true])
    func legacyShapeChangingIntermediatesRemainSupported(nested: Bool) throws {
        let legacy = Pipeline(steps: [PolynomialFeatures(degree: 1, includeBias: true), VarianceThreshold()])
        var pipeline = nested ? Pipeline(steps: [legacy]) : legacy
        let batch = try fixture().selectingRows([4, 1, 4, 7])
        var reference = pipeline
        try reference.fit(batch.rowValues())
        try pipeline.fit(batch)
        let expected = try reference.transform(batch.rowValues())
        for actual in [try pipeline.transform(batch), try pipeline.transform(consuming: batch)] {
            #expect(actual.rowValues() == expected)
            #expect(actual.columnNames == batch.columnNames && actual.originalRowIndices == [4, 1, 4, 7])
        }
    }

    @Test func emptyPipelineKeepsLegacyNumericMissingOutput() throws {
        var input = try fixture()
        try input.updateColumn(at: 0, rows: [1], values: [nil])
        let pipeline = Pipeline(steps: [])
        for result in [try pipeline.transform(input), try pipeline.transform(consuming: input)] {
            #expect(result[1,0]?.isNaN == true)
            #expect(result.nullCount(inColumn: 0) == 0)
        }
        #expect(input[1,0] == nil)
    }

}
