import Testing
import SwiftML
import SwiftPreprocessing

private enum UnexpectedBorrow: Error { case called }
private struct ConsumingIdentity: PreprocessingTransformer {
    var supportsNativePreparedBatches: Bool { true }
    mutating func fit(_ data: [[Double]]) throws { throw UnexpectedBorrow.called }
    func transform(_ data: [[Double]]) throws -> [[Double]] { throw UnexpectedBorrow.called }
    mutating func fit(_ data: PreparedNumericBatch) throws {}
    func transform(_ data: PreparedNumericBatch) throws -> PreparedNumericBatch { throw UnexpectedBorrow.called }
    func transform(consuming data: consuming PreparedNumericBatch) throws -> PreparedNumericBatch { data }
}
private actor EchoRegressor: RegressorEstimator {
    let wrongCount: Bool
    var expected: UInt?
    init(wrongCount: Bool = false, expected: UInt? = nil) { self.wrongCount = wrongCount; self.expected = expected }
    func expectAddress(_ address: UInt) { expected = address }
    func fit(features: [[Double]], targets: [Double]) async throws { throw UnexpectedBorrow.called }
    func predict(features: [[Double]]) async throws -> [Double] { throw UnexpectedBorrow.called }
    func fit(features: PreparedNumericBatch, targets: [Double]) async throws { check(features) }
    func predict(features: PreparedNumericBatch) async throws -> [Double] {
        check(features)
        return wrongCount ? [] : (0..<features.rowCount).map { features[$0,0]! }
    }
    private func check(_ features: PreparedNumericBatch) {
        if let expected {
            #expect(features.columns[0].values.withUnsafeBufferPointer { UInt(bitPattern: $0.baseAddress!) } == expected)
        }
    }
}

@Suite("Consuming regression pipeline")
struct ConsumingRegressionPipelineTests {
    private func fixture() throws -> PreparedNumericBatch {
        try PreparedNumericBatch(columnNames: ["x"], columns: [(0..<65).map(Double.init)])
    }
    private func address(_ batch: PreparedNumericBatch) -> UInt {
        batch.columns[0].values.withUnsafeBufferPointer { UInt(bitPattern: $0.baseAddress!) }
    }

    @Test func uniqueTrainingAndPredictionReachEstimatorWithoutColumnCopies() async throws {
        let training = try fixture(), trainingAddress = address(training)
        let estimator = EchoRegressor(expected: trainingAddress)
        let pipeline = RegressionPipeline(transformers: [Pipeline(steps: [StandardScaler(), ConsumingIdentity()])], estimator: estimator)
        try await pipeline.fit(consuming: consume training, targets: Array(repeating: 0, count: 65))
        let input = try fixture(), inputAddress = address(input)
        await estimator.expectAddress(inputAddress)
        let prediction = try await pipeline.predict(consuming: consume input)
        #expect(prediction.count == 65 && abs(prediction.reduce(0,+)) < 1e-10)
    }

    @Test func sharedInputsAndValidationRemainCompatible() async throws {
        let pipeline = RegressionPipeline(transformers: [StandardScaler()], estimator: EchoRegressor())
        let input = try fixture(), original = input.rowValues()
        try await pipeline.fit(features: input, targets: Array(repeating: 0, count: 65))
        let expected = try await pipeline.predict(features: input)
        #expect(try await pipeline.predict(consuming: input) == expected)
        #expect(input.rowValues() == original)
        let wrongNames = try PreparedNumericBatch(columnNames: ["wrong"], columns: [(0..<65).map(Double.init)])
        await #expect(throws: (any Error).self) { _ = try await pipeline.predict(consuming: wrongNames) }
        let nonfinite = try PreparedNumericBatch(columnNames: ["x"], columns: [[.nan]])
        await #expect(throws: (any Error).self) { _ = try await pipeline.predict(consuming: nonfinite) }
        await #expect(throws: (any Error).self) { try await pipeline.fit(consuming: input, targets: []) }
        #expect(try await pipeline.predict(consuming: input) == expected)
        let badEstimator = RegressionPipeline(estimator: EchoRegressor(wrongCount: true))
        await #expect(throws: SwiftMLError.dimensionMismatch(expected: 65, got: 0)) {
            _ = try await badEstimator.predict(consuming: input)
        }
    }
}
