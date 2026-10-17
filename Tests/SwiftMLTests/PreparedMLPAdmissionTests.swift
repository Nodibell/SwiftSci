import Testing
import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftML

@Suite("Prepared MLP admission", .serialized)
struct PreparedMLPAdmissionTests {
    @Test func exactLimitsRejectBeforeWorkAndReleaseOnSuccess() async throws {
        let c = try await preparedMLPTestContext()
        let view = try IndexedNumericBatch(c.source, selectingRows: Array((0..<65).reversed()))
        let preparation = try PreparedMLPInput.preparationEstimate(features: view, preprocessing: c.plan)
        let tooSmall = try MemoryBudget(limit: preparation.bytes - 1)
        await #expect(throws: MemoryAdmissionError.exceedsLimit(required: preparation.bytes, limit: preparation.bytes - 1)) {
            try await PreparedMLPInput.prepare(features: view, preprocessing: c.plan, budget: tooSmall)
        }
        #expect(await tooSmall.peak == 0)
        let budget = try MemoryBudget(limit: preparation.bytes)
        let input = try await PreparedMLPInput.prepare(features: view, preprocessing: c.plan, budget: budget)
        #expect(await budget.peak == preparation.bytes)
        #expect(await budget.reservedBytes == 0)
        let prediction = try c.fitted.predictionEstimate(for: input)
        let smallPrediction = try MemoryBudget(limit: prediction.bytes - 1)
        await #expect(throws: MemoryAdmissionError.exceedsLimit(required: prediction.bytes, limit: prediction.bytes - 1)) {
            try await c.model.predict(features: input, budget: smallPrediction)
        }
        #expect(await smallPrediction.peak == 0)
        let exactPrediction = try MemoryBudget(limit: prediction.bytes)
        _ = try await c.model.predict(features: input, budget: exactPrediction)
        #expect(await exactPrediction.peak == prediction.bytes)
        #expect(await exactPrediction.reservedBytes == 0)
        let unfitted = MLPRegressor(requestedDevice: .cpu)
        await #expect(throws: SwiftMLError.modelNotFitted) { try await unfitted.predict(features: input, budget: budget) }
        let rows = matrixMLPFeatures(8, width: 8)
        try await c.model.fit(features: rows, targets: rows.map { $0[0] })
        await #expect(throws: SwiftMLError.dimensionMismatch(expected: 8, got: 7)) {
            try await c.model.predict(features: input, budget: budget)
        }
        #expect(await budget.reservedBytes == 0)
    }

    @Test func capacitiesAndOverflowAreAccountedWithoutMaterializingProvenance() async throws {
        var provenance = [0, 1, 2]; provenance.reserveCapacity(4096)
        let source = PreparedNumericBatch(columnNames: ["x"],
            columns: [CompactNumericColumn([1, 2, 3])],
            rowCount: 3, sourceRows: provenance)
        var indices = [0, 1, 2]; indices.reserveCapacity(8192)
        let view = try IndexedNumericBatch(source, selectingRows: indices)
        let plan = try StandardPreprocessingPlan(training: source)
        let budget = try MemoryBudget(limit: 8 * 1024 * 1024)
        let input = try await PreparedMLPInput.prepare(features: view, preprocessing: plan, budget: budget)
        #expect(source.provenanceStorageCapacity == provenance.capacity)
        #expect(input.retainedBytes >= (provenance.capacity + indices.capacity) * MemoryLayout<Int>.stride)
        #expect(try preparedMLPBytes(0, Int.max) == 0)
        #expect(throws: MemoryAdmissionError.invalidCapacity) { try preparedMLPBytes(Int.max, 2) }
        #expect(throws: MemoryAdmissionError.invalidCapacity) { try preparedMLPBytes(-1, 8) }
        #expect(await budget.reservedBytes == 0)
    }
}
