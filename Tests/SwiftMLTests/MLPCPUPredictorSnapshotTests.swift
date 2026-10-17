import Testing
import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftML

@Suite("Fitted CPU prediction snapshots", .serialized)
struct MLPCPUPredictorSnapshotTests {
    @Test func estimateAndValuesSurviveRefitting() async throws {
        let c = try await preparedMLPTestContext()
        let view = try IndexedNumericBatch(c.source, selectingRows: Array((0..<65).reversed()))
        let budget = try MemoryBudget(limit: 8 * 1024 * 1024)
        let input = try await PreparedMLPInput.prepare(features: view, preprocessing: c.plan, budget: budget)
        let estimate = try c.fitted.predictionEstimate(for: input)
        let expected = try c.fitted.predictExternallyAdmitted(features: input)
        let rows = matrixMLPFeatures(8, width: 7)
        try await c.model.fit(features: rows, targets: rows.map { -$0[0] - 1000 })
        let revised = try await c.model.makeCPUPredictor()
        let old = try await c.fitted.predict(features: input, budget: budget)
        let changed = try revised.predictExternallyAdmitted(features: input)
        #expect(old.map(\.bitPattern) == expected.map(\.bitPattern))
        #expect(changed.map(\.bitPattern) != expected.map(\.bitPattern))
        #expect(try c.fitted.predictionEstimate(for: input).bytes == estimate.bytes)
        #expect(estimate.bytes >= input.retainedBytes + c.fitted.retainedBytes)
        #expect(await budget.reservedBytes == 0)
    }

    @Test func CPUCaptureDoesNotRejectFitRoutingMetadata() async throws {
        let c = try await preparedMLPTestContext()
        let rows = matrixMLPFeatures(8, width: 7)
        let model = MLPRegressor(hiddenLayerSizes: [9, 5], maxIter: 1, seed: 42, requestedDevice: .gpu)
        try await model.fit(features: rows, targets: rows.map { $0[0] })
        let fitted = try await model.makeCPUPredictor()
        let view = try IndexedNumericBatch(c.source, selectingRows: [64, 1, 64, 0, 33])
        let budget = try MemoryBudget(limit: 8 * 1024 * 1024)
        let input = try await PreparedMLPInput.prepare(features: view, preprocessing: c.plan, budget: budget)
        let expected = try await model.predictSourceOrdered(features: view, preprocessing: c.plan)
        let actual = try fitted.predictExternallyAdmitted(features: input)
        #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
        #expect(fitted.retainedBytes > 0)
    }
}
