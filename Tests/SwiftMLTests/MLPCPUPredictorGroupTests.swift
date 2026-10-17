import Testing
import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftML

@Suite("Stable externally admitted CPU groups", .serialized)
struct MLPCPUPredictorGroupTests {
    @Test func refitCannotChangeAnExternallyEstimatedGroup() async throws {
        let c = try await preparedMLPTestContext()
        let view = try IndexedNumericBatch(c.source, selectingRows: Array((0..<65).reversed()))
        let budget = try MemoryBudget(limit: 8 * 1024 * 1024)
        let input = try await PreparedMLPInput.prepare(features: view, preprocessing: c.plan, budget: budget)
        let estimate = try c.fitted.predictionEstimate(for: input)
        let expected = try c.fitted.predictExternallyAdmitted(features: input)
        let blocker = try await budget.acquire(MemoryEstimate(capacities: [budget.limit]))
        let pending = Task {
            let lease = try await budget.acquire(estimate)
            do {
                var result: [Double] = []
                for _ in 0..<4 { result = try c.fitted.predictExternallyAdmitted(features: input) }
                await lease.finish()
                return result
            } catch { await lease.finish(); throw error }
        }
        let queued = await preparedMLPWaitUntil { await budget.queuedCount == 1 }
        guard queued else {
            pending.cancel(); await blocker.finish(); _ = try? await pending.value
            Issue.record("Snapshot group never queued"); return
        }
        let rows = matrixMLPFeatures(8, width: 8)
        try await c.model.fit(features: rows, targets: rows.map { $0[0] })
        let revised = try await c.model.makeCPUPredictor()
        await blocker.finish()
        let result = try await pending.value
        #expect(result.map(\.bitPattern) == expected.map(\.bitPattern))
        #expect(try c.fitted.predictionEstimate(for: input).bytes == estimate.bytes)
        #expect(throws: SwiftMLError.dimensionMismatch(expected: 8, got: 7)) {
            try revised.predictionEstimate(for: input)
        }
        #expect(await budget.reservedBytes == 0)
        #expect(await budget.queuedCount == 0)
    }
}
