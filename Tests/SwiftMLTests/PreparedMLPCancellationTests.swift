import Testing
import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftML

@Suite("Prepared MLP cancellation and snapshots", .serialized)
struct PreparedMLPCancellationTests {
    @Test func queuedPreparationAndPredictionReleaseWithoutHoldingIdleBytes() async throws {
        let c = try await preparedMLPTestContext()
        let view = try IndexedNumericBatch(c.source, selectingRows: Array((0..<65).reversed()))
        let budget = try MemoryBudget(limit: 8 * 1024 * 1024)
        let input = try await PreparedMLPInput.prepare(features: view, preprocessing: c.plan, budget: budget)
        for preparation in [true, false] {
            let blocker = try await budget.acquire(MemoryEstimate(capacities: [budget.limit]))
            let pending = Task {
                if preparation { _ = try await PreparedMLPInput.prepare(features: view, preprocessing: c.plan, budget: budget) }
                else { _ = try await c.model.predict(features: input, budget: budget) }
            }
            let queued = await preparedMLPWaitUntil { await budget.queuedCount == 1 }
            pending.cancel()
            await blocker.finish()
            await #expect(throws: CancellationError.self) { try await pending.value }
            #expect(queued)
            #expect(await budget.reservedBytes == 0)
            #expect(await budget.queuedCount == 0)
        }
        let actual = try await c.model.predict(features: input, budget: budget)
        let expected = try await c.model.predictSourceOrdered(features: view, preprocessing: c.plan)
        #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
    }

    @Test func queuedPredictionUsesItsCapturedWeightsAfterRefit() async throws {
        let c = try await preparedMLPTestContext()
        let view = try IndexedNumericBatch(c.source, selectingRows: Array((0..<65).reversed()))
        let budget = try MemoryBudget(limit: 8 * 1024 * 1024)
        let input = try await PreparedMLPInput.prepare(features: view, preprocessing: c.plan, budget: budget)
        let expected = try await c.model.predict(features: input, budget: budget)
        let blocker = try await budget.acquire(MemoryEstimate(capacities: [budget.limit]))
        let pending = Task { try await c.model.predict(features: input, budget: budget) }
        let queued = await preparedMLPWaitUntil { await budget.queuedCount == 1 }
        guard queued else {
            pending.cancel(); await blocker.finish(); _ = try? await pending.value
            Issue.record("Prediction never queued"); return
        }
        let rows = matrixMLPFeatures(8, width: 7)
        try await c.model.fit(features: rows, targets: rows.map { -$0[0] - 1000 })
        await blocker.finish()
        let actual = try await pending.value
        let changed = try await c.model.predict(features: input, budget: budget)
        #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
        #expect(changed.map(\.bitPattern) != expected.map(\.bitPattern))
        #expect(await budget.reservedBytes == 0)
    }
}
