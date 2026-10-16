import Testing
import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftML

@Suite("CPU snapshot cancellation", .serialized)
struct MLPCPUPredictorCancellationTests {
    @Test func queuedSnapshotCancellationReleasesItsLease() async throws {
        let c = try await preparedMLPTestContext()
        let view = try IndexedNumericBatch(c.source, selectingRows: [64, 1, 64, 0, 33])
        let budget = try MemoryBudget(limit: 8 * 1024 * 1024)
        let input = try await PreparedMLPInput.prepare(features: view, preprocessing: c.plan, budget: budget)
        let blocker = try await budget.acquire(MemoryEstimate(capacities: [budget.limit]))
        let pending = Task { try await c.fitted.predict(features: input, budget: budget) }
        let queued = await preparedMLPWaitUntil { await budget.queuedCount == 1 }
        pending.cancel()
        await blocker.finish()
        await #expect(throws: CancellationError.self) { try await pending.value }
        #expect(queued)
        #expect(await budget.reservedBytes == 0)
        #expect(await budget.queuedCount == 0)
        let result = try await c.fitted.predict(features: input, budget: budget)
        let expected = try c.fitted.predictExternallyAdmitted(features: input)
        #expect(result.map(\.bitPattern) == expected.map(\.bitPattern))
    }
}
