import Testing
import SwiftML
import SwiftPreprocessing

@Suite("Public standardized MLP cancellation", .serialized)
struct StandardizedMLPCancellationTests {
    @Test func queuedPreparationAndPredictionCancelAndReuse() async throws {
        let training = try publicMLPTraining()
        let fitted = try await publicMLPModel().fitStandardized(features: training.batch, targets: training.targets)
        let request = MLPPredictionRequest(source: training.batch)
        let budget = try MemoryBudget(limit: 16 * 1024 * 1024)
        var blocker = try await budget.acquire(MemoryEstimate(capacities: [budget.limit]))
        let preparing = Task { try await fitted.prepare(request, budget: budget) }
        #expect(await preparedMLPWaitUntil { await budget.queuedCount == 1 })
        preparing.cancel(); await blocker.finish()
        await #expect(throws: CancellationError.self) { try await preparing.value }
        let input = try await fitted.prepare(request, budget: budget)
        blocker = try await budget.acquire(MemoryEstimate(capacities: [budget.limit]))
        let predicting = Task { try await fitted.predict(input, budget: budget) }
        #expect(await preparedMLPWaitUntil { await budget.queuedCount == 1 })
        predicting.cancel(); await blocker.finish()
        await #expect(throws: CancellationError.self) { try await predicting.value }
        #expect(try await fitted.predict(input, budget: budget).count == request.rowCount)
        #expect(await budget.reservedBytes == 0)
        #expect(await budget.queuedCount == 0)
    }
}
