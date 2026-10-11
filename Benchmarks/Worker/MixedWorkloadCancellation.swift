import Foundation
import SwiftPreprocessing
import SwiftSciBenchmarkSupport

struct MixedQueuedCancellation: Codable {
    let queuedBeforeCancellation: Int
    let queuedAfterCancellation: Int
    let cpuCompleted: Int
    let gpuCompleted: Int
    let finalReservedBytes: Int
    let finalQueued: Int
}
struct MixedWorkloadRecord: Codable {
    let samples: [MixedSample]
    let queuedCancellation: MixedQueuedCancellation
}

private func waitForMixedQueue(_ budget: MemoryBudget, count: Int) async throws {
    let deadline = ContinuousClock.now.advanced(by: .seconds(5))
    while await budget.queuedCount != count {
        guard ContinuousClock.now < deadline else { throw BenchmarkFailure("Mixed admission queue did not reach expected state") }
        try await Task.sleep(for: .milliseconds(1))
    }
}

func mixedQueuedCancellation(_ q: MixedWorkloadRequest, fixture: MixedWorkloadFixture) async throws -> MixedQueuedCancellation {
    let limit = try fixture.configuration.requiredBytes
    let budget = try MemoryBudget(limit: limit)
    let blocker = try await budget.acquire(MemoryEstimate(capacities: [limit]))
    let sampler = MixedSampler()
    let start = ContinuousClock.now
    let core = Task { try await mixedCoreMLLane(q: q, fixture: fixture, budget: budget, sampler: sampler, start: start) }
    var peers: [Task<MixedLaneResult, any Error>] = []
    do {
        try await waitForMixedQueue(budget, count: 1)
        for name in ["cpu", "gpu"] {
            peers.append(Task { try await mixedNumericLane(name, q: q, fixture: fixture,
                budget: budget, sampler: sampler, start: start) })
        }
        try await waitForMixedQueue(budget, count: 3)
        core.cancel()
        do { _ = try await core.value; throw BenchmarkFailure("Queued Core ML request was not cancelled") }
        catch is CancellationError { }
        try await waitForMixedQueue(budget, count: 2)
        await blocker.finish()
        let cpu = try await peers[0].value
        let gpu = try await peers[1].value
        guard await budget.reservedBytes == 0, await budget.queuedCount == 0,
              cpu.completed == q.batches, gpu.completed == q.batches else {
            throw BenchmarkFailure("Peers did not recover after queued cancellation")
        }
        let recovery = try await budget.acquire(MemoryEstimate(capacities: [limit]))
        await recovery.finish()
        return MixedQueuedCancellation(queuedBeforeCancellation: 3, queuedAfterCancellation: 2,
            cpuCompleted: cpu.completed, gpuCompleted: gpu.completed, finalReservedBytes: 0, finalQueued: 0)
    } catch {
        core.cancel(); peers.forEach { $0.cancel() }
        await blocker.finish()
        _ = try? await core.value
        for peer in peers { _ = try? await peer.value }
        throw error
    }
}
