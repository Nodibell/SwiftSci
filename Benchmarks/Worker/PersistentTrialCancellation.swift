import Foundation
import SwiftPreprocessing
import SwiftSciBenchmarkSupport

private func persistentQueue(_ scope: PersistentTrialScope, count: Int) async throws {
    let deadline = ContinuousClock.now.advanced(by: .seconds(5))
    while await scope.work.queuedCount != count {
        guard ContinuousClock.now < deadline else { throw BenchmarkFailure("Persistent queue did not reach target") }
        try await Task.sleep(for: .milliseconds(1))
    }
}

func persistentQueuedCancellation(_ q: MixedWorkloadRequest, fixture: MixedWorkloadFixture) async throws -> MixedQueuedCancellation {
    let scope = try PersistentTrialScope(configuration: fixture.configuration,
        totalBytes: fixture.configuration.requiredBytes)
    let result = try await scope.withModel(q, fixture: fixture) { pool in
        let blocker = try await scope.work.acquire(MemoryEstimate(capacities: [scope.work.limit]))
        let sampler = MixedSampler()
        let start = ContinuousClock.now
        let core = Task { try await persistentCoreMLLane(q, fixture: fixture,
            pool: pool, scope: scope, sampler: sampler, start: start) }
        var peers: [Task<MixedLaneResult, any Error>] = []
        do {
            try await persistentQueue(scope, count: 1)
            for name in ["cpu", "gpu"] {
                peers.append(Task { try await mixedNumericLane(name, q: q, fixture: fixture,
                    budget: scope.work, sampler: sampler, start: start) })
            }
            try await persistentQueue(scope, count: 3)
            core.cancel()
            do { _ = try await core.value; throw BenchmarkFailure("Queued persistent request was not cancelled") }
            catch is CancellationError { }
            try await persistentQueue(scope, count: 2)
            await blocker.finish()
            let cpu = try await peers[0].value
            let gpu = try await peers[1].value
            guard await scope.work.reservedBytes == 0, await scope.work.queuedCount == 0,
                  await scope.retained.reservedBytes == scope.retainedBytes else {
                throw BenchmarkFailure("Persistent cancellation lost ownership accounting")
            }
            return MixedQueuedCancellation(queuedBeforeCancellation: 3, queuedAfterCancellation: 2,
                cpuCompleted: cpu.completed, gpuCompleted: gpu.completed, finalReservedBytes: 0, finalQueued: 0)
        } catch {
            core.cancel(); peers.forEach { $0.cancel() }; await blocker.finish()
            _ = try? await core.value
            for peer in peers { _ = try? await peer.value }
            throw error
        }
    }
    try await scope.verifyDrained()
    return result
}
