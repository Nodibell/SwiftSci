import Foundation
import SwiftML
import SwiftSciBenchmarkSupport

func persistentTrialSample(_ q: MixedWorkloadRequest, fixture: MixedWorkloadFixture,
    mode: String, iteration: Int, cancel: Bool = false) async throws -> PersistentTrialSample {
    let required = try fixture.configuration.requiredBytes
    let extra = mode == "tight" ? 0 : (mode == "balanced" ? mixedSmallQuota : 2 * mixedSmallQuota)
    let scope = try PersistentTrialScope(configuration: fixture.configuration, totalBytes: required + extra)
    let sampler = MixedSampler()
    let start = ContinuousClock.now
    let monitoring = Task {
        while !Task.isCancelled {
            try await sampler.sample(scope.work)
            do { try await Task.sleep(for: .milliseconds(10)) } catch { return }
        }
    }
    let results: ([MixedLaneResult], Double)
    do {
        results = try await scope.withModel(q, fixture: fixture) { pool in
            let startup = elapsedSeconds(since: start)
            let gate = MixedStartGate(3)
            let lanes = try await withThrowingTaskGroup(of: MixedLaneResult.self) { group in
                for name in ["cpu", "gpu", "coreml"] {
                    group.addTask {
                        await gate.arrive()
                        if name == "coreml" {
                            return try await persistentCoreMLLane(q, fixture: fixture, pool: pool,
                                scope: scope, sampler: sampler, start: start, cancel: cancel)
                        }
                        return try await mixedNumericLane(name, q: q, fixture: fixture,
                            budget: scope.work, sampler: sampler, start: start)
                    }
                }
                var lanes: [MixedLaneResult] = []
                for try await lane in group { lanes.append(lane) }
                return lanes.sorted { $0.lane < $1.lane }
            }
            guard await scope.retained.reservedBytes == scope.retainedBytes,
                  await scope.work.reservedBytes == 0 else {
                throw BenchmarkFailure("Model lifetime or work cleanup changed")
            }
            return (lanes, startup)
        }
        monitoring.cancel(); try await monitoring.value
    } catch { monitoring.cancel(); _ = try? await monitoring.value; throw error }
    let wall = elapsedSeconds(since: start)
    for lane in results.0 {
        if cancel && lane.lane == "coreml" {
            guard lane.cancelled, lane.completed == 16 else { throw BenchmarkFailure("Persistent cancellation not exercised") }
        } else if lane.completed != q.batches { throw BenchmarkFailure("Persistent peer did not finish") }
    }
    let peak = await scope.retained.peak + scope.work.peak
    guard peak <= scope.totalBytes else { throw BenchmarkFailure("Partition sum exceeded total allowance") }
    try await scope.verifyDrained()
    let sample = await MixedSample(mode: cancel ? "cancel-active" : mode, iteration: iteration,
        wallSeconds: wall, budgetBytes: scope.totalBytes, peakReservedBytes: peak,
        finalReservedBytes: 0, finalQueued: 0, peakQueued: sampler.peakQueued,
        peakResidentBytes: sampler.peakResident, finalResidentBytes: try residentBytes(),
        thermalState: ProcessInfo.processInfo.thermalState.rawValue, lanes: results.0)
    return PersistentTrialSample(variant: "persistent", quantum: q.quantum, poolScopes: 1,
        startupSeconds: results.1, retainedBytes: scope.retainedBytes,
        workBudgetBytes: scope.totalBytes - scope.retainedBytes, sample: sample)
}
