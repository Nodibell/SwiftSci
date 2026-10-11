import Foundation
import SwiftML
import SwiftPreprocessing
import SwiftSciBenchmarkSupport

func persistentCoreMLLane(_ q: MixedWorkloadRequest, fixture: MixedWorkloadFixture,
    pool: CoreMLMatrixPool, scope: PersistentTrialScope, sampler: MixedSampler,
    start: ContinuousClock.Instant, cancel: Bool = false) async throws -> MixedLaneResult {
    var waits: [Double] = [], services: [Double] = [], completions: [Double] = []
    var reference: [String] = []
    var completed = 0
    var cancelled = false
    for base in stride(from: 0, to: q.batches, by: q.quantum) {
        let requested = ContinuousClock.now
        let token = try await scope.work.acquire(MemoryEstimate(capacities: [scope.windowBytes]))
        waits.append(elapsedSeconds(since: requested))
        let admitted = ContinuousClock.now
        var source: BoundedCoreMLSource?
        do {
            let current = try BoundedCoreMLSource(url: fixture.file, names: fixture.plan.columnNames,
                rows: q.rows, batches: q.quantum, delayMilliseconds: 0)
            source = current
            try await CoreMLBatchPipeline.runPersistentTrial(pool: pool, preprocessing: fixture.plan,
                configuration: fixture.configuration,
                load: { request in try await current.load(request.sequence) },
                consume: { sequence, result in
                    if cancel && base + sequence == 16 {
                        withUnsafeCurrentTask { $0?.cancel() }
                        try Task.checkCancellation()
                    }
                    try await current.consume(sequence, prediction: result)
                })
            try await current.close()
            let hashes = await current.hashes
            if reference.isEmpty { reference = hashes }
            guard hashes == reference, await current.consumed == q.quantum else {
                throw BenchmarkFailure("Persistent chunk changed results")
            }
            completed += q.quantum
            services.append(elapsedSeconds(since: admitted))
            await token.finish()
            completions.append(elapsedSeconds(since: start))
            try await sampler.sample(scope.work)
        } catch {
            try? await source?.close()
            await token.finish()
            guard cancel, error is CancellationError else { throw error }
            completed += await source?.consumed ?? 0
            cancelled = true
            break
        }
    }
    return MixedLaneResult(lane: "coreml", completed: completed, admissionSeconds: waits,
        serviceSeconds: services, completionSeconds: completions, cancelled: cancelled, hashes: reference)
}
