import CoreML
import Foundation
import SwiftML
import SwiftPreprocessing
import SwiftSciBenchmarkSupport

let mixedSmallQuota = 67_108_864

func mixedNumericLane(_ name: String, q: MixedWorkloadRequest, fixture: MixedWorkloadFixture,
    budget: MemoryBudget, sampler: MixedSampler, start: ContinuousClock.Instant) async throws -> MixedLaneResult {
    var waits: [Double] = [], services: [Double] = [], completions: [Double] = []
    for _ in 0..<q.batches {
        try Task.checkCancellation()
        let requested = ContinuousClock.now
        let token = try await budget.acquire(MemoryEstimate(capacities: [mixedSmallQuota]))
        waits.append(elapsedSeconds(since: requested))
        let admitted = ContinuousClock.now
        do {
            try Task.checkCancellation()
            if name == "cpu" {
                let output = try fixture.scaler.transform(fixture.imputer.transform(fixture.input))
                guard mixedHash(output.columns.map(\.values)) == fixture.cpuHash,
                      output.originalRowIndices == fixture.input.originalRowIndices else {
                    throw BenchmarkFailure("CPU result or row identity changed under contention")
                }
            } else {
                let output = try await fixture.regression.predict(features: fixture.input)
                guard mixedHash([output]) == fixture.gpuHash else {
                    throw BenchmarkFailure("GPU result changed under contention")
                }
            }
            services.append(elapsedSeconds(since: admitted))
            await token.finish()
            completions.append(elapsedSeconds(since: start))
            try await sampler.sample(budget)
        } catch { await token.finish(); throw error }
    }
    return MixedLaneResult(lane: name, completed: q.batches, admissionSeconds: waits,
        serviceSeconds: services, completionSeconds: completions, cancelled: false, hashes: [])
}

func mixedCoreMLLane(q: MixedWorkloadRequest, fixture: MixedWorkloadFixture,
    budget: MemoryBudget, sampler: MixedSampler, start: ContinuousClock.Instant,
    cancel: Bool = false) async throws -> MixedLaneResult {
    var waits: [Double] = [], services: [Double] = [], completions: [Double] = []
    var reference: [String] = []
    var completed = 0
    var cancelled = false
    let quota = try fixture.configuration.requiredBytes
    for base in stride(from: 0, to: q.batches, by: q.quantum) {
        let requested = ContinuousClock.now
        let token = try await budget.acquire(MemoryEstimate(capacities: [quota]))
        waits.append(elapsedSeconds(since: requested))
        let admitted = ContinuousClock.now
        // The private budget subdivides the parent reservation; never reacquire the held parent.
        let privateBudget = try MemoryBudget(limit: quota)
        let source = try BoundedCoreMLSource(url: fixture.file, names: fixture.plan.columnNames,
            rows: q.rows, batches: q.quantum, delayMilliseconds: 0)
        do {
            try await CoreMLBatchPipeline.run(compiledModelURL: fixture.model,
                inputColumns: fixture.plan.columnNames, outputName: "result",
                computeUnits: q.policy == "cpu" ? .cpuOnly : .cpuAndNeuralEngine,
                preprocessing: fixture.plan, budget: privateBudget, configuration: fixture.configuration,
                load: { request in try await source.load(request.sequence) },
                consume: { sequence, result in
                    if cancel && base + sequence == 16 {
                        withUnsafeCurrentTask { $0?.cancel() }
                        try Task.checkCancellation()
                    }
                    try await source.consume(sequence, prediction: result)
                })
        } catch {
            try? await source.close()
            await token.finish()
            guard cancel, error is CancellationError, await privateBudget.reservedBytes == 0 else { throw error }
            completed += await source.consumed
            cancelled = true
            break
        }
        try await source.close()
        let hashes = await source.hashes
        if reference.isEmpty { reference = hashes }
        guard hashes == reference, await privateBudget.reservedBytes == 0,
              await source.consumed == q.quantum else {
            await token.finish()
            throw BenchmarkFailure("Core ML chunk changed results or did not drain")
        }
        completed += q.quantum
        services.append(elapsedSeconds(since: admitted))
        await token.finish()
        completions.append(elapsedSeconds(since: start))
        try await sampler.sample(budget)
    }
    return MixedLaneResult(lane: "coreml", completed: completed, admissionSeconds: waits,
        serviceSeconds: services, completionSeconds: completions, cancelled: cancelled, hashes: reference)
}
