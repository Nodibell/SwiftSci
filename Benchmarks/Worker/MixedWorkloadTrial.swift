import CoreML
import Foundation
import SwiftPreprocessing
import SwiftSciBenchmarkSupport

func mixedSample(_ q: MixedWorkloadRequest, fixture: MixedWorkloadFixture,
    mode: String, iteration: Int) async throws -> MixedSample {
    let coreQuota = try fixture.configuration.requiredBytes
    let extra = mode == "tight" ? 0 : (mode == "balanced" ? mixedSmallQuota : 2 * mixedSmallQuota)
    let budget = try MemoryBudget(limit: coreQuota + extra)
    let sampler = MixedSampler()
    let start = ContinuousClock.now
    let monitoring = Task {
        while !Task.isCancelled {
            try await sampler.sample(budget)
            do { try await Task.sleep(for: .milliseconds(10)) } catch { return }
        }
    }
    let results: [MixedLaneResult]
    do {
        if mode == "isolated" {
            let cpu = try await mixedNumericLane("cpu", q: q, fixture: fixture, budget: budget, sampler: sampler, start: start)
            let gpu = try await mixedNumericLane("gpu", q: q, fixture: fixture, budget: budget, sampler: sampler, start: start)
            let coreml = try await mixedCoreMLLane(q: q, fixture: fixture, budget: budget, sampler: sampler, start: start)
            results = [cpu, gpu, coreml]
        } else {
            let gate = MixedStartGate(3)
            results = try await withThrowingTaskGroup(of: MixedLaneResult.self) { group in
                for name in ["cpu", "gpu", "coreml"] {
                    group.addTask {
                        await gate.arrive()
                        if name == "coreml" {
                            return try await mixedCoreMLLane(q: q, fixture: fixture, budget: budget,
                                sampler: sampler, start: start, cancel: mode == "cancel-active")
                        }
                        return try await mixedNumericLane(name, q: q, fixture: fixture,
                            budget: budget, sampler: sampler, start: start)
                    }
                }
                var lanes: [MixedLaneResult] = []
                for try await result in group { lanes.append(result) }
                return lanes.sorted { $0.lane < $1.lane }
            }
        }
        monitoring.cancel()
        try await monitoring.value
    } catch { monitoring.cancel(); _ = try? await monitoring.value; throw error }
    let wall = elapsedSeconds(since: start)
    let reserved = await budget.reservedBytes
    let queued = await budget.queuedCount
    guard reserved == 0, queued == 0, await budget.peak <= budget.limit else {
        throw BenchmarkFailure("Mixed workload admission did not drain")
    }
    for lane in results {
        if mode == "cancel-active" && lane.lane == "coreml" {
            guard lane.cancelled, lane.completed == 16 else { throw BenchmarkFailure("Active cancellation was not exercised") }
        } else if lane.completed != q.batches { throw BenchmarkFailure("Missing lane completions") }
    }
    let recovery = try await budget.acquire(MemoryEstimate(capacities: [budget.limit]))
    await recovery.finish()
    return await MixedSample(mode: mode, iteration: iteration, wallSeconds: wall,
        budgetBytes: budget.limit, peakReservedBytes: budget.peak, finalReservedBytes: reserved,
        finalQueued: queued, peakQueued: sampler.peakQueued, peakResidentBytes: sampler.peakResident,
        finalResidentBytes: try residentBytes(), thermalState: ProcessInfo.processInfo.thermalState.rawValue, lanes: results)
}

func mixedWorkloadTrial(request: URL, output: URL) async -> Int32 {
    do {
        let q = try JSONDecoder().decode(MixedWorkloadRequest.self, from: Data(contentsOf: request))
        guard [1024, 8192].contains(q.rows), ["cpu", "neural"].contains(q.policy),
              (32...1024).contains(q.batches), [8, q.batches].contains(q.quantum), q.batches % q.quantum == 0 else {
            throw BenchmarkFailure("Invalid mixed workload bounds")
        }
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        let model = try await MLModel.compileModel(at: URL(fileURLWithPath: q.model))
        defer { try? FileManager.default.removeItem(at: model) }
        let fixture = try await MixedWorkloadFixture.make(q, file: file, model: model)
        let baseline = try await mixedSample(q, fixture: fixture, mode: "isolated", iteration: -1)
        let reference = baseline.lanes.first { $0.lane == "coreml" }!.hashes
        var samples: [MixedSample] = []
        let modes = ["isolated", "tight", "balanced", "wide"]
        for iteration in 0..<3 {
            for index in modes.indices {
                let mode = modes[(index + iteration) % modes.count]
                let sample = try await mixedSample(q, fixture: fixture, mode: mode, iteration: iteration)
                guard sample.lanes.first(where: { $0.lane == "coreml" })?.hashes == reference else {
                    throw BenchmarkFailure("Core ML differs from isolated reference")
                }
                samples.append(sample)
            }
        }
        // Cancellation stays in a child task so the rest of this experiment remains live.
        let cancellation = Task { try await mixedSample(q, fixture: fixture, mode: "cancel-active", iteration: 0) }
        samples.append(try await cancellation.value)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let queued = try await mixedQueuedCancellation(q, fixture: fixture)
        let record = MixedWorkloadRecord(samples: samples, queuedCancellation: queued)
        try encoder.encode(record).write(to: output, options: .atomic)
        return 0
    } catch { fputs("Mixed workload trial failed: \(error)\n", stderr); return 1 }
}
