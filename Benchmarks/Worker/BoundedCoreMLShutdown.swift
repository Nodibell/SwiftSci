import CoreML
import Foundation
import SwiftML
import SwiftPreprocessing
import SwiftSciBenchmarkSupport

private enum BoundedInjectedFailure: Error { case source, consumer }
private actor BoundedShutdownTrigger {
    private(set) var reserved = 0
    private(set) var instant: ContinuousClock.Instant?
    func mark(_ bytes: Int) { reserved = bytes; instant = .now }
}
private struct BoundedShutdownResult: Encodable {
    let scenario: String
    let loaded: Int
    let consumed: Int
    let reservedAtTrigger: Int
    let reservedAfterDrain: Int
    let drainSeconds: Double
}

func boundedCoreMLShutdown(request: URL, output: URL) async -> Int32 {
    do {
        let q = try JSONDecoder().decode(BoundedCoreMLRequest.self, from: Data(contentsOf: request))
        guard [1024, 8192].contains(q.rows), ["cpu", "neural"].contains(q.policy), q.batches == 128 else {
            throw BenchmarkFailure("Shutdown trial requires 128 fixed-shape batches")
        }
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        let (plan, _, _) = try prepareBoundedFixture(q, file: file)
        let compiled = try await MLModel.compileModel(at: URL(fileURLWithPath: q.model))
        defer { try? FileManager.default.removeItem(at: compiled) }
        let pool = try CoreMLMatrixPool.Configuration(maximumConcurrentPredictions: 2,
            retainedBytes: 33_554_432, requestBytes: 16_777_216)
        let config = try CoreMLBatchPipeline.Configuration(pool: pool, maximumInFlightBatches: 4,
            retainedBytes: 1_048_576, batchBytes: 33_554_432)
        let quotaBytes = try config.requiredBytes
        let budget = try MemoryBudget(limit: quotaBytes)
        var results: [BoundedShutdownResult] = []
        for scenario in ["source-error", "consumer-error", "cancel-source", "cancel-consumer"] {
            let source = try BoundedCoreMLSource(url: file, names: plan.columnNames,
                rows: q.rows, batches: q.batches, delayMilliseconds: 0)
            let trigger = BoundedShutdownTrigger()
            // Keep cancellation confined to this run so the next scenario can prove recovery.
            let task = Task {
                try await CoreMLBatchPipeline.run(compiledModelURL: compiled, inputColumns: plan.columnNames,
                    outputName: "result", computeUnits: q.policy == "cpu" ? .cpuOnly : .cpuAndNeuralEngine,
                    preprocessing: plan, budget: budget, configuration: config,
                    load: { request in
                        if request.sequence == 32 && (scenario == "source-error" || scenario == "cancel-source") {
                            await trigger.mark(budget.reservedBytes)
                            if scenario == "source-error" { throw BoundedInjectedFailure.source }
                            withUnsafeCurrentTask { $0?.cancel() }
                            try Task.checkCancellation()
                        }
                        return try await source.load(request.sequence)
                    }, consume: { sequence, result in
                        if sequence == 32 && (scenario == "consumer-error" || scenario == "cancel-consumer") {
                            await trigger.mark(budget.reservedBytes)
                            if scenario == "consumer-error" { throw BoundedInjectedFailure.consumer }
                            withUnsafeCurrentTask { $0?.cancel() }
                            try Task.checkCancellation()
                        }
                        try await source.consume(sequence, prediction: result)
                    })
            }
            var expectedFailure = false
            do { try await task.value }
            catch BoundedInjectedFailure.source { expectedFailure = scenario == "source-error" }
            catch BoundedInjectedFailure.consumer { expectedFailure = scenario == "consumer-error" }
            catch is CancellationError { expectedFailure = scenario.hasPrefix("cancel-") }
            try await source.close()
            guard expectedFailure, let instant = await trigger.instant,
                  await trigger.reserved == quotaBytes, await budget.reservedBytes == 0,
                  await budget.queuedCount == 0 else { throw BenchmarkFailure("Shutdown did not preserve then release admission") }
            let result = await BoundedShutdownResult(scenario: scenario, loaded: source.loaded,
                consumed: source.consumed, reservedAtTrigger: trigger.reserved,
                reservedAfterDrain: budget.reservedBytes, drainSeconds: elapsedSeconds(since: instant))
            results.append(result)
            let recovered = try await budget.acquire(MemoryEstimate(capacities: [quotaBytes]))
            await recovered.finish()
        }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(results).write(to: output, options: .atomic)
        return 0
    } catch { fputs("Bounded shutdown trial failed: \(error)\n", stderr); return 1 }
}
