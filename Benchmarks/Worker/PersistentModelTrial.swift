import CoreML
import Foundation
import SwiftPreprocessing
import SwiftSciBenchmarkSupport

struct PersistentTrialRecord: Codable {
    let samples: [PersistentTrialSample]
    let cancellation: [PersistentTrialSample]
    let queuedCancellation: MixedQueuedCancellation
    let undersizedPartitionRejected: Bool
}

func persistentModelTrial(request: URL, output: URL) async -> Int32 {
    do {
        let q = try JSONDecoder().decode(MixedWorkloadRequest.self, from: Data(contentsOf: request))
        guard [1024, 8192].contains(q.rows), ["cpu", "neural"].contains(q.policy),
              [32, 256].contains(q.batches), q.quantum == 8 else { throw BenchmarkFailure("Invalid persistent trial request") }
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        let model = try await MLModel.compileModel(at: URL(fileURLWithPath: q.model))
        defer { try? FileManager.default.removeItem(at: model) }
        let fixture = try await MixedWorkloadFixture.make(q, file: file, model: model)
        var undersized = false
        do { _ = try PersistentTrialScope(configuration: fixture.configuration, totalBytes: fixture.configuration.requiredBytes - 1) }
        catch is MemoryAdmissionError { undersized = true }
        guard undersized else { throw BenchmarkFailure("Impossible workspace partition accepted") }
        let baseline = try await mixedSample(q, fixture: fixture, mode: "isolated", iteration: -1)
        let reference = baseline.lanes.first { $0.lane == "coreml" }!.hashes
        let variants: [(String, Int)] = [("reload", 8), ("whole-run", q.batches), ("persistent", 8), ("persistent", 32)]
        let modes = ["tight", "balanced", "wide"]
        var samples: [PersistentTrialSample] = []
        for iteration in -1..<3 {
            for m in modes.indices {
                let mode = modes[(m + iteration + 1) % modes.count]
                for v in variants.indices {
                    let (variant, quantum) = variants[(v + iteration + 1) % variants.count]
                    let request = persistentRequest(q, quantum: quantum)
                    let measured: PersistentTrialSample
                    if variant == "persistent" {
                        measured = try await persistentTrialSample(request, fixture: fixture, mode: mode, iteration: iteration)
                    } else {
                        let sample = try await mixedSample(request, fixture: fixture, mode: mode, iteration: iteration)
                        measured = PersistentTrialSample(variant: variant, quantum: quantum,
                            poolScopes: q.batches / quantum, startupSeconds: nil, retainedBytes: 0,
                            workBudgetBytes: sample.budgetBytes, sample: sample)
                    }
                    guard measured.sample.lanes.first(where: { $0.lane == "coreml" })?.hashes == reference else {
                        throw BenchmarkFailure("Persistent trial changed predictions")
                    }
                    if iteration >= 0 { samples.append(measured) }
                }
            }
        }
        var cancellations: [PersistentTrialSample] = []
        for quantum in [8, 32] {
            let request = persistentRequest(q, quantum: quantum)
            let task = Task { try await persistentTrialSample(request, fixture: fixture,
                mode: "tight", iteration: 0, cancel: true) }
            cancellations.append(try await task.value)
        }
        let queued = try await persistentQueuedCancellation(q, fixture: fixture)
        let record = PersistentTrialRecord(samples: samples, cancellation: cancellations,
            queuedCancellation: queued, undersizedPartitionRejected: undersized)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(record).write(to: output, options: .atomic)
        return 0
    } catch { fputs("Persistent model trial failed: \(error)\n", stderr); return 1 }
}
