import CoreML
import CryptoKit
import Foundation
import SwiftSciBenchmarkSupport
import SwiftDataFrame
import SwiftML
import SwiftPreprocessing

struct BoundedCoreMLRequest: Decodable {
    let model: String
    let rawFixture: String
    let rows: Int
    let policy: String
    let batches: Int
    let consumerDelayMilliseconds: Int
    let variant: Int?
}
struct BoundedCoreMLSample: Encodable {
    let mode: String
    let window: Int
    let slots: Int
    let iteration: Int
    let wallSeconds: Double
    let firstResultSeconds: Double
    let modelReadySeconds: Double
    let steadyStateSeconds: Double
    let steadyStateBatches: Int
    let checkpoints: [BoundedMemoryCheckpoint]
    let peakReservedBytes: Int
    let sampledResidentBytes: UInt64
    let maximumOutstanding: Int
    let thermalState: Int
}
struct BoundedCoreMLRecord: Encodable {
    let modelSHA256: String
    let rawSHA256: String
    let sourceSHA256: String
    let rows: Int
    let batches: Int
    let policy: String
    let consumerDelayMilliseconds: Int
    let samples: [BoundedCoreMLSample]
    let referenceHashes: [String]
}

private func boundedReference(pool: CoreMLMatrixPool, source: BoundedCoreMLSource,
    preparation: CoreMLMatrixInputPreparation, plan: StandardPreprocessingPlan,
    budget: MemoryBudget, sequence: Int) async throws {
    guard let input = try await source.load(sequence) else { throw BenchmarkFailure("Missing reference batch") }
    let owner = try await preparation.prepare(input, preprocessing: plan, budget: budget)
    let prediction = try await pool.predict(owner)
    try await source.consume(sequence, prediction: prediction)
}

private func boundedSample(_ q: BoundedCoreMLRequest, model: URL, file: URL,
    plan: StandardPreprocessingPlan, window: Int, slots: Int, reference: Bool,
    iteration: Int) async throws -> (BoundedCoreMLSample, [String]) {
    let pool = try CoreMLMatrixPool.Configuration(maximumConcurrentPredictions: slots,
        retainedBytes: 33_554_432, requestBytes: 16_777_216)
    let config = try CoreMLBatchPipeline.Configuration(pool: pool, maximumInFlightBatches: window,
        retainedBytes: 1_048_576, batchBytes: 33_554_432)
    let budget = try MemoryBudget(limit: config.requiredBytes)
    let start = ContinuousClock.now
    let source = try BoundedCoreMLSource(url: file, names: plan.columnNames,
        rows: q.rows, batches: q.batches, delayMilliseconds: q.consumerDelayMilliseconds)
    do {
        if reference {
            let inputBudget = try MemoryBudget(limit: config.batchBytes)
            try await CoreMLMatrixPool.withPool(compiledModelURL: model, inputColumns: plan.columnNames,
                outputName: "result", computeUnits: q.policy == "cpu" ? .cpuOnly : .cpuAndNeuralEngine,
                budget: budget, configuration: pool) { pool in
                let preparation = try await pool.inputPreparation()
                for sequence in 0..<q.batches {
                    try await boundedReference(pool: pool, source: source, preparation: preparation,
                        plan: plan, budget: inputBudget, sequence: sequence)
                    try await awaitFusionRelease(inputBudget)
                }
            }
        } else {
            try await CoreMLBatchPipeline.run(compiledModelURL: model, inputColumns: plan.columnNames,
                outputName: "result", computeUnits: q.policy == "cpu" ? .cpuOnly : .cpuAndNeuralEngine,
                preprocessing: plan, budget: budget, configuration: config,
                load: { request in try await source.load(request.sequence) },
                consume: { sequence, prediction in try await source.consume(sequence, prediction: prediction) })
        }
        try await source.close()
    } catch { try? await source.close(); throw error }
    let wall = elapsedSeconds(since: start)
    guard await budget.reservedBytes == 0, await source.consumed == q.batches,
          await source.maximumOutstanding <= window else { throw BenchmarkFailure("Bounded workflow did not drain") }
    let sample = await BoundedCoreMLSample(mode: reference ? "serial-reference" : "pipeline",
        window: window, slots: slots, iteration: iteration, wallSeconds: wall,
        firstResultSeconds: source.firstResultSeconds, modelReadySeconds: source.firstLoadSeconds,
        steadyStateSeconds: source.lastResultSeconds - source.steadyStartSeconds,
        steadyStateBatches: q.batches - source.warmupBatches,
        checkpoints: source.checkpoints, peakReservedBytes: budget.peak,
        sampledResidentBytes: source.sampledResidentBytes, maximumOutstanding: source.maximumOutstanding,
        thermalState: ProcessInfo.processInfo.thermalState.rawValue)
    return (sample, await source.hashes)
}

func prepareBoundedFixture(_ q: BoundedCoreMLRequest, file: URL) throws
    -> (StandardPreprocessingPlan, String, String) {
        let rawData = try Data(contentsOf: URL(fileURLWithPath: q.rawFixture))
        let raw = try JSONDecoder().decode(FusionRawFixture.self, from: rawData)
        guard raw.trainingColumns.count == 54, raw.trainingColumns.allSatisfy({ $0.count == 11340 && $0.allSatisfy(\.isFinite) }),
              raw.heldOutColumns.count == 54, raw.heldOutColumns.allSatisfy({ $0.count == 8192 && $0.allSatisfy(\.isFinite) }) else {
            throw BenchmarkFailure("Covertype fixture dimensions differ")
        }
        let names = (0..<54).map { "feature_\($0)" }
        let training = try PreparedNumericBatch(columnNames: names, columns: raw.trainingColumns)
        let plan = try StandardPreprocessingPlan(training: training)
        try writeBoundedCoreMLSource(raw, rows: q.rows, batches: q.batches, to: file)
        let hash = SHA256.hash(data: rawData).map { String(format: "%02x", $0) }.joined()
        return (plan, hash, raw.sourceSHA256)
}

func boundedCoreMLWorkflow(request: URL, output: URL) async -> Int32 {
    do {
        let q = try JSONDecoder().decode(BoundedCoreMLRequest.self, from: Data(contentsOf: request))
        guard [1024, 8192].contains(q.rows), ["cpu", "neural"].contains(q.policy),
              q.batches > 0, q.batches <= 4096, (q.variant == nil || (0..<5).contains(q.variant!)), (0...100).contains(q.consumerDelayMilliseconds) else {
            throw BenchmarkFailure("Invalid bounded workflow request")
        }
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        let (plan, rawHash, sourceHash) = try prepareBoundedFixture(q, file: file)
        let url = URL(fileURLWithPath: q.model)
        let modelHash = try coreMLPackageHash(url)
        let compiled = try await MLModel.compileModel(at: url)
        defer { try? FileManager.default.removeItem(at: compiled) }
        let allVariants = [(1, 1, true), (1, 1, false), (2, 1, false), (2, 2, false), (4, 2, false)]
        let variants = q.variant.map { [allVariants[$0]] } ?? allVariants
        var samples: [BoundedCoreMLSample] = []
        var reference: [String] = []
        if q.variant != nil {
            let (_, hashes) = try await boundedSample(q, model: compiled, file: file, plan: plan,
                window: 1, slots: 1, reference: true, iteration: -2)
            reference = hashes
        }
        for iteration in -1..<3 {
            for offset in variants.indices {
                let (window, slots, baseline) = variants[(offset + iteration + 1) % variants.count]
                let (sample, hashes) = try await boundedSample(q, model: compiled, file: file, plan: plan,
                    window: window, slots: slots, reference: baseline, iteration: iteration)
                if reference.isEmpty { reference = hashes }
                guard hashes == reference else { throw BenchmarkFailure("Pipeline predictions differ from direct reference") }
                if iteration >= 0 { samples.append(sample) }
            }
        }
        let record = BoundedCoreMLRecord(modelSHA256: modelHash, rawSHA256: rawHash,
            sourceSHA256: sourceHash, rows: q.rows, batches: q.batches, policy: q.policy,
            consumerDelayMilliseconds: q.consumerDelayMilliseconds, samples: samples, referenceHashes: reference)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(record).write(to: output, options: .atomic)
        return 0
    } catch { fputs("Bounded Core ML workflow failed: \(error)\n", stderr); return 1 }
}
