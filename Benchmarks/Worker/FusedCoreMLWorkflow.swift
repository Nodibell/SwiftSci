import CoreML
import Foundation
import SwiftDataFrame
import SwiftML
import SwiftPreprocessing
import SwiftSciBenchmarkSupport

struct FusionCoreMLRequest: Decodable {
    let model: String
    let fixture: String
    let rawFixture: String
    let rows: Int
    let callers: Int
    let slots: Int
    let policy: String
}
struct FusionRawFixture: Decodable {
    let trainingColumns: [[Double]]
    let heldOutColumns: [[Double]]
    let sourceSHA256: String
    let selectedRows: [Int]
}
struct FusionCoreMLSample: Encodable, Sendable {
    let mode: String
    let iteration: Int
    let wallSeconds: Double
    let preparationSeconds: [Double]
    let predictionSeconds: [Double]
    let residentBytes: UInt64
    let thermalState: Int
}
struct FusionCoreMLRecord: Encodable {
    let scope = "Covertype training-only Swift preprocessing and frozen Core ML model; Float16 transport"
    let modelSHA256: String
    let sourceSHA256: String
    let rows: Int
    let callers: Int
    let slots: Int
    let policy: String
    let samples: [FusionCoreMLSample]
    let referenceHashes: [String]
    let finalPoolReservedBytes: Int
    let finalInputReservedBytes: Int
    let poolReleased: Bool
}
struct FusionPrediction: Sendable {
    let output: CoreMLPrediction
    let preparation: Double
    let prediction: Double
    let caller: Int
}

func fusionPredict(mode: String, caller: Int, plan: FusedPreprocessingPlan,
    input: consuming PreparedNumericBatch, pool: CoreMLMatrixPool, budget: MemoryBudget,
    contract: CoreMLTrialInputContract) async throws -> FusionPrediction {
    let start = ContinuousClock.now
    let prepared: CoreMLPreparedMatrix
    if mode == "fused-direct" {
        prepared = try await contract.prepareFused(input, plan: plan, budget: budget)
    } else if mode == "native-owned" {
        let imputed = try plan.imputer.transform(consuming: consume input)
        let scaled = try plan.scaler.transform(consuming: consume imputed)
        prepared = try await pool.prepare(scaled, budget: budget)
    } else if mode == "native" {
        let imputed = try plan.imputer.transform(input)
        let scaled = try plan.scaler.transform(consuming: consume imputed)
        prepared = try await pool.prepare(scaled, budget: budget)
    } else {
        let values: [Float16]
        if mode == "staged-packed" { values = try plan.staged(input, as: Float16.self) }
        else { values = try plan.fused(input, as: Float16.self, tiled: true) }
        prepared = try await pool.preparePackedForTrial(values, source: input, budget: budget)
    }
    let preparation = elapsedSeconds(since: start)
    let predictionStart = ContinuousClock.now
    let output = try await pool.predict(prepared)
    return FusionPrediction(output: output, preparation: preparation,
        prediction: elapsedSeconds(since: predictionStart), caller: caller)
}

func awaitFusionRelease(_ budget: MemoryBudget) async throws {
    let deadline = ContinuousClock.now.advanced(by: .seconds(5))
    while await budget.reservedBytes != 0 {
        guard ContinuousClock.now < deadline else { throw BenchmarkFailure("Fusion input reservation did not drain") }
        await Task.yield()
    }
}

func measureFusionCoreML(_ q: FusionCoreMLRequest) async throws -> FusionCoreMLRecord {
    guard [1024, 8192].contains(q.rows), [1, 4].contains(q.callers), q.slots > 0, q.slots <= 2,
          q.slots <= q.callers, ["cpu", "neural"].contains(q.policy) else {
        throw BenchmarkFailure("Invalid fused Core ML bounds")
    }
    let raw = try JSONDecoder().decode(FusionRawFixture.self, from: Data(contentsOf: URL(fileURLWithPath: q.rawFixture)))
    let fixture = try CoreMLQualificationFixture(contentsOf: URL(fileURLWithPath: q.fixture), name: "trained", rows: q.rows)
    guard raw.trainingColumns.count == fixture.width, raw.trainingColumns.allSatisfy({ $0.count == 11340 && $0.allSatisfy(\.isFinite) }),
          raw.heldOutColumns.count == fixture.width, raw.heldOutColumns.allSatisfy({ $0.count == 8192 && $0.allSatisfy(\.isFinite) }),
          raw.selectedRows.count == 8192, Set(raw.selectedRows).count == 8192,
          raw.selectedRows.allSatisfy({ $0 >= 15120 && $0 < 581012 }) else {
        throw BenchmarkFailure("Covertype training and held-out data contract differs")
    }
    let training = try PreparedNumericBatch(columnNames: fixture.names, columns: raw.trainingColumns)
    let plan = try FusedPreprocessingPlan(training: training)
    let all = try PreparedNumericBatch(columnNames: fixture.names, columns: raw.heldOutColumns)
    let inputs: [PreparedNumericBatch] = try (0..<q.callers).map { caller in
        let indices: [Int] = (0..<q.rows).map { (q.rows - 1 - $0 + caller * 17) % q.rows }
        return try all.selectingRows(indices)
    }
    // Verify packed values before model loading or timing. The model's own Float16
    // error is independent of whether preprocessing was staged or fused.
    for input in inputs {
        guard fusedTrialEqual(try plan.staged(input, as: Float16.self),
                              try plan.fused(input, as: Float16.self, tiled: true)) else {
            throw BenchmarkFailure("Real-data packed inputs differ")
        }
    }
    let url = URL(fileURLWithPath: q.model)
    let hash = try coreMLPackageHash(url)
    let compiled = try await MLModel.compileModel(at: url)
    defer { try? FileManager.default.removeItem(at: compiled) }
    let poolBudget = try MemoryBudget(limit: 100_663_296)
    let inputBudget = try MemoryBudget(limit: 16_777_216)
    let config = try CoreMLMatrixPool.Configuration(maximumConcurrentPredictions: q.slots,
        retainedBytes: 33_554_432, requestBytes: 16_777_216)
    let measured = try await CoreMLMatrixPool.withPool(compiledModelURL: compiled, inputColumns: fixture.names,
        outputName: "result", computeUnits: q.policy == "cpu" ? .cpuOnly : .cpuAndNeuralEngine,
        budget: poolBudget, configuration: config) { pool in
        let contract = try await pool.inputContractForTrial()
        var hashes = [String]()
        for (caller, input) in inputs.enumerated() {
            let reference = try await fusionPredict(mode: "native", caller: caller, plan: plan, input: input, pool: pool, budget: inputBudget, contract: contract)
            hashes.append(try coreMLResultHash(reference.output.values))
        }
        try await awaitFusionRelease(inputBudget)
        var samples = [FusionCoreMLSample]()
        let modes = ["native", "staged-packed", "fused-packed", "fused-direct"]
            + (q.callers == 1 ? ["native-owned"] : [])
        for iteration in -1..<5 {
            for index in 0..<modes.count {
                let mode = modes[(index + iteration + 1) % modes.count]
                let (wall, outputs) = try await measureFusionCallers(mode: mode, plan: plan, inputs: inputs,
                    pool: pool, budget: inputBudget, contract: contract)
                guard outputs.count == q.callers else { throw BenchmarkFailure("Missing fusion caller result") }
                for result in outputs {
                    guard result.output.values.originalRowIndices == inputs[result.caller].originalRowIndices,
                          try coreMLResultHash(result.output.values) == hashes[result.caller] else {
                        throw BenchmarkFailure("Fused workflow changed prediction or row identity")
                    }
                }
                let rss = try residentBytes()
                guard rss < 1_073_741_824 else { throw BenchmarkFailure("Fusion workflow RSS cutoff") }
                if iteration >= 0 {
                    samples.append(FusionCoreMLSample(mode: mode, iteration: iteration, wallSeconds: wall,
                        preparationSeconds: outputs.map(\.preparation), predictionSeconds: outputs.map(\.prediction),
                        residentBytes: rss, thermalState: ProcessInfo.processInfo.thermalState.rawValue))
                }
            }
        }
        return (samples, hashes, WeakPredictor(pool))
    }
    let poolReserved = await poolBudget.reservedBytes
    let inputReserved = await inputBudget.reservedBytes
    guard poolReserved == 0, inputReserved == 0, measured.2.value == nil else {
        throw BenchmarkFailure("Fusion workflow retained an owner or reservation")
    }
    return FusionCoreMLRecord(modelSHA256: hash, sourceSHA256: raw.sourceSHA256, rows: q.rows,
        callers: q.callers, slots: q.slots, policy: q.policy, samples: measured.0,
        referenceHashes: measured.1, finalPoolReservedBytes: poolReserved,
        finalInputReservedBytes: inputReserved, poolReleased: true)
}

func fusedCoreMLWorkflow(request: URL, output: URL) async -> Int32 {
    do {
        let q = try JSONDecoder().decode(FusionCoreMLRequest.self, from: Data(contentsOf: request))
        let record = try await measureFusionCoreML(q)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(record).write(to: output, options: .atomic)
        return 0
    } catch {
        fputs("Fused Core ML workflow failed: \(error)\n", stderr)
        return 1
    }
}
