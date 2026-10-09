import CoreML
import Foundation
import SwiftDataFrame
import SwiftML
import SwiftPreprocessing
import SwiftSciBenchmarkSupport

struct PreparedWorkflowCase: Codable, Sendable {
    let mode: String
    let reuse: Int
}
struct PreparedWorkflowRequest: Codable, Sendable {
    let model: String
    let fixture: String
    let rows: Int
    let policy: String
    let callers: Int
    let slots: Int
    let schedule: [PreparedWorkflowCase]

    func validate() throws {
        guard [1024, 8192].contains(rows), ["cpu", "neural"].contains(policy),
              [1, 4].contains(callers), (1...4).contains(slots), slots <= callers,
              !schedule.isEmpty, schedule.count <= 200,
              schedule.allSatisfy({ ["direct", "retained"].contains($0.mode) && [1, 2, 4, 16, 64].contains($0.reuse) }) else {
            throw BenchmarkFailure("Invalid prepared workflow bounds")
        }
    }
}
struct PreparedWorkflowSample: Codable {
    let mode: String
    let reuse: Int
    let preparationSeconds: Double
    let predictionSeconds: Double
    let totalSeconds: Double
    let inputReservedBytes: Int
    var releaseSeconds = 0.0
    var residentBytes: UInt64 = 0
    var thermalState = 0
}
struct PreparedWorkflowRecord: Codable {
    let request: PreparedWorkflowRequest
    let modelSHA256: String
    let transportElementBytes: Int
    let samples: [PreparedWorkflowSample]
    let outputHash: String
    let strictReferenceMismatches: Int
    let maxAbsoluteError: Double
    let poolReservationBytes: Int
    let finalPoolReservedBytes: Int
    let finalInputReservedBytes: Int
    let poolReleased: Bool
}

private func workflowPredictions(_ q: PreparedWorkflowRequest, trial: PreparedWorkflowCase,
    input: PreparedNumericBatch, pool: CoreMLMatrixPool,
    budget: MemoryBudget) async throws -> (PreparedWorkflowSample, [CoreMLPrediction]) {
    let start = ContinuousClock.now
    let retained = trial.mode == "retained" ? try await pool.prepare(input, budget: budget) : nil
    let preparation = elapsedSeconds(since: start)
    let predictionStart = ContinuousClock.now
    let outputs = try await withThrowingTaskGroup(of: [CoreMLPrediction].self) { group in
        for caller in 0..<min(q.callers, trial.reuse) {
            group.addTask {
                var results: [CoreMLPrediction] = []
                for _ in stride(from: caller, to: trial.reuse, by: q.callers) {
                    let result: CoreMLPrediction
                    if let retained { result = try await pool.predict(retained) }
                    else { result = try await pool.predict(input) }
                    results.append(result)
                }
                return results
            }
        }
        var results: [CoreMLPrediction] = []
        for try await batch in group { results.append(contentsOf: batch) }
        return results
    }
    let prediction = elapsedSeconds(since: predictionStart)
    let total = elapsedSeconds(since: start)
    return (PreparedWorkflowSample(mode: trial.mode, reuse: trial.reuse, preparationSeconds: preparation,
        predictionSeconds: prediction, totalSeconds: total, inputReservedBytes: retained?.reservedBytes ?? 0), outputs)
}

private func awaitInputRelease(_ budget: MemoryBudget) async throws {
    let deadline = ContinuousClock.now.advanced(by: .seconds(5))
    while await budget.reservedBytes != 0 {
        guard ContinuousClock.now < deadline else { throw BenchmarkFailure("Prepared input reservation did not drain") }
        await Task.yield()
    }
}

private func measurePreparedWorkflow(_ q: PreparedWorkflowRequest) async throws -> PreparedWorkflowRecord {
    try q.validate()
    let fixture = try CoreMLQualificationFixture(contentsOf: URL(fileURLWithPath: q.fixture), name: "trained", rows: q.rows)
    let mapping = (0..<q.rows).map { q.rows - 1 - ($0 / 2) * 2 }
    let input = try PreparedNumericBatch(columnNames: Array(fixture.names.reversed()), columns: Array(fixture.columns.reversed()))
        .selectingRows(mapping)
    let url = URL(fileURLWithPath: q.model)
    let hash = try coreMLPackageHash(url)
    let compiled = try await MLModel.compileModel(at: url)
    defer { try? FileManager.default.removeItem(at: compiled) }
    let bytes = try programMatrixElementBytes(at: compiled, rows: q.rows, width: fixture.width, outputs: fixture.outputWidth)
    let requestBytes = input.payloadByteCount + q.rows * fixture.outputWidth * (8 + bytes) + 8_388_608
    let quota = 33_554_432 + q.slots * requestBytes
    let budget = try MemoryBudget(limit: quota)
    let inputBudget = try MemoryBudget(limit: 16_777_216)
    let configuration = try CoreMLMatrixPool.Configuration(maximumConcurrentPredictions: q.slots,
        retainedBytes: 33_554_432, requestBytes: requestBytes)
    let result = try await CoreMLMatrixPool.withPool(compiledModelURL: compiled, inputColumns: fixture.names,
        outputName: "result", computeUnits: q.policy == "cpu" ? .cpuOnly : .cpuAndNeuralEngine,
        budget: budget, configuration: configuration) { pool in
        let reference = try await pool.predict(input)
        let referenceHash = try coreMLResultHash(reference.values)
        var mismatches = 0, maximumError = 0.0
        for row in 0..<q.rows {
            for column in 0..<fixture.outputWidth {
                let expected = fixture.expected[mapping[row] * fixture.outputWidth + column]
                let error = abs(reference.values[row, column]! - expected)
                maximumError = max(maximumError, error)
                if error > 1e-5 + abs(expected) * 1e-5 { mismatches += 1 }
            }
        }
        var samples: [PreparedWorkflowSample] = []
        let warmup = [PreparedWorkflowCase(mode: "direct", reuse: 4), PreparedWorkflowCase(mode: "retained", reuse: 4)]
        for (index, trial) in (warmup + q.schedule).enumerated() {
            var (sample, outputs) = try await workflowPredictions(q, trial: trial, input: input, pool: pool, budget: inputBudget)
            let releaseStart = ContinuousClock.now
            try await awaitInputRelease(inputBudget)
            sample.releaseSeconds = elapsedSeconds(since: releaseStart)
            guard outputs.count == trial.reuse else { throw BenchmarkFailure("Incomplete workflow") }
            for output in outputs {
                guard output.values.originalRowIndices == mapping,
                      output.values.columnNames == reference.values.columnNames,
                      try coreMLResultHash(output.values) == referenceHash else {
                    throw BenchmarkFailure("Prepared output or identity differs from direct prediction")
                }
            }
            sample.residentBytes = try residentBytes()
            sample.thermalState = ProcessInfo.processInfo.thermalState.rawValue
            guard sample.residentBytes < 1_073_741_824 else { throw BenchmarkFailure("Prepared workflow RSS cutoff") }
            if index >= warmup.count { samples.append(sample) }
        }
        return (samples, referenceHash, mismatches, maximumError, WeakPredictor(pool))
    }
    let reserved = await budget.reservedBytes
    let inputReserved = await inputBudget.reservedBytes
    guard reserved == 0, inputReserved == 0, result.4.value == nil else {
        throw BenchmarkFailure("Prepared workflow retained an owner or reservation")
    }
    return PreparedWorkflowRecord(request: q, modelSHA256: hash, transportElementBytes: bytes,
        samples: result.0, outputHash: result.1, strictReferenceMismatches: result.2, maxAbsoluteError: result.3,
        poolReservationBytes: quota, finalPoolReservedBytes: reserved, finalInputReservedBytes: inputReserved,
        poolReleased: result.4.value == nil)
}

func preparedCoreMLWorkflow(request: URL, output: URL) async -> Int32 {
    do {
        let q = try JSONDecoder().decode(PreparedWorkflowRequest.self, from: Data(contentsOf: request))
        let record = try await measurePreparedWorkflow(q)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(record).write(to: output)
        return 0
    } catch {
        FileHandle.standardError.write(Data("Prepared Core ML workflow failed: \(error)\n".utf8))
        return 1
    }
}
