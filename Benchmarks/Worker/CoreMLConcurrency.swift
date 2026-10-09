import CoreML
import CryptoKit
import Foundation
import os
import SwiftDataFrame
import SwiftML
import SwiftPreprocessing
import SwiftSciBenchmarkSupport

struct StressRequest: Codable, Sendable {
    let modelPackage: String?
    let retainedBytes: Int?
    let execution: String?
    let fixture: String
    let rows: Int
    let policy: String
    let callers: Int
    let instances: Int
    let admissionSlots: Int
    let seconds: Int
    let rssLimitBytes: UInt64

    func validate() throws {
        guard execution == nil || ["synchronous", "asynchronous", "persistent"].contains(execution!),
              (1...8192).contains(rows), ["cpu", "neural"].contains(policy),
              (1...4).contains(callers), instances == 1 || instances == callers,
              (1...4).contains(admissionSlots), (1...1800).contains(seconds),
              (268_435_456...2_147_483_648).contains(rssLimitBytes) else {
            throw BenchmarkFailure("Invalid concurrency request or safety bounds")
        }
        if execution == "persistent" {
            guard instances == 1, let retainedBytes, (16_777_216...268_435_456).contains(retainedBytes) else {
                throw BenchmarkFailure("Persistent trial requires one model and a bounded retained allowance")
            }
        } else if retainedBytes != nil {
            throw BenchmarkFailure("Fresh-buffer trials do not reserve model lifetime capacity")
        }
    }
}

struct CallerResult: Encodable, Sendable {
    let caller: Int
    let latency: CoreMLLatency
    let first: PreparedNumericBatch
    let last: PreparedNumericBatch
    let firstVariant: Int
    let lastVariant: Int
    enum CodingKeys: String, CodingKey { case caller, latency, firstVariant, lastVariant }
}

// Written at construction; read only after all prediction tasks have completed.
final class WeakPredictor: @unchecked Sendable {
    weak var value: AnyObject?
    init(_ value: AnyObject) { self.value = value }
}

private enum StressEvent: Sendable {
    case caller(CallerResult)
    case memory([CoreMLStressMemory])
}

struct PoolResult: Sendable {
    let callers: [CallerResult]
    let owners: [WeakPredictor]
    let hashes: [String]
    let memory: [CoreMLStressMemory]
    let elapsed: Double
    let peak: Int
    let reserved: Int
    let queued: Int
    let cancellationChecks: Int
    let preflightReferenceMismatches: Int
    let beforePoolRSS: UInt64
    var poolPredictions: Int? = nil
    var poolOutputBackingIdentityMatches: Int? = nil
}

private struct StressRecord: Encodable {
    let request: StressRequest
    let modelSHA256: String
    let plannedDevices: [String]
    let executionTraceVerified = false
    let latencyIncludesActorQueueAdmissionPackingPredictionAndOutputCopy = true
    let validationOutsideRequestLatency = true
    let throughputIncludesValidation = true
    let transportElementBytes: Int
    let requestReservationBytes: Int
    let budgetLimitBytes: Int
    let elapsedSeconds: Double
    let callers: [CallerResult]
    let memory: [CoreMLStressMemory]
    let budgetPeakBytes: Int
    let finalReservedBytes: Int
    let finalQueuedCount: Int
    let cancelledRequestsVerified: Int
    let preflightReferenceMismatches: Int
    let outputHashes: [String]
    let predictorsReleased: Bool
    let retainedOutputsVerified: Bool
    let beforePoolRSS: UInt64
    let afterPoolRSS: UInt64
    let poolPredictions: Int?
    let poolOutputBackingIdentityMatches: Int?
}

private func cancellationProbe(predictors: [CoreMLPredictor], input: PreparedNumericBatch,
                               budget: MemoryBudget, limit: Int) async throws -> Int {
    let blocker = try await budget.acquire(MemoryEstimate(capacities: [limit]))
    var requests: [Task<CoreMLPrediction, any Error>] = []
    for i in 0..<4 {
        let predictor = predictors[i % predictors.count]
        requests.append(Task { try await predictor.predict(input, budget: budget, workspaceBytes: 8_388_608) })
    }
    let deadline = ContinuousClock.now.advanced(by: .seconds(5))
    while await budget.queuedCount < 4, ContinuousClock.now < deadline {
        try? await Task.sleep(for: .milliseconds(1))
    }
    let allQueued = await budget.queuedCount == 4
    for i in requests.indices where !allQueued || i < 2 { requests[i].cancel() }
    await blocker.finish()
    var cancellations = 0
    var unexpected = false
    for (i, request) in requests.enumerated() {
        do {
            _ = try await request.value
            if i < 2 { unexpected = true }
        } catch is CancellationError {
            cancellations += 1
            if i >= 2 { unexpected = true }
        } catch { unexpected = true }
    }
    let reserved = await budget.reservedBytes
    let queued = await budget.queuedCount
    guard allQueued, !unexpected, cancellations == 2, reserved == 0, queued == 0 else {
        throw BenchmarkFailure("Concurrent cancellation did not drain admission")
    }
    return cancellations
}

private func runCaller(id: Int, predictor: StressPredictor, inputs: [PreparedNumericBatch],
                       hashes: [String], budget: MemoryBudget,
                       deadline: ContinuousClock.Instant) async throws -> CallerResult {
    var latency = CoreMLLatency()
    var first: PreparedNumericBatch?
    var last: PreparedNumericBatch?
    let firstVariant = id % 2
    var lastVariant = firstVariant
    repeat {
        try Task.checkCancellation()
        let variant = (latency.count + id) % 2
        let start = ContinuousClock.now
        let result = try await predictor.predict(inputs[variant], budget: budget, workspaceBytes: 8_388_608)
        let elapsed = elapsedSeconds(since: start)
        guard try coreMLResultHash(result.values) == hashes[variant],
              result.values.originalRowIndices == inputs[variant].originalRowIndices else {
            throw BenchmarkFailure("Concurrent prediction changed output or row identity")
        }
        latency.record(elapsed)
        if first == nil { first = result.values }
        last = result.values
        lastVariant = variant
    } while ContinuousClock.now < deadline
    return CallerResult(caller: id, latency: latency, first: first!, last: last!,
                        firstVariant: firstVariant, lastVariant: lastVariant)
}

private func runPool(_ request: StressRequest, fixture: CoreMLQualificationFixture,
                     compiled: URL, estimate: Int) async throws -> PoolResult {
    let beforePoolRSS = try residentBytes()
    let units: MLComputeUnits = request.policy == "cpu" ? .cpuOnly : .cpuAndNeuralEngine
    let limit = estimate * request.admissionSlots
    let budget = try MemoryBudget(limit: limit)
    let original = try PreparedNumericBatch(columnNames: fixture.names, columns: fixture.columns)
    let reversed = try original.selectingRows(Array((0..<request.rows).reversed()))
    let inputs = [original, reversed]
    var predictors: [CoreMLPredictor] = []
    for _ in 0..<request.instances {
        predictors.append(try CoreMLPredictor(compiledModelURL: compiled, inputColumns: fixture.names,
            outputName: "result", computeUnits: units, asynchronousMatrix: request.execution == "asynchronous"))
    }
    let owners = predictors.map { WeakPredictor($0) }
    var hashes: [String] = []
    var mismatches = 0
    for (index, predictor) in predictors.enumerated() {
        for variant in 0..<2 {
            let result = try await predictor.predict(inputs[variant], budget: budget, workspaceBytes: 8_388_608)
            let hash = try coreMLResultHash(result.values)
            if index == 0 { hashes.append(hash) }
            else if hash != hashes[variant] { throw BenchmarkFailure("Model instances disagree before stress") }
            if index == 0 && variant == 0 {
                for row in 0..<request.rows {
                    for column in 0..<fixture.outputWidth {
                        let expected = fixture.expected[row * fixture.outputWidth + column]
                        let actual = result.values[row, column]!
                        if abs(actual - expected) > 1e-5 + 1e-5 * abs(expected) { mismatches += 1 }
                    }
                }
            }
        }
    }
    guard hashes[0] != hashes[1] else { throw BenchmarkFailure("Stress input variants must differ") }
    let probeBudget = try MemoryBudget(limit: limit)
    let cancelled = try await cancellationProbe(predictors: predictors, input: original, budget: probeBudget, limit: limit)
    var bad = original
    try bad.updateColumn(at: 0, rows: [request.rows - 1], values: [.nan])
    do {
        _ = try await predictors[0].predict(bad, budget: budget, workspaceBytes: 8_388_608)
        throw BenchmarkFailure("Nonfinite request unexpectedly succeeded")
    } catch is SwiftMLError { }
    guard await budget.reservedBytes == 0 else { throw BenchmarkFailure("Invalid input leaked admission") }
    return try await measurePredictions(request, predictors: predictors.map { .fresh($0) },
        inputs: inputs, hashes: hashes, budget: budget, owners: owners, cancelled: cancelled,
        mismatches: mismatches, beforePoolRSS: beforePoolRSS, expectedReservation: 0)
}

func measurePredictions(_ request: StressRequest, predictors: [StressPredictor],
    inputs: [PreparedNumericBatch], hashes: [String], budget: MemoryBudget,
    owners: [WeakPredictor], cancelled: Int, mismatches: Int, beforePoolRSS: UInt64,
    expectedReservation: Int) async throws -> PoolResult {
    let signposter = OSSignposter(subsystem: "org.swiftsci.benchmarks", category: "PointsOfInterest")
    let interval = signposter.beginInterval("CoreML measurement")
    defer { signposter.endInterval("CoreML measurement", interval) }
    let limit = await budget.limit
    let referenceHashes = hashes
    let start = ContinuousClock.now
    let deadline = start.advanced(by: .seconds(request.seconds))
    var memory: [CoreMLStressMemory] = []
    var completed: [CallerResult] = []
    try await withThrowingTaskGroup(of: StressEvent.self) { group in
        for id in 0..<request.callers {
            let predictor = predictors[id % predictors.count]
            group.addTask {
                let result = try await runCaller(id: id, predictor: predictor, inputs: inputs,
                    hashes: referenceHashes, budget: budget, deadline: deadline)
                return .caller(result)
            }
        }
        // One observer; fixed duration bounds telemetry to at most 1,802 records.
        group.addTask {
            var points: [CoreMLStressMemory] = []
            repeat {
                let rss = try residentBytes()
                if rss > request.rssLimitBytes { throw BenchmarkFailure("RSS safety limit exceeded") }
                let point = CoreMLStressMemory(seconds: elapsedSeconds(since: start), residentBytes: rss,
                    reservedBytes: await budget.reservedBytes, queuedCount: await budget.queuedCount,
                    thermalState: ProcessInfo.processInfo.thermalState.rawValue)
                points.append(point)
                try await Task.sleep(for: .seconds(1))
            } while ContinuousClock.now < deadline
            return .memory(points)
        }
        for try await event in group {
            switch event {
            case .caller(let result): completed.append(result)
            case .memory(let points): memory = points
            }
        }
    }
    guard completed.count == request.callers else { throw BenchmarkFailure("Caller did not complete") }
    let reserved = await budget.reservedBytes
    let queued = await budget.queuedCount
    let peak = await budget.peak
    guard reserved == expectedReservation, queued == 0, peak <= limit else { throw BenchmarkFailure("Admission accounting failed") }
    return PoolResult(callers: completed.sorted { $0.caller < $1.caller }, owners: owners, hashes: hashes,
        memory: memory, elapsed: elapsedSeconds(since: start), peak: peak, reserved: reserved, queued: queued,
        cancellationChecks: cancelled, preflightReferenceMismatches: mismatches, beforePoolRSS: beforePoolRSS)
}

func stressCoreML(request url: URL, output: URL) async -> Int32 {
    do {
        let request = try JSONDecoder().decode(StressRequest.self, from: Data(contentsOf: url))
        try request.validate()
        let fixture = try CoreMLQualificationFixture(contentsOf: URL(fileURLWithPath: request.fixture),
            name: "concurrency", rows: request.rows)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let artifact = fixture.artifact(matrix: true)
        let source = directory.appendingPathComponent("Stress.mlmodel")
        try artifact.write(to: source)
        let package = request.modelPackage.map { URL(fileURLWithPath: $0) }
        let modelHash = try package.map { try coreMLPackageHash($0) }
        let compiled = try await MLModel.compileModel(at: package ?? source)
        defer { try? FileManager.default.removeItem(at: compiled) }
        let transportElementBytes = try package == nil ? 8 : programMatrixElementBytes(at: compiled,
            rows: request.rows, width: fixture.width, outputs: fixture.outputWidth)
        let units: MLComputeUnits = request.policy == "cpu" ? .cpuOnly : .cpuAndNeuralEngine
        let devices = try await plannedDevices(at: compiled, units: units)
        // Prepared inputs and owned results are Double; transport follows the verified model dtype.
        let inputBytes = request.rows * fixture.width * 8
        let outputBytes = request.rows * fixture.outputWidth * 8
        let transportBytes = request.rows * (fixture.width + fixture.outputWidth) * transportElementBytes
        let estimate = inputBytes + outputBytes + transportBytes + 8_388_608
        let signposter = OSSignposter(subsystem: "org.swiftsci.benchmarks", category: "PointsOfInterest")
        let scope = signposter.beginInterval("CoreML owner scope")
        let result: PoolResult
        if request.execution == "persistent" {
            result = try await runPersistentPool(request, fixture: fixture, compiled: compiled, estimate: estimate)
        } else {
            result = try await runPool(request, fixture: fixture, compiled: compiled, estimate: estimate)
        }
        signposter.endInterval("CoreML owner scope", scope)
        guard result.owners.allSatisfy({ $0.value == nil }) else { throw BenchmarkFailure("Predictor outlived completed tasks") }
        for caller in result.callers {
            guard try coreMLResultHash(caller.first) == result.hashes[caller.firstVariant],
                  try coreMLResultHash(caller.last) == result.hashes[caller.lastVariant] else {
                throw BenchmarkFailure("Retained result changed after releasing predictor")
            }
        }
        let hash = modelHash ?? SHA256.hash(data: artifact).map { String(format: "%02x", $0) }.joined()
        let record = StressRecord(request: request, modelSHA256: hash, plannedDevices: devices,
            transportElementBytes: transportElementBytes, requestReservationBytes: estimate, budgetLimitBytes: estimate * request.admissionSlots + (request.retainedBytes ?? 0),
            elapsedSeconds: result.elapsed, callers: result.callers, memory: result.memory,
            budgetPeakBytes: result.peak, finalReservedBytes: result.reserved, finalQueuedCount: result.queued,
            cancelledRequestsVerified: result.cancellationChecks, preflightReferenceMismatches: result.preflightReferenceMismatches,
            outputHashes: result.hashes, predictorsReleased: true, retainedOutputsVerified: true,
            beforePoolRSS: result.beforePoolRSS, afterPoolRSS: try residentBytes(),
            poolPredictions: result.poolPredictions, poolOutputBackingIdentityMatches: result.poolOutputBackingIdentityMatches)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(record).write(to: output, options: .atomic)
        return 0
    } catch {
        fputs("Core ML concurrency failed: \(error)\n", stderr)
        return 1
    }
}
