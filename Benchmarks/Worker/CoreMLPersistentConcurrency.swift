import CoreML
import Foundation
import SwiftDataFrame
import SwiftML
import SwiftPreprocessing
import SwiftSciBenchmarkSupport

enum StressPredictor: Sendable {
    case fresh(CoreMLPredictor)
    case persistent(CoreMLMatrixPool)

    func predict(_ input: PreparedNumericBatch, budget: MemoryBudget,
                 workspaceBytes: Int) async throws -> CoreMLPrediction {
        switch self {
        case .fresh(let predictor):
            return try await predictor.predict(input, budget: budget, workspaceBytes: workspaceBytes)
        case .persistent(let pool): return try await pool.predict(input)
        }
    }
}

func runPersistentPool(_ request: StressRequest, fixture: CoreMLQualificationFixture,
                       compiled: URL, estimate: Int) async throws -> PoolResult {
    let before = try residentBytes()
    let retained = request.retainedBytes!
    let limit = retained + estimate * request.admissionSlots
    let budget = try MemoryBudget(limit: limit)
    let config = try CoreMLMatrixPool.Configuration(maximumConcurrentPredictions: request.admissionSlots,
        retainedBytes: retained, requestBytes: estimate)
    let original = try PreparedNumericBatch(columnNames: fixture.names, columns: fixture.columns)
    let reversed = try original.selectingRows(Array((0..<request.rows).reversed()))
    let inputs = [original, reversed]
    let outputWidth = fixture.outputWidth
    let reference = fixture.expected
    let result = try await CoreMLMatrixPool.withPool(compiledModelURL: compiled, inputColumns: fixture.names,
        outputName: "result", computeUnits: request.policy == "cpu" ? .cpuOnly : .cpuAndNeuralEngine,
        budget: budget, configuration: config) { pool in
        var hashes: [String] = []
        var mismatches = 0
        for variant in 0..<2 {
            let prediction = try await pool.predict(inputs[variant])
            hashes.append(try coreMLResultHash(prediction.values))
            if variant == 0 {
                for row in 0..<request.rows {
                    for column in 0..<outputWidth {
                        let expected = reference[row * outputWidth + column]
                        let actual = prediction.values[row, column]!
                        if abs(actual - expected) > 1e-5 + 1e-5 * abs(expected) { mismatches += 1 }
                    }
                }
            }
        }
        guard hashes[0] != hashes[1] else { throw BenchmarkFailure("Stress variants must differ") }
        var bad = original
        try bad.updateColumn(at: 0, rows: [request.rows - 1], values: [.nan])
        do {
            _ = try await pool.predict(bad)
            throw BenchmarkFailure("Nonfinite pool input unexpectedly succeeded")
        } catch is SwiftMLError { }
        // Lifetime capacity stays reserved even when no prediction is active.
        guard await budget.reservedBytes == limit else { throw BenchmarkFailure("Idle quota changed") }
        var measured = try await measurePredictions(request, predictors: [.persistent(pool)], inputs: inputs,
            hashes: hashes, budget: budget, owners: [WeakPredictor(pool)], cancelled: 0,
            mismatches: mismatches, beforePoolRSS: before, expectedReservation: limit)
        measured.poolPredictions = await pool.completedPredictions
        measured.poolOutputBackingIdentityMatches = await pool.outputBackingIdentityMatches
        return measured
    }
    let reserved = await budget.reservedBytes
    let queued = await budget.queuedCount
    guard reserved == 0, queued == 0 else { throw BenchmarkFailure("Pool quota did not drain") }
    return PoolResult(callers: result.callers, owners: result.owners, hashes: result.hashes,
        memory: result.memory, elapsed: result.elapsed, peak: result.peak, reserved: reserved,
        queued: queued, cancellationChecks: 0, preflightReferenceMismatches: result.preflightReferenceMismatches,
        beforePoolRSS: before, poolPredictions: result.poolPredictions,
        poolOutputBackingIdentityMatches: result.poolOutputBackingIdentityMatches)
}
