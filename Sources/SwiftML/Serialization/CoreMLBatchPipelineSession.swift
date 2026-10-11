import CoreML
import Foundation
import SwiftPreprocessing

extension CoreMLBatchPipeline.Configuration {
    /// Capacity held for the model pool and source/consumer context across successive runs.
    public var sessionBytes: Int {
        get throws { try MemoryEstimate(capacities: [pool.quota.bytes, retainedBytes]).bytes }
    }

    /// Capacity reacquired for each run, including every loaded but not consumed batch.
    public var windowBytes: Int {
        get throws { try coreMLByteCount(maximumInFlightBatches, batchBytes) }
    }
}

extension CoreMLBatchPipeline {
    /// Retains a model while successive runs acquire and release a separate work budget.
    ///
    /// Supply distinct lifetime and work partitions. Their limits add to the participating
    /// memory allowance. CPU/GPU peers may share `workBudget`. Never acquire lifetime
    /// capacity while holding workspace, or acquire workspace again inside run callbacks.
    /// Idle lifetime capacity is unavailable to work; reservations do not cap process RSS.
    ///
    /// The lifetime charge includes pool slots and source/consumer context. Each run holds
    /// its entire window until callbacks and predictions finish. Bound each source to choose
    /// how often workspace is yielded. No automatic chunk size or scheduling policy is chosen.
    ///
    /// Await structured child tasks inside `operation`. Exit rejects new runs, drains an
    /// active run, closes the pool, and releases its lifetime charge. Callbacks must cooperate
    /// with cancellation. Escaped session references reject runs after this scope closes.
    public static func withSession<Result: Sendable>(compiledModelURL: URL, inputColumns: [String],
        outputName: String, computeUnits: MLComputeUnits = .all,
        preprocessing: StandardPreprocessingPlan, lifetimeBudget: MemoryBudget,
        workBudget: MemoryBudget, configuration: Configuration,
        operation: @Sendable (CoreMLBatchSession) async throws -> Result) async throws -> Result {
        guard lifetimeBudget !== workBudget else {
            throw SwiftMLError.invalidParameter("Retained sessions require distinct lifetime and work budgets")
        }
        guard preprocessing.columnNames == inputColumns else {
            throw SwiftMLError.invalidParameter("Pipeline columns differ from the fitted plan")
        }
        let window = try configuration.windowBytes
        let workLimit = await workBudget.limit
        guard window <= workLimit else {
            throw MemoryAdmissionError.exceedsLimit(required: window, limit: workLimit)
        }
        return try await lifetimeBudget.withReservation(MemoryEstimate(capacities: [configuration.sessionBytes])) {
            let privatePoolBudget = try MemoryBudget(limit: configuration.pool.quota.bytes)
            return try await CoreMLMatrixPool.withPool(compiledModelURL: compiledModelURL,
                inputColumns: inputColumns, outputName: outputName, computeUnits: computeUnits,
                budget: privatePoolBudget, configuration: configuration.pool) { pool in
                let session = CoreMLBatchSession(pool: pool, preprocessing: preprocessing,
                    configuration: configuration, workBudget: workBudget)
                do {
                    let result = try await operation(session)
                    await session.close()
                    return result
                } catch {
                    await session.close()
                    throw error
                }
            }
        }
    }
}
