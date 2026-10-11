import CoreML
import Foundation
import SwiftDataFrame
import SwiftPreprocessing

/// Runs a pull-based source through fitted preprocessing and a fixed-shape Core ML model.
/// Holds admission through ordered, awaited consumption, including completed predictions.
public enum CoreMLBatchPipeline {
    /// Conservative allowances for a run. These account for estimates, not process RSS.
    public struct Configuration: Sendable {
        /// Native prediction slots and their retained and request allowances.
        public let pool: CoreMLMatrixPool.Configuration
        /// Maximum loaded but not yet consumed batches, including completed outputs.
        public let maximumInFlightBatches: Int
        /// Fitted plan, source cursor, consumer state, and scheduling overhead for the run.
        public let retainedBytes: Int
        /// Per-batch peak including loading, source storage, preparation, output, and consumption.
        public let batchBytes: Int

        /// Requires positive allowances and at least as many batch positions as prediction slots.
        public init(pool: CoreMLMatrixPool.Configuration, maximumInFlightBatches: Int,
                    retainedBytes: Int, batchBytes: Int) throws {
            guard maximumInFlightBatches >= pool.maximumConcurrentPredictions,
                  retainedBytes > 0, batchBytes > 0 else { throw MemoryAdmissionError.invalidCapacity }
            self.pool = pool
            self.maximumInFlightBatches = maximumInFlightBatches
            self.retainedBytes = retainedBytes
            self.batchBytes = batchBytes
            _ = try requiredBytes
        }

        /// Complete reservation acquired before model loading or the first source callback.
        public var requiredBytes: Int {
            get throws {
                try MemoryEstimate(capacities: [pool.quota.bytes, retainedBytes,
                    coreMLByteCount(maximumInFlightBatches, batchBytes)]).bytes
            }
        }
    }

    /// Model-derived source request. Sequence numbers start at zero and increase in load order.
    public struct Request: Sendable {
        /// Identifies a batch independently of its local originalRowIndices.
        public let sequence: Int
        /// Required number of rows. Short batches throw; the pipeline never pads or drops rows.
        public let rowCount: Int
        /// Required column names in fitted model order.
        public let columnNames: [String]
    }

    /// Reserves the complete run before loading, then pulls at most the configured batch window.
    ///
    /// Fit `preprocessing` on training data beforehand. Training allocations are outside this scope.
    /// `load` runs serially and returns nil only at exhaustion. Each batch must have exactly the
    /// requested shape and column order. Its original row indices survive prediction unchanged;
    /// use the sequence number as well when separate chunks have local row indices.
    ///
    /// `consume` runs serially in source order. Await persistence or reduction before returning.
    /// Results retained outside the callback, and preallocated source data, need caller accounting.
    /// The supplied estimates must cover loader and consumer working memory and Core ML internals.
    /// Known storage minima are checked, but this API cannot impose an allocator or RSS limit.
    ///
    /// Failure stops loading when observed and drains structured tasks and native predictions.
    /// Earlier consumer effects remain committed. There are no retries or implicit transactions.
    /// Callbacks must cooperate with cancellation; the run waits for them before releasing capacity.
    public static func run(compiledModelURL: URL, inputColumns: [String], outputName: String,
        computeUnits: MLComputeUnits = .all, preprocessing: StandardPreprocessingPlan,
        budget: MemoryBudget, configuration: Configuration,
        load: @Sendable (Request) async throws -> PreparedNumericBatch?,
        consume: @Sendable (Int, CoreMLPrediction) async throws -> Void) async throws {
        guard preprocessing.columnNames == inputColumns else {
            throw SwiftMLError.invalidParameter("Pipeline columns differ from the fitted plan")
        }
        let quota = try MemoryEstimate(capacities: [configuration.requiredBytes])
        try await budget.withReservation(quota) {
            try Task.checkCancellation()
            // These budgets subdivide an already admitted quota. Acquiring the parent again
            // would deadlock a run that owns all remaining shared capacity.
            let poolBudget = try MemoryBudget(limit: configuration.pool.quota.bytes)
            try await CoreMLMatrixPool.withPool(compiledModelURL: compiledModelURL,
                inputColumns: inputColumns, outputName: outputName, computeUnits: computeUnits,
                budget: poolBudget, configuration: configuration.pool) { pool in
                let preparation = try await pool.inputPreparation()
                let prepared = try CoreMLPreparedMatrix.allowance(schema: preparation.schema, names: inputColumns)
                let workspace = try preprocessing.workspaceAllowance(rowCount: preparation.rowCount)
                let preparationBytes = try MemoryEstimate(capacities: [prepared.bytes, workspace.bytes]).bytes
                let sourceMinimum = try coreMLByteCount(preparation.rowCount, inputColumns.count, 8)
                try requireBatchCapacity(sourceBytes: sourceMinimum, preparationBytes: preparationBytes,
                    configuration: configuration)
                try await runWindow(pool: pool, preparation: preparation, plan: preprocessing,
                    preparationBytes: preparationBytes, configuration: configuration, load: load, consume: consume)
            }
        }
    }
}
