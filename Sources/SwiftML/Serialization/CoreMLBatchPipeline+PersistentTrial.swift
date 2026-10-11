import SwiftPreprocessing

extension CoreMLBatchPipeline {
    // Experiment-only entry point. The caller holds the pool lifetime reservation and
    // the complete batch-window allowance in separate, pre-sized budget partitions.
    package static func runPersistentTrial(pool: CoreMLMatrixPool, preprocessing: StandardPreprocessingPlan,
        configuration: Configuration, load: @Sendable (Request) async throws -> PreparedNumericBatch?,
        consume: @Sendable (Int, CoreMLPrediction) async throws -> Void) async throws {
        let preparation = try await pool.inputPreparation()
        guard preprocessing.columnNames == preparation.columnNames else {
            throw SwiftMLError.invalidParameter("Persistent trial columns differ from fitted plan")
        }
        let prepared = try CoreMLPreparedMatrix.allowance(schema: preparation.schema, names: preparation.columnNames)
        let workspace = try preprocessing.workspaceAllowance(rowCount: preparation.rowCount)
        let bytes = try MemoryEstimate(capacities: [prepared.bytes, workspace.bytes]).bytes
        let source = try coreMLByteCount(preparation.rowCount, preparation.columnNames.count, 8)
        try requireBatchCapacity(sourceBytes: source, preparationBytes: bytes, configuration: configuration)
        try await runWindow(pool: pool, preparation: preparation, plan: preprocessing,
            preparationBytes: bytes, configuration: configuration, load: load, consume: consume)
    }
}
