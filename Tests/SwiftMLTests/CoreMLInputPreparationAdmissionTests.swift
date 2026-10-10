import CoreML
import Testing
@testable import SwiftML
import SwiftPreprocessing

@Suite("Fitted input admission and lifetime", .serialized)
struct CoreMLInputPreparationAdmissionTests {
    private func fixtures() throws -> (PreparedNumericBatch, StandardPreprocessingPlan) {
        let training = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[0, 1, 2], [3, 4, 5]])
        let source = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[2, .nan, 0], [5, 4, 3]])
        return (source, try StandardPreprocessingPlan(training: training))
    }

    @Test func metadataOutlivesPoolWithoutRetainingItsQuota() async throws {
        let (source, plan) = try fixtures()
        try await withPreparedMatrixModel(4) { url in
            let poolBudget = try MemoryBudget(limit: 2_097_152)
            let config = try CoreMLMatrixPool.Configuration(retainedBytes: 1_048_576, requestBytes: 65_536)
            let (preparation, closedPool) = try await CoreMLMatrixPool.withPool(compiledModelURL: url,
                inputColumns: ["x", "y"], outputName: "result", computeUnits: .cpuOnly,
                budget: poolBudget, configuration: config) { pool in
                (try await pool.inputPreparation(), pool)
            }
            #expect(await poolBudget.reservedBytes == 0)
            await #expect(throws: SwiftMLError.self) { try await closedPool.inputPreparation() }
            let inputBudget = try MemoryBudget(limit: 1_048_576)
            try await checkPreparedLifetime(preparation, source: source, plan: plan, budget: inputBudget)
            await waitForPreparedRelease(inputBudget)
        }
    }

    private func checkPreparedLifetime(_ preparation: CoreMLMatrixInputPreparation,
        source: PreparedNumericBatch, plan: StandardPreprocessingPlan, budget: MemoryBudget) async throws {
        let prepared = try await preparation.prepare(source, preprocessing: plan, budget: budget)
        #expect(await budget.reservedBytes == prepared.reservedBytes)
        let workspace = try plan.workspaceAllowance(for: source)
        #expect(await budget.peak == prepared.reservedBytes + workspace.bytes)
        #expect(prepared.rowCount == source.rowCount)
    }

    @Test func rejectionAndQueuedCancellationReleaseAllCapacity() async throws {
        let (source, plan) = try fixtures()
        try await withPreparedMatrixModel(4) { url in
            let poolBudget = try MemoryBudget(limit: 2_097_152)
            let config = try CoreMLMatrixPool.Configuration(retainedBytes: 1_048_576, requestBytes: 65_536)
            try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: ["x", "y"],
                outputName: "result", computeUnits: .cpuOnly, budget: poolBudget, configuration: config) { pool in
                let preparation = try await pool.inputPreparation()
                let lifetime = try CoreMLPreparedMatrix.allowance(schema: preparation.schema, names: preparation.names)
                let tooSmall = try MemoryBudget(limit: lifetime.bytes)
                await #expect(throws: MemoryAdmissionError.self) {
                    try await preparation.prepare(source, preprocessing: plan, budget: tooSmall)
                }
                #expect(await tooSmall.reservedBytes == 0)
                #expect(await tooSmall.peak == 0)
                let budget = try MemoryBudget(limit: 1_048_576)
                let blocker = try await budget.acquire(MemoryEstimate(capacities: [1_048_576]))
                let pending = Task { try await preparation.prepare(source, preprocessing: plan, budget: budget) }
                let deadline = ContinuousClock.now.advanced(by: .seconds(2))
                while await budget.queuedCount == 0, ContinuousClock.now < deadline { await Task.yield() }
                #expect(await budget.queuedCount == 1)
                pending.cancel()
                await #expect(throws: CancellationError.self) { try await pending.value }
                #expect(await budget.reservedBytes == 1_048_576)
                #expect(await budget.queuedCount == 0)
                await blocker.finish()
                let reordered = try PreparedNumericBatch(columnNames: ["y", "x"], columns: [[1, 2, 3], [4, 5, 6]])
                await #expect(throws: SwiftMLError.self) {
                    try await preparation.prepare(reordered, preprocessing: plan, budget: budget)
                }
                let wrongPlan = try StandardPreprocessingPlan(training: reordered)
                await #expect(throws: SwiftMLError.self) {
                    try await preparation.prepare(source, preprocessing: wrongPlan, budget: budget)
                }
                let short = try source.selectingRows([0])
                await #expect(throws: SwiftMLError.self) {
                    try await preparation.prepare(short, preprocessing: plan, budget: budget)
                }
                let overflow = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[1e100, 2, 3], [4, 5, 6]])
                await #expect(throws: SwiftMLError.self) {
                    try await preparation.prepare(overflow, preprocessing: plan, budget: budget)
                }
                #expect(await budget.reservedBytes == 0)
            }
        }
    }
}
