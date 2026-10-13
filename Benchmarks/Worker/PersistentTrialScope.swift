import CoreML
import Foundation
import SwiftML
import SwiftPreprocessing
import SwiftSciBenchmarkSupport

struct PersistentTrialScope: Sendable {
    let work: MemoryBudget
    let retained: MemoryBudget
    let retainedBytes: Int
    let windowBytes: Int
    let totalBytes: Int

    init(configuration: CoreMLBatchPipeline.Configuration, totalBytes: Int) throws {
        windowBytes = try configuration.windowBytes
        retainedBytes = try configuration.sessionBytes
        let minimum = try MemoryEstimate(capacities: [retainedBytes, max(windowBytes, mixedSmallQuota)]).bytes
        guard totalBytes >= minimum else {
            throw MemoryAdmissionError.exceedsLimit(required: minimum, limit: totalBytes)
        }
        self.totalBytes = totalBytes
        retained = try MemoryBudget(limit: retainedBytes)
        work = try MemoryBudget(limit: totalBytes - retainedBytes)
    }

    func withModel<T: Sendable>(_ q: MixedWorkloadRequest, fixture: MixedWorkloadFixture,
        operation: @Sendable (CoreMLBatchSession) async throws -> T) async throws -> T {
        try await CoreMLBatchPipeline.withSession(compiledModelURL: fixture.model,
            inputColumns: fixture.plan.columnNames, outputName: "result",
            computeUnits: q.policy == "cpu" ? .cpuOnly : .cpuAndNeuralEngine,
            preprocessing: fixture.plan, lifetimeBudget: retained, workBudget: work,
            configuration: fixture.configuration, operation: operation)
    }

    func verifyDrained() async throws {
        guard await retained.reservedBytes == 0, await retained.queuedCount == 0,
              await work.reservedBytes == 0, await work.queuedCount == 0 else {
            throw BenchmarkFailure("Persistent trial partition did not drain")
        }
        let modelToken = try await retained.acquire(MemoryEstimate(capacities: [retainedBytes]))
        let workToken = try await work.acquire(MemoryEstimate(capacities: [totalBytes - retainedBytes]))
        await workToken.finish(); await modelToken.finish()
    }
}

func persistentRequest(_ q: MixedWorkloadRequest, quantum: Int) -> MixedWorkloadRequest {
    MixedWorkloadRequest(model: q.model, rawFixture: q.rawFixture, rows: q.rows,
        policy: q.policy, batches: q.batches, quantum: quantum)
}

struct PersistentTrialSample: Codable {
    let variant: String
    let quantum: Int
    let poolScopes: Int
    let startupSeconds: Double?
    let retainedBytes: Int
    let workBudgetBytes: Int
    let sample: MixedSample
}
