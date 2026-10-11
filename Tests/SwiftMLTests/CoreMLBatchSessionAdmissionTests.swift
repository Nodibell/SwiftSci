import CoreML
import Foundation
import Testing
@testable import SwiftML
import SwiftPreprocessing

@Suite("Retained session admission", .serialized)
struct CoreMLBatchSessionAdmissionTests {
    @Test func rejectsInvalidPartitionsBeforeLoading() async throws {
        let (_, plan) = try pipelineFixture()
        let config = try pipelineConfiguration()
        let lifetime = try MemoryBudget(limit: config.sessionBytes)
        let small = try MemoryBudget(limit: config.windowBytes - 1)
        let missing = URL(fileURLWithPath: "/nonexistent-model")
        await #expect(throws: SwiftMLError.self) {
            try await CoreMLBatchPipeline.withSession(compiledModelURL: missing,
                inputColumns: plan.columnNames, outputName: "result", preprocessing: plan,
                lifetimeBudget: lifetime, workBudget: lifetime, configuration: config) { _ in }
        }
        await #expect(throws: MemoryAdmissionError.self) {
            try await CoreMLBatchPipeline.withSession(compiledModelURL: missing,
                inputColumns: plan.columnNames, outputName: "result", preprocessing: plan,
                lifetimeBudget: lifetime, workBudget: small, configuration: config) { _ in }
        }
        #expect(await lifetime.reservedBytes == 0)
    }

    @Test func queuedCancellationDoesNotLoadAndSessionRecovers() async throws {
        let (_, plan) = try pipelineFixture()
        let config = try pipelineConfiguration()
        let lifetime = try MemoryBudget(limit: config.sessionBytes)
        let work = try MemoryBudget(limit: config.windowBytes)
        try await withPreparedMatrixModel(4) { url in
            try await CoreMLBatchPipeline.withSession(compiledModelURL: url,
                inputColumns: plan.columnNames, outputName: "result", computeUnits: .cpuOnly,
                preprocessing: plan, lifetimeBudget: lifetime, workBudget: work,
                configuration: config) { session in
                let blocker = try await work.acquire(MemoryEstimate(capacities: [work.limit]))
                let observations = PipelineObservations()
                let task = Task { try await session.run(load: { _ in
                    await observations.loaded(); return nil
                }, consume: { _, _ in }) }
                let deadline = ContinuousClock.now.advanced(by: .seconds(5))
                while await work.queuedCount == 0, ContinuousClock.now < deadline { await Task.yield() }
                #expect(await work.queuedCount == 1)
                await #expect(throws: SwiftMLError.self) {
                    try await session.run(load: { _ in nil }, consume: { _, _ in })
                }
                task.cancel()
                await #expect(throws: CancellationError.self) { try await task.value }
                #expect(await observations.loads == 0)
                #expect(await work.queuedCount == 0)
                #expect(await lifetime.reservedBytes == lifetime.limit)
                await blocker.finish()
                try await session.run(load: { _ in nil }, consume: { _, _ in })
            }
        }
        #expect(await lifetime.reservedBytes == 0)
        #expect(await work.reservedBytes == 0)
    }
}
