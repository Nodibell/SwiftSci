import CoreML
import Foundation
import Testing
@testable import SwiftML
import SwiftPreprocessing

private actor BatchSessionTask {
    var task: Task<Void, any Error>?
    func install(_ value: Task<Void, any Error>) { task = value }
}

@Suite("Retained session shutdown", .serialized)
struct CoreMLBatchSessionShutdownTests {
    @Test(arguments: [false, true]) func scopeDrainsConsumerBeforeReleasing(cancel: Bool) async throws {
        let (source, plan) = try pipelineFixture()
        let config = try pipelineConfiguration()
        let lifetime = try MemoryBudget(limit: config.sessionBytes)
        let work = try MemoryBudget(limit: config.windowBytes)
        let gate = PipelineGate()
        let holder = BatchSessionTask()
        let scopeLeaving = PipelineGate()
        try await withPreparedMatrixModel(4) { url in
            let scope = Task {
                try await CoreMLBatchPipeline.withSession(compiledModelURL: url,
                    inputColumns: plan.columnNames, outputName: "result", computeUnits: .cpuOnly,
                    preprocessing: plan, lifetimeBudget: lifetime, workBudget: work,
                    configuration: config) { session in
                    let task = Task { try await session.run(load: { $0.sequence == 0 ? source : nil },
                        consume: { _, _ in await gate.wait(); try Task.checkCancellation() }) }
                    await holder.install(task)
                    let deadline = ContinuousClock.now.advanced(by: .seconds(5))
                    while !(await gate.entered), ContinuousClock.now < deadline { await Task.yield() }
                    #expect(await gate.entered)
                    if cancel { task.cancel() }
                    await scopeLeaving.release()
                    throw PipelineTestFailure.source
                }
            }
            await scopeLeaving.wait()
            #expect(await lifetime.reservedBytes == lifetime.limit)
            #expect(await work.reservedBytes == work.limit)
            await gate.release()
            await #expect(throws: PipelineTestFailure.source) { try await scope.value }
            let task = try #require(await holder.task)
            if cancel { await #expect(throws: CancellationError.self) { try await task.value } }
            else { try await task.value }
        }
        #expect(await lifetime.reservedBytes == 0)
        #expect(await work.reservedBytes == 0)
        #expect(await work.queuedCount == 0)
    }
}
