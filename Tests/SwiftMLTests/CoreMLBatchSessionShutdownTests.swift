import CoreML
import Foundation
import Testing
@testable import SwiftML
import SwiftPreprocessing

private actor BatchSessionTask {
    var task: Task<Void, any Error>?
    var session: CoreMLBatchSession?
    var completed = false
    func install(_ value: Task<Void, any Error>, session: CoreMLBatchSession) {
        task = value; self.session = session
    }
    func finish() { completed = true }
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
        try await withPreparedMatrixModel(4) { url in
            let scope = Task {
                do {
                    try await CoreMLBatchPipeline.withSession(compiledModelURL: url,
                        inputColumns: plan.columnNames, outputName: "result", computeUnits: .cpuOnly,
                        preprocessing: plan, lifetimeBudget: lifetime, workBudget: work,
                        configuration: config) { session in
                        let task = Task { try await session.run(load: { $0.sequence == 0 ? source : nil },
                            consume: { _, _ in await gate.wait(); try Task.checkCancellation() }) }
                        await holder.install(task, session: session)
                        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
                        while !(await gate.entered), ContinuousClock.now < deadline { await Task.yield() }
                        #expect(await gate.entered)
                        if cancel { task.cancel() }
                        throw PipelineTestFailure.source
                    }
                } catch { await holder.finish(); throw error }
            }
            let deadline = ContinuousClock.now.advanced(by: .seconds(5))
            while ContinuousClock.now < deadline {
                if let session = await holder.session, await session.closed { break }
                await Task.yield()
            }
            let session = await holder.session
            #expect(await session?.closed == true)
            #expect(!(await holder.completed))
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
