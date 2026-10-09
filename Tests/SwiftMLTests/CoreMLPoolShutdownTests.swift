import CoreML
import Foundation
import Testing
@testable import SwiftML
import SwiftPreprocessing

private enum ShutdownFailure: Error, Equatable { case scope, operation }
private actor ShutdownGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private var open = false
    private(set) var entered = false
    func wait() async {
        entered = true
        if !open { await withCheckedContinuation { continuation = $0 } }
    }
    func release() { open = true; continuation?.resume(); continuation = nil }
}
private actor ShutdownState {
    private(set) var pool: CoreMLMatrixPool?
    private(set) var tasks: [Task<Void, any Error>] = []
    private(set) var completed = false
    private(set) var queuedOperationRan = false
    func install(_ pool: CoreMLMatrixPool, tasks: [Task<Void, any Error>]) { self.pool = pool; self.tasks = tasks }
    func finish() { completed = true }
    func ranQueuedOperation() { queuedOperationRan = true }
}

@Suite("Core ML pool shutdown ordering", .serialized)
struct CoreMLPoolShutdownTests {
    @Test(arguments: ["success", "error", "cancel"])
    func holdsQuotaUntilAdmittedOperationFinishes(outcome: String) async throws {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [3, 1])) { url in
            let gate = ShutdownGate()
            let state = ShutdownState()
            let budget = try MemoryBudget(limit: 131_072)
            let configuration = try CoreMLMatrixPool.Configuration(retainedBytes: 65_536, requestBytes: 65_536)
            let scope = Task {
                do {
                    try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: ["x"],
                        outputName: "result", computeUnits: .cpuOnly, budget: budget, configuration: configuration) { pool in
                        let held = Task<Void, any Error> {
                            try await pool.withSlot { _, buffers in
                                buffers.input.array[0] = 7
                                await gate.wait()
                                #expect(buffers.input.array[0].intValue == 7)
                                try Task.checkCancellation()
                                if outcome == "error" { throw ShutdownFailure.operation }
                            }
                        }
                        let entryDeadline = ContinuousClock.now.advanced(by: .seconds(5))
                        while !(await gate.entered), ContinuousClock.now < entryDeadline { await Task.yield() }
                        #expect(await gate.entered)
                        let queued = Task<Void, any Error> {
                            try await pool.withSlot { _, _ in await state.ranQueuedOperation() }
                        }
                        await state.install(pool, tasks: [held, queued])
                        let queueDeadline = ContinuousClock.now.advanced(by: .seconds(5))
                        while await pool.pending < 2, ContinuousClock.now < queueDeadline { await Task.yield() }
                        #expect(await pool.pending == 2)
                        if outcome == "cancel" { held.cancel() }
                        throw ShutdownFailure.scope
                    }
                } catch {
                    await state.finish()
                    throw error
                }
            }
            let closeDeadline = ContinuousClock.now.advanced(by: .seconds(5))
            while ContinuousClock.now < closeDeadline {
                if let pool = await state.pool, await pool.closed { break }
                await Task.yield()
            }
            let pool = await state.pool
            #expect(await pool?.closed == true)
            #expect(await pool?.pending == 2)
            #expect(await budget.reservedBytes == budget.limit)
            #expect(!(await state.completed))
            await gate.release()
            await #expect(throws: ShutdownFailure.scope) { try await scope.value }
            let tasks = await state.tasks
            if tasks.count == 2 {
                switch outcome {
                case "error": await #expect(throws: ShutdownFailure.operation) { try await tasks[0].value }
                case "cancel": await #expect(throws: CancellationError.self) { try await tasks[0].value }
                default: try await tasks[0].value
                }
                await #expect(throws: SwiftMLError.self) { try await tasks[1].value }
            } else { Issue.record("Missing admitted and queued tasks") }
            #expect(!(await state.queuedOperationRan))
            #expect(await pool?.pending == 0)
            #expect(await budget.reservedBytes == 0)
            #expect(await budget.queuedCount == 0)
        }
    }
}
