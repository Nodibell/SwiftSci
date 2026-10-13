import CoreML
import Foundation
import Testing
@testable import SwiftML
import SwiftPreprocessing

@Suite("Retained session scope cancellation", .serialized)
struct CoreMLBatchSessionCancellationTests {
    @Test func cancellationHoldsBothChargesUntilConsumerReturns() async throws {
        let (source, plan) = try pipelineFixture()
        let config = try pipelineConfiguration()
        let lifetime = try MemoryBudget(limit: config.sessionBytes)
        let work = try MemoryBudget(limit: config.windowBytes)
        let gate = PipelineGate()
        try await withPreparedMatrixModel(4) { url in
            let scope = Task {
                try await CoreMLBatchPipeline.withSession(compiledModelURL: url,
                    inputColumns: plan.columnNames, outputName: "result", computeUnits: .cpuOnly,
                    preprocessing: plan, lifetimeBudget: lifetime, workBudget: work,
                    configuration: config) { session in
                    try await session.run(load: { $0.sequence == 0 ? source : nil },
                        consume: { _, _ in await gate.wait(); try Task.checkCancellation() })
                }
            }
            let deadline = ContinuousClock.now.advanced(by: .seconds(5))
            while !(await gate.entered), ContinuousClock.now < deadline { await Task.yield() }
            #expect(await gate.entered)
            scope.cancel()
            #expect(await lifetime.reservedBytes == lifetime.limit)
            #expect(await work.reservedBytes == work.limit)
            await gate.release()
            await #expect(throws: CancellationError.self) { try await scope.value }
        }
        #expect(await lifetime.reservedBytes == 0)
        #expect(await work.reservedBytes == 0)
    }
}
