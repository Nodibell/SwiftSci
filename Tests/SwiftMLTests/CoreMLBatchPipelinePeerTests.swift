import CoreML
import Foundation
import Testing
import SwiftML
import SwiftPreprocessing

@Suite("Bounded pipeline peer progress", .serialized)
struct CoreMLBatchPipelinePeerTests {
    @Test func competingRunsUseOneWholeReservationWithoutDeadlock() async throws {
        let (source, plan) = try pipelineFixture()
        let configuration = try pipelineConfiguration(window: 1)
        try await withPreparedMatrixModel(4) { url in
            let budget = try MemoryBudget(limit: configuration.requiredBytes)
            let gate = PipelineGate()
            let first = Task {
                try await CoreMLBatchPipeline.run(compiledModelURL: url, inputColumns: ["x", "y"],
                    outputName: "result", computeUnits: .cpuOnly, preprocessing: plan,
                    budget: budget, configuration: configuration,
                    load: { request in request.sequence == 0 ? source : nil },
                    consume: { _, _ in await gate.wait() })
            }
            let deadline = ContinuousClock.now.advanced(by: .seconds(5))
            while !(await gate.entered), ContinuousClock.now < deadline { await Task.yield() }
            #expect(await gate.entered)
            let secondObservations = PipelineObservations()
            let second = Task {
                try await CoreMLBatchPipeline.run(compiledModelURL: url, inputColumns: ["x", "y"],
                    outputName: "result", computeUnits: .cpuOnly, preprocessing: plan,
                    budget: budget, configuration: configuration, load: { request in
                        await secondObservations.loaded()
                        return request.sequence == 0 ? source : nil
                    }, consume: { index, prediction in try await secondObservations.consumed(index, prediction) })
            }
            let queuedDeadline = ContinuousClock.now.advanced(by: .seconds(5))
            while await budget.queuedCount == 0, ContinuousClock.now < queuedDeadline { await Task.yield() }
            #expect(await budget.queuedCount == 1)
            #expect(await secondObservations.loads == 0)
            await gate.release()
            try await first.value
            try await second.value
            #expect(await secondObservations.sequences == [0])
            #expect(await budget.reservedBytes == 0)
            #expect(await budget.peak == (try configuration.requiredBytes))
        }
    }
}
