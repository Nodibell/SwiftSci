import CoreML
import Foundation
import Testing
@testable import SwiftML
import SwiftPreprocessing

@Suite("Bounded Core ML source admission", .serialized)
struct CoreMLBatchPipelineAdmissionTests {
    @Test func impossibleAndCancelledAdmissionNeverLoads() async throws {
        let (_, plan) = try pipelineFixture()
        let configuration = try pipelineConfiguration()
        let observations = PipelineObservations()
        let budget = try MemoryBudget(limit: configuration.requiredBytes)
        let token = try await budget.acquire(MemoryEstimate(capacities: [configuration.requiredBytes]))
        let task = Task {
            try await CoreMLBatchPipeline.run(compiledModelURL: URL(fileURLWithPath: "/nonexistent-model"),
                inputColumns: ["x", "y"], outputName: "result", preprocessing: plan,
                budget: budget, configuration: configuration,
                load: { _ in await observations.loaded(); return nil }, consume: { _, _ in })
        }
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while await budget.queuedCount == 0, ContinuousClock.now < deadline { await Task.yield() }
        #expect(await budget.queuedCount == 1)
        #expect(await observations.loads == 0)
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        await token.finish()
        #expect(await budget.reservedBytes == 0)
        let tooSmall = try MemoryBudget(limit: configuration.requiredBytes - 1)
        await #expect(throws: MemoryAdmissionError.self) {
            try await CoreMLBatchPipeline.run(compiledModelURL: URL(fileURLWithPath: "/nonexistent-model"),
                inputColumns: ["x", "y"], outputName: "result", preprocessing: plan,
                budget: tooSmall, configuration: configuration,
                load: { _ in await observations.loaded(); return nil }, consume: { _, _ in })
        }
        #expect(await observations.loads == 0)
    }

    @Test(arguments: [false, true]) func consumerHoldsWindowAndQuota(cancel: Bool) async throws {
        let (source, plan) = try pipelineFixture()
        let configuration = try pipelineConfiguration()
        try await withPreparedMatrixModel(4) { url in
            let budget = try MemoryBudget(limit: configuration.requiredBytes)
            let observations = PipelineObservations()
            let gate = PipelineGate()
            let task = Task {
                try await CoreMLBatchPipeline.run(compiledModelURL: url, inputColumns: ["x", "y"],
                    outputName: "result", computeUnits: .cpuOnly, preprocessing: plan,
                    budget: budget, configuration: configuration, load: { request in
                        await observations.loaded()
                        return request.sequence < 5 ? source : nil
                    }, consume: { sequence, result in
                        if sequence == 0 { await gate.wait() }
                        try Task.checkCancellation()
                        try await observations.consumed(sequence, result)
                    })
            }
            let deadline = ContinuousClock.now.advanced(by: .seconds(5))
            while !(await gate.entered), ContinuousClock.now < deadline { await Task.yield() }
            #expect(await gate.entered)
            #expect(await observations.loads == 2)
            #expect(await budget.reservedBytes == (try configuration.requiredBytes))
            if cancel { task.cancel() }
            #expect(await budget.reservedBytes == (try configuration.requiredBytes))
            await gate.release()
            if cancel {
                await #expect(throws: CancellationError.self) { try await task.value }
                #expect(await observations.loads == 2)
            } else {
                try await task.value
                #expect(await observations.sequences == Array(0..<5))
            }
            #expect(await budget.reservedBytes == 0)
        }
    }

    @Test func overflowAndKnownStorageMinimumReject() async throws {
        let (_, plan) = try pipelineFixture()
        let pool = try pipelineConfiguration().pool
        #expect(throws: MemoryAdmissionError.self) {
            try CoreMLBatchPipeline.Configuration(pool: pool, maximumInFlightBatches: Int.max,
                retainedBytes: 1, batchBytes: 2)
        }
        #expect(throws: MemoryAdmissionError.self) {
            try CoreMLBatchPipeline.Configuration(pool: pool, maximumInFlightBatches: 0,
                retainedBytes: 1, batchBytes: 1)
        }
        try await withPreparedMatrixModel(4) { url in
            let config = try CoreMLBatchPipeline.Configuration(pool: pool, maximumInFlightBatches: 1,
                retainedBytes: 65_536, batchBytes: 1)
            let budget = try MemoryBudget(limit: config.requiredBytes)
            let observations = PipelineObservations()
            await #expect(throws: MemoryAdmissionError.self) {
                try await CoreMLBatchPipeline.run(compiledModelURL: url, inputColumns: ["x", "y"],
                    outputName: "result", computeUnits: .cpuOnly, preprocessing: plan, budget: budget,
                    configuration: config, load: { _ in await observations.loaded(); return nil },
                    consume: { _, _ in })
            }
            #expect(await observations.loads == 0)
            #expect(await budget.reservedBytes == 0)
        }
    }
}
