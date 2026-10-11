import CoreML
import Testing
@testable import SwiftML
import SwiftPreprocessing

@Suite("Bounded Core ML batch results", .serialized)
struct CoreMLBatchPipelineTests {
    @Test(arguments: [4, 8]) func matchesDirectPreparation(bytes: Int) async throws {
        try await checkParity(bytes: bytes)
    }

    @available(macOS 15, *)
    @Test func float16MatchesDirectPreparation() async throws { try await checkParity(bytes: 2) }

    private func checkParity(bytes: Int) async throws {
        let (source, plan) = try pipelineFixture()
        let configuration = try pipelineConfiguration(slots: 2)
        try await withPreparedMatrixModel(bytes) { url in
            let budget = try MemoryBudget(limit: configuration.requiredBytes)
            let observations = PipelineObservations()
            try await CoreMLBatchPipeline.run(compiledModelURL: url, inputColumns: ["x", "y"],
                outputName: "result", computeUnits: .cpuOnly, preprocessing: plan,
                budget: budget, configuration: configuration, load: { request in
                    #expect(request.rowCount == 3)
                    #expect(request.columnNames == ["x", "y"])
                    return request.sequence < 7 ? source : nil
                }, consume: { sequence, result in try await observations.consumed(sequence, result) })
            #expect(await budget.reservedBytes == 0)
            #expect(await budget.peak == (try configuration.requiredBytes))
            #expect(await observations.sequences == Array(0..<7))
            #expect(await observations.rows == Array(repeating: [2, 0, 2], count: 7))
            let preparationBudget = try MemoryBudget(limit: 262_144)
            let expected = try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: ["x", "y"],
                outputName: "result", computeUnits: .cpuOnly, budget: budget, configuration: configuration.pool) { pool in
                let preparation = try await pool.inputPreparation()
                let owner = try await preparation.prepare(source, preprocessing: plan, budget: preparationBudget)
                return try await pool.predict(owner)
            }
            let expectedValues = try expected.values.matrix().values
            #expect(await observations.values == Array(repeating: expectedValues, count: 7))
            await waitForPreparedRelease(preparationBudget)
        }
    }

    @Test(arguments: [0, 1, 17]) func emptyAndLongSources(count: Int) async throws {
        let (source, plan) = try pipelineFixture()
        let configuration = try pipelineConfiguration(window: 1)
        try await withPreparedMatrixModel(4) { url in
            let observations = PipelineObservations()
            let budget = try MemoryBudget(limit: configuration.requiredBytes)
            try await CoreMLBatchPipeline.run(compiledModelURL: url, inputColumns: ["x", "y"],
                outputName: "result", computeUnits: .cpuOnly, preprocessing: plan, budget: budget,
                configuration: configuration, load: { request in
                    await observations.loaded()
                    return request.sequence < count ? source : nil
                }, consume: { sequence, result in try await observations.consumed(sequence, result) })
            #expect(await observations.loads == count + 1)
            #expect(await observations.sequences == Array(0..<count))
            #expect(await budget.reservedBytes == 0)
        }
    }

    @Test(arguments: ["source", "consumer", "partial", "nonfinite"])
    func failuresDrainAndRelease(stage: String) async throws {
        let (source, plan) = try pipelineFixture()
        let configuration = try pipelineConfiguration()
        try await withPreparedMatrixModel(4) { url in
            let observations = PipelineObservations()
            let budget = try MemoryBudget(limit: configuration.requiredBytes)
            await #expect(throws: (any Error).self) {
                try await CoreMLBatchPipeline.run(compiledModelURL: url, inputColumns: ["x", "y"],
                    outputName: "result", computeUnits: .cpuOnly, preprocessing: plan,
                    budget: budget, configuration: configuration, load: { request in
                        await observations.loaded()
                        if request.sequence == 3 {
                            if stage == "source" { throw PipelineTestFailure.source }
                            if stage == "partial" { return try source.selectingRows([0]) }
                            if stage == "nonfinite" {
                                return try PreparedNumericBatch(columnNames: ["x", "y"],
                                    columns: [[Double.infinity, 0, 1], [1, 2, 3]])
                            }
                        }
                        return request.sequence < 5 ? source : nil
                    }, consume: { sequence, result in
                        if stage == "consumer" { throw PipelineTestFailure.consumer }
                        try await observations.consumed(sequence, result)
                    })
            }
            #expect(await budget.reservedBytes == 0)
            #expect(await budget.queuedCount == 0)
            let sequences = await observations.sequences
            #expect(sequences == Array(0..<sequences.count))
            #expect(sequences.count <= 3)
        }
    }
}
