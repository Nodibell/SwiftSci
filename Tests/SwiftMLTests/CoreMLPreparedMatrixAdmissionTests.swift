import CoreML
import Foundation
import Testing
import SwiftML
import SwiftPreprocessing

@Suite("Prepared Core ML matrix admission", .serialized)
struct CoreMLPreparedMatrixAdmissionTests {
    @available(macOS 15, *)
    @Test func failedPreparationAndCancellationReleaseAdmission() async throws {
        try await withPreparedMatrixModel(2) { url in
            let budget = try MemoryBudget(limit: 131_072)
            let inputBudget = try MemoryBudget(limit: 1_048_576)
            let input = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[1, 2, 3], [4, 5, 6]])
            let config = try CoreMLMatrixPool.Configuration(retainedBytes: 65_536, requestBytes: 65_536)
            let pool = try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: ["x", "y"],
                outputName: "result", computeUnits: .cpuOnly, budget: budget, configuration: config) { pool in
                for value: Double? in [nil, .nan, .infinity, 65520] {
                    var bad = input
                    try bad.updateColumn(at: 0, rows: [1], values: [value])
                    await #expect(throws: SwiftMLError.self) { try await pool.prepare(bad, budget: inputBudget) }
                    #expect(await inputBudget.reservedBytes == 0)
                }
                await #expect(throws: MemoryAdmissionError.self) {
                    try await pool.prepare(input, budget: MemoryBudget(limit: 1))
                }
                let short = try input.selectingRows([0])
                await #expect(throws: SwiftMLError.self) { try await pool.prepare(short, budget: inputBudget) }
                let held = try await inputBudget.acquire(MemoryEstimate(capacities: [inputBudget.limit]))
                let attempt = Task { try await pool.prepare(input, budget: inputBudget) }
                let deadline = ContinuousClock.now.advanced(by: .seconds(5))
                while await inputBudget.queuedCount == 0, ContinuousClock.now < deadline { await Task.yield() }
                #expect(await inputBudget.queuedCount == 1)
                attempt.cancel()
                await #expect(throws: CancellationError.self) { try await attempt.value }
                #expect(await inputBudget.reservedBytes == inputBudget.limit)
                await held.finish()
                #expect(await inputBudget.reservedBytes == 0)
                let prepared = try await pool.prepare(input, budget: inputBudget)
                #expect(try await pool.predict(prepared).values[2, 1] == 6)
                return pool
            }
            await waitForPreparedRelease(inputBudget)
            await #expect(throws: SwiftMLError.self) { try await pool.prepare(input, budget: inputBudget) }
            #expect(await budget.reservedBytes == 0)
        }
    }

    @Test func queuedPreparationRejectsAfterPoolCloses() async throws {
        try await withPreparedMatrixModel(4) { url in
            let budget = try MemoryBudget(limit: 131_072)
            let inputBudget = try MemoryBudget(limit: 1_048_576)
            let blocker = try await inputBudget.acquire(MemoryEstimate(capacities: [inputBudget.limit]))
            let input = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[1, 2, 3], [4, 5, 6]])
            let config = try CoreMLMatrixPool.Configuration(retainedBytes: 65_536, requestBytes: 65_536)
            let attempt = try await CoreMLMatrixPool.withPool(compiledModelURL: url,
                inputColumns: ["x", "y"], outputName: "result", computeUnits: .cpuOnly,
                budget: budget, configuration: config) { pool in
                let attempt = Task { try await pool.prepare(input, budget: inputBudget) }
                let deadline = ContinuousClock.now.advanced(by: .seconds(5))
                while await inputBudget.queuedCount == 0, ContinuousClock.now < deadline { await Task.yield() }
                #expect(await inputBudget.queuedCount == 1)
                return attempt
            }
            #expect(await budget.reservedBytes == 0)
            await blocker.finish()
            await #expect(throws: SwiftMLError.self) { try await attempt.value }
            await waitForPreparedRelease(inputBudget)
        }
    }

    @available(macOS 15, *)
    @Test func incompatiblePoolRejectsBeforePrediction() async throws {
        try await withPreparedMatrixModel(2) { half in
            try await withPreparedMatrixModel(4) { single in
                let budget = try MemoryBudget(limit: 262_144)
                let inputBudget = try MemoryBudget(limit: 1_048_576)
                let input = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[1, 2, 3], [4, 5, 6]])
                let config = try CoreMLMatrixPool.Configuration(retainedBytes: 65_536, requestBytes: 65_536)
                let prepared = try await CoreMLMatrixPool.withPool(compiledModelURL: half,
                    inputColumns: ["x", "y"], outputName: "result", computeUnits: .cpuOnly,
                    budget: budget, configuration: config) { try await $0.prepare(input, budget: inputBudget) }
                for (url, names) in [(single, ["x", "y"]), (half, ["y", "x"])] {
                    try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: names,
                        outputName: "result", computeUnits: .cpuOnly, budget: budget, configuration: config) { pool in
                        await #expect(throws: SwiftMLError.self) { try await pool.predict(prepared) }
                        #expect(await pool.completedPredictions == 0)
                        _ = try await pool.predict(input)
                    }
                }
                #expect(await inputBudget.reservedBytes == prepared.reservedBytes)
            }
        }
    }
}
