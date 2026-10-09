import CoreML
import Foundation
import Testing
import SwiftML
import SwiftPreprocessing

func withPreparedMatrixModel(_ bytes: Int, body: (URL) async throws -> Void) async throws {
    if bytes == 2 {
        let source = try #require(Bundle.module.url(forResource: "matrix", withExtension: "mlmodel", subdirectory: "CoreMLFloat16"))
        try await withCompiledCoreML(Data(contentsOf: source), body: body)
    } else {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [3, 2], float32: bytes == 4), body: body)
    }
}

func waitForPreparedRelease(_ budget: MemoryBudget) async {
    let deadline = ContinuousClock.now.advanced(by: .seconds(5))
    while await budget.reservedBytes != 0, ContinuousClock.now < deadline { await Task.yield() }
    #expect(await budget.reservedBytes == 0)
    #expect(await budget.queuedCount == 0)
}

@Suite("Prepared Core ML matrix ownership", .serialized)
struct CoreMLPreparedMatrixTests {
    @available(macOS 15, *)
    @Test(arguments: [2, 4, 8]) func snapshotIdentityConcurrencyAndFinalOwner(bytes: Int) async throws {
        try await withPreparedMatrixModel(bytes) { url in
            let input = try PreparedNumericBatch(columnNames: ["y", "x"], columns: [[7, -3, 9], [1.1, 3, -1]])
                .selectingRows([2, 0, 2])
            let budget = try MemoryBudget(limit: 2_097_152)
            let inputBudget = try MemoryBudget(limit: 1_048_576)
            let config = try CoreMLMatrixPool.Configuration(maximumConcurrentPredictions: 2,
                retainedBytes: 1_048_576, requestBytes: 65_536)
            var owner: CoreMLPreparedMatrix?
            let (escaped, result, prepared) = try await CoreMLMatrixPool.withPool(compiledModelURL: url,
                inputColumns: ["x", "y"], outputName: "result", computeUnits: .cpuOnly,
                budget: budget, configuration: config) { pool in
                let prepared = try await pool.prepare(input, budget: inputBudget)
                #expect(prepared.rowCount == 3)
                #expect(prepared.columnNames == ["x", "y"])
                #expect(prepared.payloadByteCount == 6 * bytes)
                #expect(await inputBudget.reservedBytes == prepared.reservedBytes)
                let expected = try await pool.predict(input)
                var changed = input
                try changed.updateColumn(at: 1, rows: [0, 1, 2], values: [20, 21, 22])
                _ = try await pool.predict(changed)
                try await withThrowingTaskGroup(of: Void.self) { group in
                    for _ in 0..<8 {
                        group.addTask {
                            for _ in 0..<8 {
                                let output = try await pool.predict(prepared)
                                #expect(try output.values.matrix().values == expected.values.matrix().values)
                                #expect(output.values.originalRowIndices == [2, 0, 2])
                                #expect(output.inputPackingBytes == 6 * bytes)
                            }
                        }
                    }
                    try await group.waitForAll()
                }
                let result = try await pool.predict(prepared)
                return (pool, result, prepared)
            }
            #expect(await budget.reservedBytes == 0)
            owner = prepared
            // Exercise another pool with the same input contract, after the original has closed.
            try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: ["x", "y"],
                outputName: "result", computeUnits: .cpuOnly, budget: budget, configuration: config) { pool in
                let output = try await pool.predict(prepared)
                #expect(try output.values.matrix().values == result.values.matrix().values)
            }
            #expect(await inputBudget.reservedBytes == owner!.reservedBytes)
            await #expect(throws: SwiftMLError.self) { try await escaped.predict(prepared) }
            owner = nil
            #expect(await inputBudget.reservedBytes == prepared.reservedBytes)
            #expect(result.values.originalRowIndices == [2, 0, 2])
            // The function below has a separate scope so optimizer lifetime extension cannot
            // accidentally make this assertion depend on when the local `prepared` dies.
            try await checkLastOwner(url: url, config: config, poolBudget: budget, input: input)
        }
    }

    private func checkLastOwner(url: URL, config: CoreMLMatrixPool.Configuration,
                                poolBudget: MemoryBudget, input: PreparedNumericBatch) async throws {
        let inputBudget = try MemoryBudget(limit: 1_048_576)
        let result = try await scopedPreparedResult(url: url, config: config,
            poolBudget: poolBudget, inputBudget: inputBudget, input: input)
        await waitForPreparedRelease(inputBudget)
        #expect(result.values.originalRowIndices == [2, 0, 2])
        #expect(result.values[0, 1] == 9)
    }

    @inline(never) private func scopedPreparedResult(url: URL, config: CoreMLMatrixPool.Configuration,
        poolBudget: MemoryBudget, inputBudget: MemoryBudget, input: PreparedNumericBatch) async throws -> CoreMLPrediction {
        try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: ["x", "y"],
            outputName: "result", computeUnits: .cpuOnly, budget: poolBudget, configuration: config) { pool in
            let prepared = try await pool.prepare(input, budget: inputBudget)
            let alias = prepared
            let result = try await pool.predict(alias)
            #expect(await inputBudget.reservedBytes == alias.reservedBytes)
            return result
        }
    }
}
