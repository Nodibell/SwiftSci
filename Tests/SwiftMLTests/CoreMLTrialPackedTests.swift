import CoreML
import Testing
import SwiftML
import SwiftPreprocessing

@Suite("Trial packed Core ML input", .serialized)
struct CoreMLTrialPackedTests {
    @available(macOS 15, *)
    @Test func copyIdentityAndRelease() async throws {
        try await withPreparedMatrixModel(2) { url in
            let input = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[1, 2, 3], [4, 5, 6]])
                .selectingRows([2, 0, 2])
            let budget = try MemoryBudget(limit: 2_097_152)
            let inputBudget = try MemoryBudget(limit: 1_048_576)
            let config = try CoreMLMatrixPool.Configuration(retainedBytes: 1_048_576, requestBytes: 65_536)
            try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: ["x", "y"],
                outputName: "result", computeUnits: .cpuOnly, budget: budget, configuration: config) { pool in
                let expected = try await pool.predict(input)
                var packed: [Float16] = [3, 6, 1, 4, 3, 6]
                let prepared = try await pool.preparePackedForTrial(packed, source: input, budget: inputBudget)
                packed[0] = 100
                let result = try await pool.predict(prepared)
                #expect(result.values.originalRowIndices == [2, 0, 2])
                #expect(try result.values.matrix().values == expected.values.matrix().values)
                #expect(packed[0] == 100)
            }
            await waitForPreparedRelease(inputBudget)
            #expect(await budget.reservedBytes == 0)
        }
    }

    @available(macOS 15, *)
    @Test(arguments: [2, 4]) func rejectsInvalidInputWithoutReserving(bytes: Int) async throws {
        try await withPreparedMatrixModel(bytes) { url in
            let input = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[1, 2, 3], [4, 5, 6]])
            let wrong = try PreparedNumericBatch(columnNames: ["y", "x"], columns: [[1, 2, 3], [4, 5, 6]])
            let budget = try MemoryBudget(limit: 2_097_152)
            let inputBudget = try MemoryBudget(limit: 1_048_576)
            let config = try CoreMLMatrixPool.Configuration(retainedBytes: 1_048_576, requestBytes: 65_536)
            let escaped = try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: ["x", "y"],
                outputName: "result", computeUnits: .cpuOnly, budget: budget, configuration: config) { pool in
                let invalid: [[Float16]] = [[1], [1, 2, 3, 4, 5, .infinity], [1, 2, 3, 4, 5, .nan]]
                for values in invalid {
                    await #expect(throws: SwiftMLError.self) {
                        try await pool.preparePackedForTrial(values, source: input, budget: inputBudget)
                    }
                }
                let values: [Float16] = [1, 4, 2, 5, 3, 6]
                await #expect(throws: SwiftMLError.self) {
                    try await pool.preparePackedForTrial(values, source: wrong, budget: inputBudget)
                }
                if bytes == 4 {
                    await #expect(throws: SwiftMLError.self) {
                        try await pool.preparePackedForTrial(values, source: input, budget: inputBudget)
                    }
                }
                #expect(await inputBudget.reservedBytes == 0)
                return pool
            }
            await #expect(throws: SwiftMLError.self) {
                try await escaped.preparePackedForTrial([1, 4, 2, 5, 3, 6], source: input, budget: inputBudget)
            }
            #expect(await budget.reservedBytes == 0)
        }
    }
}
