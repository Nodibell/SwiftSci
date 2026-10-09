import CoreML
import Testing
import SwiftML
import SwiftPreprocessing

@Suite("Fitted Core ML input preparation", .serialized)
struct CoreMLInputPreparationTests {
    @available(macOS 15, *)
    @Test(arguments: [2, 4, 8]) func directMatchesStagedAndDrainsConcurrentOwners(bytes: Int) async throws {
        try await withPreparedMatrixModel(bytes) { url in
            let input = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[1, 2, .nan], [4, 5, 6]])
                .selectingRows([2, 0, 2])
            let training = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[0, 1, 2], [3, 4, 5]])
            let plan = try StandardPreprocessingPlan(training: training)
            let budget = try MemoryBudget(limit: 2_097_152)
            let inputBudget = try MemoryBudget(limit: 1_048_576)
            let config = try CoreMLMatrixPool.Configuration(maximumConcurrentPredictions: 2,
                retainedBytes: 1_048_576, requestBytes: 65_536)
            try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: ["x", "y"],
                outputName: "result", computeUnits: .cpuOnly, budget: budget, configuration: config) { pool in
                let contract = try await pool.inputPreparation()
                var imputer = Imputer()
                try imputer.fit(training)
                var scaler = StandardScaler()
                try scaler.fit(imputer.transform(training))
                let scaled = try scaler.transform(imputer.transform(input))
                let reference = try await pool.prepare(scaled, budget: inputBudget)
                #expect(contract.rowCount == 3)
                #expect(contract.columnNames == ["x", "y"])
                let expected = try await pool.predict(reference)
                try await withThrowingTaskGroup(of: Void.self) { group in
                    for _ in 0..<4 {
                        group.addTask {
                            let direct = try await contract.prepare(input, preprocessing: plan, budget: inputBudget)
                            let actual = try await pool.predict(direct)
                            #expect(actual.values.originalRowIndices == [2, 0, 2])
                            #expect(try actual.values.matrix().values == expected.values.matrix().values)
                        }
                    }
                    try await group.waitForAll()
                }
                let bad = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[Double.infinity, 2, 3], [4, 5, 6]])
                await #expect(throws: SwiftMLError.self) {
                    try await contract.prepare(bad, preprocessing: plan, budget: inputBudget)
                }
                let task = Task {
                    try Task.checkCancellation()
                    return try await contract.prepare(input, preprocessing: plan, budget: inputBudget)
                }
                // Cancellation can race with completion. Either path must release its owner.
                task.cancel()
                do { _ = try await task.value } catch is CancellationError { }
            }
            await waitForPreparedRelease(inputBudget)
            #expect(await budget.reservedBytes == 0)
        }
    }

    @Test func consumingReferenceReusesUniqueColumns() throws {
        let training = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[0, 1, 2], [3, 4, 5]])
        let plan = try StandardPreprocessingPlan(training: training)
        let input = try training.selectingRows([2, 0, 2])
        let addresses = input.columns.map { $0.values.withUnsafeBufferPointer { UInt(bitPattern: $0.baseAddress!) } }
        let imputed = try plan.imputer.transform(consuming: consume input)
        let scaled = try plan.scaler.transform(consuming: consume imputed)
        let after = scaled.columns.map { $0.values.withUnsafeBufferPointer { UInt(bitPattern: $0.baseAddress!) } }
        #expect(addresses == after)
    }
}
