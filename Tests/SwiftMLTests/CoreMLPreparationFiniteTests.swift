import CoreML
import Testing
import SwiftML
import SwiftPreprocessing

@Suite("Prepared Core ML finite boundaries", .serialized)
struct CoreMLPreparationFiniteTests {
    @Test(arguments: [4, 8]) func rejectsNonfiniteValuesAcrossScanLanes(bytes: Int) async throws {
        let names = (0..<7).map { "x\($0)" }
        let trainingColumns = [[Double]](repeating: [-1, 1], count: names.count)
        let training = try PreparedNumericBatch(columnNames: names, columns: trainingColumns)
        let plan = try StandardPreprocessingPlan(training: training)
        let inputBudget = try MemoryBudget(limit: 1_048_576)
        let poolBudget = try MemoryBudget(limit: 2_097_152)
        let config = try CoreMLMatrixPool.Configuration(retainedBytes: 1_048_576, requestBytes: 131_072)
        let artifact = coreMLReLUArtifact(shape: [33, 7], float32: bytes == 4)
        try await withCompiledCoreML(artifact) { url in
            try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: names,
                outputName: "result", computeUnits: .cpuOnly, budget: poolBudget, configuration: config) { pool in
                let preparation = try await pool.inputPreparation()
                var invalid: [Double] = [.infinity, -.infinity]
                if bytes == 4 { invalid.append(Double.greatestFiniteMagnitude) }
                for index in [0, 7, 15, 16, 127, 128, 230] {
                    for value in invalid {
                        var columns = [[Double]](repeating: [Double](repeating: 1, count: 33), count: 7)
                        columns[index % 7][index / 7] = value
                        let source = try PreparedNumericBatch(columnNames: names, columns: columns)
                        await #expect(throws: SwiftMLError.self) {
                            try await preparation.prepare(source, preprocessing: plan, budget: inputBudget)
                        }
                        #expect(await inputBudget.reservedBytes == 0)
                        #expect(await inputBudget.queuedCount == 0)
                    }
                }
            }
        }
        #expect(await poolBudget.reservedBytes == 0)
    }

    @available(macOS 15, *)
    @Test func rejectsFloat16OverflowAfterConversion() async throws {
        try await withPreparedMatrixModel(2) { url in
            let names = ["x", "y"]
            let training = try PreparedNumericBatch(columnNames: names, columns: [[-1, 1], [-1, 1]])
            let plan = try StandardPreprocessingPlan(training: training)
            let inputBudget = try MemoryBudget(limit: 1_048_576)
            let poolBudget = try MemoryBudget(limit: 2_097_152)
            let config = try CoreMLMatrixPool.Configuration(retainedBytes: 1_048_576, requestBytes: 131_072)
            try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: names,
                outputName: "result", computeUnits: .cpuOnly, budget: poolBudget, configuration: config) { pool in
                let preparation = try await pool.inputPreparation()
                for index in 0..<6 {
                    var columns: [[Double]] = [[1, 1, 1], [1, 1, 1]]
                    columns[index % 2][index / 2] = 65520
                    let source = try PreparedNumericBatch(columnNames: names, columns: columns)
                    await #expect(throws: SwiftMLError.self) {
                        try await preparation.prepare(source, preprocessing: plan, budget: inputBudget)
                    }
                    #expect(await inputBudget.reservedBytes == 0)
                }
            }
            #expect(await poolBudget.reservedBytes == 0)
        }
    }

    @available(macOS 15, *)
    @Test(arguments: [2, 4, 8]) func rejectsNaNProducedByScaling(bytes: Int) async throws {
        try await withPreparedMatrixModel(bytes) { url in
            let names = ["x", "y"]
            let training = try PreparedNumericBatch(columnNames: names, columns: [[.infinity, .infinity], [-1, 1]])
            let plan = try StandardPreprocessingPlan(training: training)
            let source = try PreparedNumericBatch(columnNames: names, columns: [[1, 1, 1], [1, 1, 1]])
            let inputBudget = try MemoryBudget(limit: 1_048_576)
            let poolBudget = try MemoryBudget(limit: 2_097_152)
            let config = try CoreMLMatrixPool.Configuration(retainedBytes: 1_048_576, requestBytes: 131_072)
            try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: names,
                outputName: "result", computeUnits: .cpuOnly, budget: poolBudget, configuration: config) { pool in
                let preparation = try await pool.inputPreparation()
                await #expect(throws: SwiftMLError.self) {
                    try await preparation.prepare(source, preprocessing: plan, budget: inputBudget)
                }
                #expect(await inputBudget.reservedBytes == 0)
            }
            #expect(await poolBudget.reservedBytes == 0)
        }
    }

}
