import CoreML
import Testing
import SwiftML
import SwiftPreprocessing

@Suite("Core ML bounded batches", .serialized)
struct CoreMLBatchTests {
    @Test func vectorChunksKeepIndependentRowsAndTail() async throws {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [2])) { url in
            let predictor = try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x", "y"],
                                                outputName: "result", computeUnits: .cpuOnly)
            let raw = try PreparedNumericBatch(columnNames: ["y", "x"],
                                               columns: [[7, -3, 9, 4, 5], [1, 2, -1, 4, 6]])
            let input = try raw.selectingRows([4, 1, 0, 4, 2])
            for size in [1, 2, 4, 5, 8, Int.max] {
                let budget = try MemoryBudget(limit: 8192)
                let output = try await predictor.predict(input, budget: budget, workspaceBytes: 512,
                                                         maximumBatchSize: size)
                #expect(output.values.originalRowIndices == [4, 1, 0, 4, 2])
                #expect(output.values.columnNames == ["result[0]", "result[1]"])
                for row in 0..<5 {
                    #expect(output.values[row, 0] == max(try #require(input[row, 1]), 0))
                    #expect(output.values[row, 1] == max(try #require(input[row, 0]), 0))
                }
                #expect(output.inputPackingBytes == 40)
                #expect(output.outputCopyBytes == 80)
                let expectedPeak = input.payloadByteCount + 80 + min(size, 5) * 16 + 512
                #expect(await budget.peak == expectedPeak)
                #expect(await budget.reservedBytes == 0)
            }
        }
    }

    @Test func scalarChunksAndEmptyInput() async throws {
        let artifact = CoreMLExporter.exportBinaryLinearModel(inputNames: ["x"], outputName: "result",
                                                              weights: [2], bias: 1)
        try await withCompiledCoreML(artifact) { url in
            let predictor = try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x"],
                                                outputName: "result", computeUnits: .cpuOnly)
            let budget = try MemoryBudget(limit: 8192)
            let input = try PreparedNumericBatch(columnNames: ["x"], columns: [[1, 2, 3, 4, 5]])
            let output = try await predictor.predict(input, budget: budget, workspaceBytes: 512, maximumBatchSize: 2)
            #expect(output.values.columnValues(at: 0) == [3, 5, 7, 9, 11])
            #expect(output.inputPackingBytes == 0)
            let empty = try PreparedNumericBatch(columnNames: ["x"], columns: [[]])
            let result = try await predictor.predict(empty, budget: budget, workspaceBytes: 0, maximumBatchSize: 64)
            #expect(result.values.rowCount == 0)
            #expect(await budget.reservedBytes == 0)
        }
    }

    @Test func failuresReleaseBatchAdmission() async throws {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [2])) { url in
            let predictor = try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x", "y"],
                                                outputName: "result", computeUnits: .cpuOnly)
            let input = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
            let budget = try MemoryBudget(limit: 8192)
            for size in [0, -1] {
                await #expect(throws: (any Error).self) {
                    try await predictor.predict(input, budget: budget, workspaceBytes: 512, maximumBatchSize: size)
                }
            }
            for value: Double? in [nil, .nan, .greatestFiniteMagnitude] {
                var bad = input
                try bad.updateColumn(at: 0, rows: [3], values: [value])
                await #expect(throws: (any Error).self) {
                    try await predictor.predict(bad, budget: budget, workspaceBytes: 512, maximumBatchSize: 2)
                }
                #expect(await budget.reservedBytes == 0)
                #expect(input[3, 0] == 4)
            }
            let tiny = try MemoryBudget(limit: input.payloadByteCount + 80 + 16)
            await #expect(throws: MemoryAdmissionError.self) {
                try await predictor.predict(input, budget: tiny, workspaceBytes: 0, maximumBatchSize: 2)
            }
            let result = try await predictor.predict(input, budget: budget, workspaceBytes: 512, maximumBatchSize: 2)
            #expect(result.values[4, 1] == 10)
            #expect(await budget.reservedBytes == 0)
        }
    }
}
