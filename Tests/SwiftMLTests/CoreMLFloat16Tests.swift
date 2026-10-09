import CoreML
import Foundation
import Testing
import SwiftML
import SwiftPreprocessing

private func withHalfModel(_ name: String, body: (URL) async throws -> Void) async throws {
    let source = try #require(Bundle.module.url(forResource: name, withExtension: "mlmodel", subdirectory: "CoreMLFloat16"))
    try await withCompiledCoreML(Data(contentsOf: source), body: body)
}

@Suite("Core ML Float16 contracts", .serialized)
struct CoreMLFloat16Tests {
    @Test func availabilityPreservesExistingDeploymentTarget() async throws {
        try await withHalfModel("matrix") { url in
            if #available(macOS 15, *) {
                _ = try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x", "y"],
                    outputName: "result", computeUnits: .cpuOnly, inputLayout: .matrix)
            } else {
                #expect(throws: SwiftMLError.self) {
                    try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x", "y"],
                        outputName: "result", computeUnits: .cpuOnly, inputLayout: .matrix)
                }
            }
        }
    }

    @available(macOS 15, *)
    @Test(arguments: [false, true]) func matrixPreservesIdentityAndCountsTwoByteElements(asynchronous: Bool) async throws {
        try await withHalfModel("matrix") { url in
            let predictor = try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x", "y"],
                outputName: "result", computeUnits: .cpuOnly, asynchronousMatrix: asynchronous)
            let input = try PreparedNumericBatch(columnNames: ["y", "x"], columns: [[7, -3, 9], [1.1, 65504, -1]])
                .selectingRows([2, 0, 2])
            let budget = try MemoryBudget(limit: 4096)
            let output = try await predictor.predict(input, budget: budget, workspaceBytes: 128)
            #expect(output.values.originalRowIndices == [2, 0, 2])
            #expect(output.values.columnNames == ["result[0]", "result[1]"])
            #expect(output.values.columnValues(at: 0) == [0, Double(Float16(1.1)), 0])
            #expect(output.values.columnValues(at: 1) == [9, 7, 9])
            #expect(output.inputPackingBytes == 12)
            #expect(output.outputCopyBytes == 48)
            #expect(await budget.peak == input.payloadByteCount + 48 + 24 + 128)
            #expect(await budget.reservedBytes == 0)
            var changed = input
            try changed.updateColumn(at: 1, rows: [0], values: [3.5])
            _ = try await predictor.predict(changed, budget: budget, workspaceBytes: 128)
            #expect(output.values.columnValues(at: 0) == [0, Double(Float16(1.1)), 0])
        }
    }

    @available(macOS 15, *)
    @Test(arguments: [1, 2]) func vectorSupportsSingleAndBatchedExamples(batchSize: Int) async throws {
        try await withHalfModel("vector") { url in
            let predictor = try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x", "y"],
                outputName: "result", computeUnits: .cpuOnly)
            let input = try PreparedNumericBatch(columnNames: ["y", "x"], columns: [[-2, 4], [1.1, 65504]])
            let budget = try MemoryBudget(limit: 4096)
            let output = try await predictor.predict(input, budget: budget, workspaceBytes: 0, maximumBatchSize: batchSize)
            #expect(output.values.columnValues(at: 0) == [Double(Float16(1.1)), 65504])
            #expect(output.values.columnValues(at: 1) == [0, 4])
            #expect(output.inputPackingBytes == 8)
            #expect(output.outputCopyBytes == 32)
            #expect(await budget.reservedBytes == 0)
        }
    }

    @available(macOS 15, *)
    @Test(arguments: ["matrix", "vector"]) func invalidValuesReleaseAdmissionAndAllowRecovery(layout: String) async throws {
        try await withHalfModel(layout) { url in
            let predictor = try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x", "y"],
                outputName: "result", computeUnits: .cpuOnly, inputLayout: layout == "matrix" ? .matrix : .examples)
            let input = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[1, 2, 3], [4, 5, 6]])
            let budget = try MemoryBudget(limit: 4096)
            for value: Double? in [nil, .nan, .infinity, 65520, -65520] {
                var bad = input
                try bad.updateColumn(at: 1, rows: [2], values: [value])
                await #expect(throws: SwiftMLError.self) { try await predictor.predict(bad, budget: budget, workspaceBytes: 0) }
                #expect(await budget.reservedBytes == 0)
            }
            let result = try await predictor.predict(input, budget: budget, workspaceBytes: 0)
            #expect(result.values.columnValues(at: 1) == [4, 5, 6])
            #expect(await budget.reservedBytes == 0)
        }
    }

    @available(macOS 15, *)
    @Test(arguments: [1, 4]) func poolConcurrentRequestsKeepOwnedResults(slots: Int) async throws {
        try await withHalfModel("matrix") { url in
            let input = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[1.1, -2, 3], [4, 5, 6]])
            let configuration = try CoreMLMatrixPool.Configuration(maximumConcurrentPredictions: slots,
                retainedBytes: 1_048_576, requestBytes: 65536)
            let budget = try MemoryBudget(limit: 1_048_576 + slots * 65536)
            let (escaped, retained) = try await CoreMLMatrixPool.withPool(compiledModelURL: url,
                inputColumns: ["x", "y"], outputName: "result", computeUnits: .cpuOnly,
                budget: budget, configuration: configuration) { pool in
                var bad = input
                try bad.updateColumn(at: 0, rows: [1], values: [65520])
                await #expect(throws: SwiftMLError.self) { try await pool.predict(bad) }
                let retained = try await pool.predict(input)
                try await withThrowingTaskGroup(of: Void.self) { group in
                    for caller in 0..<8 {
                        group.addTask {
                            var changed = input
                            try changed.updateColumn(at: 0, rows: [1], values: [Double(caller)])
                            let result = try await pool.predict(changed)
                            #expect(result.values.columnValues(at: 0) == [Double(Float16(1.1)), Double(caller), 3])
                            #expect(result.inputPackingBytes == 12)
                        }
                    }
                    try await group.waitForAll()
                }
                #expect(await pool.completedPredictions == 9)
                return (pool, retained)
            }
            #expect(await budget.reservedBytes == 0)
            #expect(retained.values.columnValues(at: 0) == [Double(Float16(1.1)), 0, 3])
            await #expect(throws: SwiftMLError.self) { try await escaped.predict(input) }
        }
    }
}
