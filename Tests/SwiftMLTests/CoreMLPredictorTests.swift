import CoreML
import Foundation
import Testing
import SwiftDataFrame
import SwiftML
import SwiftPreprocessing

@Suite("Core ML prepared inference", .serialized)
struct CoreMLPredictorTests {
    private func linearArtifact() -> Data {
        CoreMLExporter.exportBinaryLinearModel(inputNames: ["x", "y"], outputName: "result",
                                              weights: [2, -3], bias: 4)
    }

    private func batch() throws -> PreparedNumericBatch {
        try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[1, 2, 3], [4, 5, 6]])
    }

    @Test func trainExportReloadAndPredictHeldOutRows() async throws {
        let x: [Double] = (0..<64).map { Double($0) / 8 }
        let y: [Double] = (0..<64).map { row -> Double in Double((row * 7) % 13) }
        let training = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [x, y])
        var scaler = StandardScaler()
        try scaler.fit(training)
        let clean = try scaler.transform(training)
        let targets: [Double] = (0..<64).map { row -> Double in 2 * x[row] - 3 * y[row] + 4 }
        let cpu = LinearRegression(device: .cpu)
        try await cpu.fit(features: clean, targets: targets)
        let data = try await cpu.exportCoreML(featureNames: ["x", "y"], outputName: "result")
        let raw = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[8, 10, -5], [7, 4, 2]])
        let heldOut = try scaler.transform(raw).selectingRows([2, 0, 2, 1])
        let expected = try await cpu.predict(features: heldOut)
        let budget = try MemoryBudget(limit: 1_048_576)
        try await withCompiledCoreML(data) { compiled in
            let predictor = try CoreMLPredictor(compiledModelURL: compiled, inputColumns: ["x", "y"],
                                                outputName: "result", computeUnits: .cpuOnly)
            for _ in 0..<3 {
                let result = try await predictor.predict(heldOut, budget: budget, workspaceBytes: 65_536)
                #expect(result.values.originalRowIndices == [2, 0, 2, 1])
                #expect(result.values.columnNames == ["result"])
                #expect(result.inputPackingBytes == 0)
                #expect(result.outputCopyBytes == expected.count * 8)
                for row in expected.indices {
                    #expect(abs(try #require(result.values[row, 0]) - expected[row]) < 1e-10)
                }
                #expect(await budget.reservedBytes == 0)
            }
        }
    }

    @Test func vectorPredictionsPreserveOrderAndCountCopies() async throws {
        let layer = LayerWeights(W: [2, -1, -3, 0.5], b: [4, 1], inDim: 2, outDim: 2)
        let data = CoreMLExporter.exportBinaryMLPRegressor(inputNames: ["features"], outputName: "result", layers: [layer])
        // Physical column order differs from the vector's declared feature order.
        let input = try PreparedNumericBatch(columnNames: ["y", "x"], columns: [[4, 5, 6], [1, 2, 3]])
        let budget = try MemoryBudget(limit: 1_048_576)
        try await withCompiledCoreML(data) { compiled in
            for units in [MLComputeUnits.cpuOnly, .cpuAndGPU, .cpuAndNeuralEngine] {
                let predictor = try CoreMLPredictor(compiledModelURL: compiled, inputColumns: ["x", "y"],
                                                    outputName: "result", computeUnits: units)
                let result = try await predictor.predict(input, budget: budget, workspaceBytes: 65_536, maximumBatchSize: 2)
                #expect(result.values.columnNames == ["result[0]", "result[1]"])
                #expect(result.inputPackingBytes == 48)
                #expect(result.outputCopyBytes == 48)
                for row in 0..<3 {
                    let x = Double(row + 1)
                    let y = Double(row + 4)
                    let first: Double = 2 * x - 3 * y + 4
                    let second: Double = -x + 0.5 * y + 1
                    #expect(abs(try #require(result.values[row, 0]) - first) < 1e-5)
                    #expect(abs(try #require(result.values[row, 1]) - second) < 1e-5)
                }
                #expect(await budget.reservedBytes == 0)
            }
        }
    }

    @Test func invalidInputsReleaseAdmissionAndLeaveSourceUntouched() async throws {
        let budget = try MemoryBudget(limit: 1_048_576)
        try await withCompiledCoreML(linearArtifact()) { compiled in
            let predictor = try CoreMLPredictor(compiledModelURL: compiled, inputColumns: ["x", "y"], outputName: "result")
            for value: Double? in [nil, .nan, .infinity] {
                var input = try batch()
                try input.updateColumn(at: 0, rows: [2], values: [value])
                await #expect(throws: (any Error).self) {
                    try await predictor.predict(input, budget: budget, workspaceBytes: 65_536)
                }
                #expect(await budget.reservedBytes == 0)
                #expect(input[0, 0] == 1)
                #expect(input.nullCount(inColumn: 0) == (value == nil ? 1 : 0))
            }
            let result = try await predictor.predict(batch(), budget: budget, workspaceBytes: 65_536)
            #expect(result.values[0, 0] == -6)
        }
    }

    @Test func schemaAdmissionAndEmptyInput() async throws {
        try await withCompiledCoreML(linearArtifact()) { compiled in
            #expect(throws: (any Error).self) {
                try CoreMLPredictor(compiledModelURL: compiled, inputColumns: ["x", "x"], outputName: "result")
            }
            #expect(throws: (any Error).self) {
                try CoreMLPredictor(compiledModelURL: compiled, inputColumns: ["x", "z"], outputName: "result")
            }
            #expect(throws: (any Error).self) {
                try CoreMLPredictor(compiledModelURL: compiled, inputColumns: ["x", "y"], outputName: "absent")
            }
            let predictor = try CoreMLPredictor(compiledModelURL: compiled, inputColumns: ["x", "y"], outputName: "result")
            let tiny = try MemoryBudget(limit: 1)
            await #expect(throws: MemoryAdmissionError.self) {
                try await predictor.predict(batch(), budget: tiny, workspaceBytes: 0)
            }
            let budget = try MemoryBudget(limit: 4096)
            await #expect(throws: MemoryAdmissionError.self) {
                try await predictor.predict(batch(), budget: budget, workspaceBytes: -1)
            }
            let wrong = try PreparedNumericBatch(columnNames: ["x", "z"], columns: [[1], [2]])
            await #expect(throws: (any Error).self) {
                try await predictor.predict(wrong, budget: budget, workspaceBytes: 0)
            }
            let empty = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[], []])
            let result = try await predictor.predict(empty, budget: budget, workspaceBytes: 0)
            #expect(result.values.rowCount == 0)
            #expect(result.outputCopyBytes == 0)
            #expect(await budget.reservedBytes == 0)
        }
    }

    @Test func queuedCancellationReturnsReservation() async throws {
        try await withCompiledCoreML(linearArtifact()) { compiled in
            let predictor = try CoreMLPredictor(compiledModelURL: compiled, inputColumns: ["x", "y"], outputName: "result")
            let budget = try MemoryBudget(limit: 4096)
            let blocker = try await budget.acquire(MemoryEstimate(capacities: [4096]))
            let input = try batch()
            let request = Task { try await predictor.predict(input, budget: budget, workspaceBytes: 128, maximumBatchSize: 2) }
            let deadline = ContinuousClock.now.advanced(by: .seconds(5))
            while await budget.queuedCount == 0, ContinuousClock.now < deadline { await Task.yield() }
            let queued = await budget.queuedCount
            request.cancel()
            await blocker.finish()
            await #expect(throws: CancellationError.self) { try await request.value }
            #expect(queued == 1)
            #expect(await budget.queuedCount == 0)
            #expect(await budget.reservedBytes == 0)
            let result = try await predictor.predict(input, budget: budget, workspaceBytes: 128)
            #expect(result.values[0, 0] == -6)
        }
    }

    @Test func concurrentCallersShareModelWithoutSharingMutableBuffers() async throws {
        try await withCompiledCoreML(linearArtifact()) { compiled in
            let predictor = try CoreMLPredictor(compiledModelURL: compiled, inputColumns: ["x", "y"], outputName: "result")
            let budget = try MemoryBudget(limit: 4096)
            let input = try batch()
            try await withThrowingTaskGroup(of: Void.self) { group in
                for _ in 0..<8 {
                    group.addTask {
                        let result = try await predictor.predict(input, budget: budget, workspaceBytes: 128, maximumBatchSize: 2)
                        #expect(result.values[2, 0] == -8)
                    }
                }
                try await group.waitForAll()
            }
            #expect(await budget.reservedBytes == 0)
            #expect(input[0, 0] == 1)
        }
    }
}
