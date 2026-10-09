import CoreML
import Foundation
import Testing
import SwiftML
import SwiftPreprocessing

@Suite("Core ML fixed matrix inference", .serialized)
struct CoreMLMatrixTests {
    @Test(arguments: [true, false], [false, true]) func preservesRowsColumnsAndAdmission(float32: Bool, asynchronous: Bool) async throws {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [3, 2], float32: float32)) { url in
            let predictor = try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x", "y"],
                outputName: "result", computeUnits: .cpuOnly, asynchronousMatrix: asynchronous)
            let raw = try PreparedNumericBatch(columnNames: ["y", "x"], columns: [[7, -3, 9], [1.25, 2, -1]])
            let input = try raw.selectingRows([2, 0, 2])
            let budget = try MemoryBudget(limit: 4096)
            let output = try await predictor.predict(input, budget: budget, workspaceBytes: 128)
            #expect(output.values.originalRowIndices == [2, 0, 2])
            #expect(output.values.columnNames == ["result[0]", "result[1]"])
            let middle = 1.25
            #expect(output.values.columnValues(at: 0) == [0, middle, 0])
            #expect(output.values.columnValues(at: 1) == [9, 7, 9])
            let elementBytes = float32 ? 4 : 8
            #expect(output.inputPackingBytes == 6 * elementBytes)
            #expect(output.outputCopyBytes == 48)
            let expectedPeak = input.payloadByteCount + 48 + 12 * elementBytes + 128
            #expect(await budget.peak == expectedPeak)
            #expect(await budget.reservedBytes == 0)
            let again = try await predictor.predict(raw, budget: budget, workspaceBytes: 128)
            #expect(again.values.columnValues(at: 1) == [7, 0, 9])
            #expect(output.values.columnValues(at: 1) == [9, 7, 9])
            #expect(raw[0, 1] == 1.25)
        }
    }

    @Test(arguments: [false, true]) func rejectsWrongRowsAndBatchingBeforeAdmission(asynchronous: Bool) async throws {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [3, 2])) { url in
            let predictor = try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x", "y"],
                outputName: "result", computeUnits: .cpuOnly, asynchronousMatrix: asynchronous)
            let budget = try MemoryBudget(limit: 4096)
            for rows in [0, 1, 2, 4] {
                let values = [Double](repeating: 1, count: rows)
                let input = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [values, values])
                await #expect(throws: (any Error).self) {
                    try await predictor.predict(input, budget: budget, workspaceBytes: 0)
                }
            }
            let input = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[1, 2, 3], [4, 5, 6]])
            await #expect(throws: (any Error).self) {
                try await predictor.predict(input, budget: budget, workspaceBytes: 0, maximumBatchSize: 2)
            }
            #expect(await budget.peak == 0)
            #expect(await budget.reservedBytes == 0)
        }
    }

    @Test(arguments: [false, true]) func rejectsMismatchedColumnsBeforeAdmission(asynchronous: Bool) async throws {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [3, 2])) { url in
            let predictor = try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x", "y"],
                outputName: "result", computeUnits: .cpuOnly, asynchronousMatrix: asynchronous)
            let budget = try MemoryBudget(limit: 4096)
            for names in [["x"], ["x", "x"], ["x", "z"], ["x", "y", "z"]] {
                let columns = names.map { _ in [1.0, 2.0, 3.0] }
                let input = try PreparedNumericBatch(columnNames: names, columns: columns)
                await #expect(throws: SwiftMLError.self) {
                    try await predictor.predict(input, budget: budget, workspaceBytes: 0)
                }
            }
            #expect(await budget.peak == 0)
            #expect(await budget.reservedBytes == 0)
        }
    }

    @Test func matrixLayoutRequiresMatrixModel() async throws {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [2])) { url in
            #expect(throws: (any Error).self) {
                try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x", "y"],
                    outputName: "result", inputLayout: .matrix)
            }
        }
    }

    @Test(arguments: [false, true]) func badValuesAndInsufficientBudgetLeaveNoReservation(asynchronous: Bool) async throws {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [3, 1])) { url in
            let predictor = try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x"],
                outputName: "result", computeUnits: .cpuOnly, asynchronousMatrix: asynchronous)
            let input = try PreparedNumericBatch(columnNames: ["x"], columns: [[1, 2, 3]])
            let budget = try MemoryBudget(limit: 4096)
            for value: Double? in [nil, .nan, .infinity, .greatestFiniteMagnitude] {
                var bad = input
                try bad.updateColumn(at: 0, rows: [2], values: [value])
                await #expect(throws: (any Error).self) {
                    try await predictor.predict(bad, budget: budget, workspaceBytes: 0)
                }
                #expect(await budget.reservedBytes == 0)
            }
            let peak = input.payloadByteCount + 24 + 12 + 12
            let tiny = try MemoryBudget(limit: peak - 1)
            await #expect(throws: MemoryAdmissionError.self) {
                try await predictor.predict(input, budget: tiny, workspaceBytes: 0)
            }
            #expect(await tiny.peak == 0)
            let exact = try MemoryBudget(limit: peak)
            let result = try await predictor.predict(input, budget: exact, workspaceBytes: 0)
            #expect(result.values.columnNames == ["result"])
            #expect(result.values.columnValues(at: 0) == [1, 2, 3])
            #expect(await exact.reservedBytes == 0)
        }
    }

    @Test(arguments: [false, true]) func concurrentRequestsKeepIndependentOwnedResults(asynchronous: Bool) async throws {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [3, 1])) { url in
            let budget = try MemoryBudget(limit: 4096)
            weak var owner: CoreMLPredictor?
            var retained: CoreMLPrediction?
            do {
                let predictor = try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x"],
                    outputName: "result", computeUnits: .cpuOnly, asynchronousMatrix: asynchronous)
                owner = predictor
                let first = try PreparedNumericBatch(columnNames: ["x"], columns: [[1, 2, 3]])
                let second = try PreparedNumericBatch(columnNames: ["x"], columns: [[4, 5, 6]])
                async let a = predictor.predict(first, budget: budget, workspaceBytes: 128)
                async let b = predictor.predict(second, budget: budget, workspaceBytes: 128)
                let (one, two) = try await (a, b)
                #expect(one.values.columnValues(at: 0) == [1, 2, 3])
                #expect(two.values.columnValues(at: 0) == [4, 5, 6])
                retained = one
            }
            #expect(owner == nil)
            #expect(retained?.values[2, 0] == 3)
            #expect(await budget.reservedBytes == 0)
        }
    }

    @Test(arguments: [false, true]) func queuedCancellationReleasesAdmission(asynchronous: Bool) async throws {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [3, 1])) { url in
            let predictor = try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x"],
                outputName: "result", computeUnits: .cpuOnly, asynchronousMatrix: asynchronous)
            let input = try PreparedNumericBatch(columnNames: ["x"], columns: [[1, 2, 3]])
            let budget = try MemoryBudget(limit: 4096)
            let blocker = try await budget.acquire(MemoryEstimate(capacities: [4096]))
            let request = Task { try await predictor.predict(input, budget: budget, workspaceBytes: 128) }
            let deadline = ContinuousClock.now.advanced(by: .seconds(5))
            while await budget.queuedCount == 0, ContinuousClock.now < deadline { await Task.yield() }
            let queued = await budget.queuedCount
            request.cancel()
            await blocker.finish()
            await #expect(throws: CancellationError.self) { try await request.value }
            #expect(queued == 1)
            #expect(await budget.reservedBytes == 0)
            let result = try await predictor.predict(input, budget: budget, workspaceBytes: 128)
            #expect(result.values[2, 0] == 3)
        }
    }
}
