import CoreML
import Foundation
import Testing
import SwiftML
import SwiftPreprocessing

@Suite("Scoped Core ML matrix pool", .serialized)
struct CoreMLMatrixPoolTests {
    @Test(arguments: [true, false]) func concurrentPredictionsPreserveOwnedResults(float32: Bool) async throws {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [513, 7], float32: float32)) { url in
            let names: [String] = (0..<7).map { "x\($0)" }
            let columns: [[Double]] = (0..<7).map { column -> [Double] in
                (0..<513).map { row -> Double in
                    let numerator: Int = (row + column) % 17 - 8
                    return Double(numerator) / 4.0
                }
            }
            let selection: [Int] = (0..<513).map { 512 - $0 / 2 }
            let input = try PreparedNumericBatch(columnNames: Array(names.reversed()), columns: Array(columns.reversed()))
                .selectingRows(selection)
            let serial = try CoreMLPredictor(compiledModelURL: url, inputColumns: names,
                outputName: "result", computeUnits: .cpuOnly, inputLayout: .matrix)
            let expected = try await serial.predict(input, budget: MemoryBudget(limit: 1_048_576), workspaceBytes: 128)
            let configuration = try CoreMLMatrixPool.Configuration(maximumConcurrentPredictions: 2,
                retainedBytes: 131_072, requestBytes: 131_072)
            let budget = try MemoryBudget(limit: 393_216)
            let (escaped, retained) = try await CoreMLMatrixPool.withPool(compiledModelURL: url,
                inputColumns: names, outputName: "result", computeUnits: .cpuOnly,
                budget: budget, configuration: configuration) { pool in
                #expect(await budget.reservedBytes == budget.limit)
                let retained = try await pool.predict(input)
                try await withThrowingTaskGroup(of: Void.self) { group in
                    for _ in 0..<8 {
                        group.addTask {
                            for _ in 0..<10 {
                                let result = try await pool.predict(input)
                                #expect(try result.values.matrix().values == expected.values.matrix().values)
                                #expect(result.values.originalRowIndices == input.originalRowIndices)
                                #expect(result.inputPackingBytes == expected.inputPackingBytes)
                                #expect(result.outputCopyBytes == expected.outputCopyBytes)
                            }
                        }
                    }
                    try await group.waitForAll()
                }
                #expect(await budget.reservedBytes == budget.limit)
                #expect(await pool.completedPredictions == 81)
                let matches = await pool.outputBackingIdentityMatches
                #expect((0...81).contains(matches))
                return (pool, retained)
            }
            #expect(await budget.reservedBytes == 0)
            #expect(try retained.values.matrix().values == expected.values.matrix().values)
            await #expect(throws: SwiftMLError.self) { try await escaped.predict(input) }
        }
    }

    @Test func invalidRequestsLeavePoolReusable() async throws {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [3, 2])) { url in
            let budget = try MemoryBudget(limit: 131_072)
            let configuration = try CoreMLMatrixPool.Configuration(retainedBytes: 65_536, requestBytes: 65_536)
            let input = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[1, 2, 3], [4, 5, 6]])
            try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: ["x", "y"],
                outputName: "result", computeUnits: .cpuOnly, budget: budget, configuration: configuration) { pool in
                for value: Double? in [nil, .nan, .infinity, .greatestFiniteMagnitude] {
                    var bad = input
                    try bad.updateColumn(at: 1, rows: [2], values: [value])
                    await #expect(throws: SwiftMLError.self) { try await pool.predict(bad) }
                }
                let short = try input.selectingRows([0])
                await #expect(throws: SwiftMLError.self) { try await pool.predict(short) }
                let renamed = try PreparedNumericBatch(columnNames: ["wrong", "y"], columns: [[1,2,3],[4,5,6]])
                await #expect(throws: SwiftMLError.self) { try await pool.predict(renamed) }
                let result = try await pool.predict(input)
                #expect(result.values[2,1] == 6)
            }
            #expect(await budget.reservedBytes == 0)
        }
    }

    @Test func errorsAndRequestUnderestimatesReturnQuota() async throws {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [3, 2])) { url in
            let budget = try MemoryBudget(limit: 131_072)
            for retained in [1, 65_536] {
                let configuration = try CoreMLMatrixPool.Configuration(retainedBytes: retained, requestBytes: 1)
                await #expect(throws: MemoryAdmissionError.self) {
                    try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: ["x", "y"],
                        outputName: "result", computeUnits: .cpuOnly, budget: budget, configuration: configuration) { pool in
                        let input = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[1,2,3],[4,5,6]])
                        _ = try await pool.predict(input)
                    }
                }
                #expect(await budget.reservedBytes == 0)
            }
            let configuration = try CoreMLMatrixPool.Configuration(retainedBytes: 65_536, requestBytes: 65_536)
            await #expect(throws: CancellationError.self) {
                try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: ["x", "y"],
                    outputName: "result", budget: budget, configuration: configuration) { _ in throw CancellationError() }
            }
            #expect(await budget.reservedBytes == 0)
            await #expect(throws: (any Error).self) {
                try await CoreMLMatrixPool.withPool(compiledModelURL: url.appendingPathComponent("missing"),
                    inputColumns: ["x", "y"], outputName: "result", budget: budget, configuration: configuration) { _ in }
            }
            #expect(await budget.reservedBytes == 0)
        }
    }

    @Test func queuedPoolCancellationDoesNotConsumeQuota() async throws {
        let budget = try MemoryBudget(limit: 131_072)
        let held = try await budget.acquire(MemoryEstimate(capacities: [131_072]))
        let configuration = try CoreMLMatrixPool.Configuration(retainedBytes: 65_536, requestBytes: 65_536)
        let attempt = Task {
            try await CoreMLMatrixPool.withPool(compiledModelURL: URL(fileURLWithPath: "/missing"),
                inputColumns: ["x"], outputName: "result", budget: budget, configuration: configuration) { _ in }
        }
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while await budget.queuedCount == 0, ContinuousClock.now < deadline { await Task.yield() }
        #expect(await budget.queuedCount == 1)
        attempt.cancel()
        await #expect(throws: CancellationError.self) { try await attempt.value }
        #expect(await budget.reservedBytes == 131_072)
        await held.finish()
        #expect(await budget.reservedBytes == 0)
    }

    @Test func invalidConfigurationFailsBeforeLoading() throws {
        #expect(throws: MemoryAdmissionError.self) {
            try CoreMLMatrixPool.Configuration(maximumConcurrentPredictions: 0, retainedBytes: 1, requestBytes: 1)
        }
        #expect(throws: MemoryAdmissionError.self) {
            try CoreMLMatrixPool.Configuration(maximumConcurrentPredictions: 2, retainedBytes: Int.max, requestBytes: 1)
        }
        #expect(throws: MemoryAdmissionError.self) {
            try CoreMLMatrixPool.Configuration(retainedBytes: -1, requestBytes: 1)
        }
    }
}
