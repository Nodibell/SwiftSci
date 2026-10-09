import CoreML
import Testing
import SwiftML
import SwiftPreprocessing

private actor PoolStartGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private var open = false
    private(set) var entered = false
    func wait() async {
        entered = true
        if !open { await withCheckedContinuation { continuation = $0 } }
    }
    func release() { open = true; continuation?.resume(); continuation = nil }
}

@Suite("Core ML matrix pool shared admission", .serialized)
struct CoreMLMatrixPoolAdmissionTests {
    @Test func admittedPoolProgressesWhilePeerWaits() async throws {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [3, 1])) { url in
            let budget = try MemoryBudget(limit: 131_072)
            let configuration = try CoreMLMatrixPool.Configuration(retainedBytes: 65_536, requestBytes: 65_536)
            let input = try PreparedNumericBatch(columnNames: ["x"], columns: [[1, 2, 3]])
            let gate = PoolStartGate()
            let first = Task {
                try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: ["x"],
                    outputName: "result", computeUnits: .cpuOnly, budget: budget, configuration: configuration) { pool in
                    await gate.wait()
                    for _ in 0..<10 { _ = try await pool.predict(input) }
                }
            }
            let deadline = ContinuousClock.now.advanced(by: .seconds(5))
            while !(await gate.entered), ContinuousClock.now < deadline { await Task.yield() }
            #expect(await gate.entered)
            let second = Task {
                try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: ["x"],
                    outputName: "result", computeUnits: .cpuOnly, budget: budget, configuration: configuration) { pool in
                    try await pool.predict(input)
                }
            }
            let queueDeadline = ContinuousClock.now.advanced(by: .seconds(5))
            while await budget.queuedCount == 0, ContinuousClock.now < queueDeadline { await Task.yield() }
            #expect(await budget.queuedCount == 1)
            #expect(await budget.reservedBytes == budget.limit)
            await gate.release()
            try await first.value
            let result = try await second.value
            #expect(result.values[2,0] == 3)
            #expect(await budget.reservedBytes == 0)
        }
    }

    @Test func canceledPredictionsDrainBeforeScopeExit() async throws {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [513, 7])) { url in
            let names = (0..<7).map { "x\($0)" }
            let input = try PreparedNumericBatch(columnNames: names,
                columns: [[Double]](repeating: [Double](repeating: 1, count: 513), count: 7))
            let configuration = try CoreMLMatrixPool.Configuration(retainedBytes: 131_072, requestBytes: 131_072)
            let budget = try MemoryBudget(limit: 262_144)
            try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: names,
                outputName: "result", computeUnits: .cpuOnly, budget: budget, configuration: configuration) { pool in
                let attempts = (0..<32).map { _ in Task { try await pool.predict(input) } }
                for attempt in attempts { attempt.cancel() }
                for attempt in attempts {
                    do { _ = try await attempt.value } catch is CancellationError {}
                }
                let result = try await pool.predict(input)
                #expect(result.values[512,6] == 1)
                #expect(await budget.reservedBytes == budget.limit)
            }
            #expect(await budget.reservedBytes == 0)
        }
    }
}
