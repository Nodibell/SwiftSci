import Foundation
import SwiftML
import SwiftPreprocessing

func pipelineFixture() throws -> (PreparedNumericBatch, StandardPreprocessingPlan) {
    let training = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[0, 1, 2], [3, 4, 5]])
    let input = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[2, .nan, 0], [5, 4, 3]])
    return (try input.selectingRows([2, 0, 2]), try StandardPreprocessingPlan(training: training))
}

func pipelineConfiguration(window: Int = 2, slots: Int = 1) throws -> CoreMLBatchPipeline.Configuration {
    let pool = try CoreMLMatrixPool.Configuration(maximumConcurrentPredictions: slots,
        retainedBytes: 1_048_576, requestBytes: 65_536)
    return try CoreMLBatchPipeline.Configuration(pool: pool, maximumInFlightBatches: window,
        retainedBytes: 65_536, batchBytes: 262_144)
}

actor PipelineObservations {
    private(set) var loads = 0
    private(set) var sequences: [Int] = []
    private(set) var values: [[Double]] = []
    private(set) var rows: [[Int]] = []
    func loaded() { loads += 1 }
    func consumed(_ sequence: Int, _ result: CoreMLPrediction) throws {
        sequences.append(sequence)
        values.append(try result.values.matrix().values)
        rows.append(result.values.originalRowIndices)
    }
}

actor PipelineGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private var open = false
    private(set) var entered = false
    func wait() async {
        entered = true
        if !open { await withCheckedContinuation { continuation = $0 } }
    }
    func release() { open = true; continuation?.resume(); continuation = nil }
}

enum PipelineTestFailure: Error { case source, consumer }
