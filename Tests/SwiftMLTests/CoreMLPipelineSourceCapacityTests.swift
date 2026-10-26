import Testing
@testable import SwiftML
import SwiftDataFrame
import SwiftPreprocessing

@Suite("Pipeline source storage accounting")
struct CoreMLPipelineSourceCapacityTests {
    @Test func identityProvenanceDoesNotChargeUnretainedIndices() throws {
        let source = try PreparedNumericBatch(columnNames: ["x"], columns: [[1, 2, 3]])
        let expected = payloadAndNames(source)
        #expect(try CoreMLBatchPipeline.sourceCapacity(source) == expected)
    }

    @Test func selectedProvenanceChargesRetainedCapacity() throws {
        let input = try PreparedNumericBatch(columnNames: ["x"], columns: [[1, 2, 3]])
        let source = try input.selectingRows([2, 0, 2])
        let rows = source.originalRowIndices
        let expected = payloadAndNames(source) + rows.capacity * MemoryLayout<Int>.stride
        #expect(rows == [2, 0, 2])
        #expect(try CoreMLBatchPipeline.sourceCapacity(source) == expected)
    }

    @Test func storedIndicesStillCountAgainstAdmission() throws {
        let source = try PreparedNumericBatch(columnNames: ["x"], columns: [[1, 2, 3]])
        let selected = try source.selectingRows([2, 0, 2])
        let identityBytes = try CoreMLBatchPipeline.sourceCapacity(source)
        let selectedBytes = try CoreMLBatchPipeline.sourceCapacity(selected)
        let pool = try CoreMLMatrixPool.Configuration(maximumConcurrentPredictions: 1,
            retainedBytes: 1, requestBytes: 1)
        let configuration = try CoreMLBatchPipeline.Configuration(pool: pool,
            maximumInFlightBatches: 1, retainedBytes: 1, batchBytes: identityBytes + 1)
        try CoreMLBatchPipeline.requireBatchCapacity(sourceBytes: identityBytes,
            preparationBytes: 0, configuration: configuration)
        #expect(throws: MemoryAdmissionError.self) {
            try CoreMLBatchPipeline.requireBatchCapacity(sourceBytes: selectedBytes,
                preparationBytes: 0, configuration: configuration)
        }
    }

    private func payloadAndNames(_ source: PreparedNumericBatch) -> Int {
        var bytes = source.columnNames.capacity * MemoryLayout<String>.stride
        bytes += source.columnNames.reduce(0) { $0 + $1.utf8.count }
        for column in source.columns {
            bytes += column.values.capacity * MemoryLayout<Double>.stride
            bytes += (column.validity?.capacity ?? 0) * MemoryLayout<UInt64>.stride
        }
        return bytes
    }
}
