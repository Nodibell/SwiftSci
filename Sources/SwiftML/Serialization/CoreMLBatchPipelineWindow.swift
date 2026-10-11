import SwiftDataFrame
import SwiftPreprocessing

extension CoreMLBatchPipeline {
    static func requireBatchCapacity(sourceBytes: Int, preparationBytes: Int,
                                     configuration: Configuration) throws {
        let required = try MemoryEstimate(capacities: [sourceBytes, preparationBytes,
            configuration.pool.requestBytes]).bytes
        guard required <= configuration.batchBytes else {
            throw MemoryAdmissionError.exceedsLimit(required: required, limit: configuration.batchBytes)
        }
    }

    private static func sourceCapacity(_ source: PreparedNumericBatch) throws -> Int {
        var capacities: [Int] = []
        for column in source.columns {
            capacities.append(try coreMLByteCount(column.values.capacity, MemoryLayout<Double>.stride))
            capacities.append(try coreMLByteCount(column.validity?.capacity ?? 0, MemoryLayout<UInt64>.stride))
        }
        capacities.append(try coreMLByteCount(source.originalRowIndices.capacity, MemoryLayout<Int>.stride))
        capacities.append(try coreMLByteCount(source.columnNames.capacity, MemoryLayout<String>.stride))
        capacities.append(contentsOf: source.columnNames.map { $0.utf8.count })
        return try MemoryEstimate(capacities: capacities).bytes
    }

    static func runWindow(pool: CoreMLMatrixPool, preparation: CoreMLMatrixInputPreparation,
        plan: StandardPreprocessingPlan, preparationBytes: Int, configuration: Configuration,
        load: @Sendable (Request) async throws -> PreparedNumericBatch?,
        consume: @Sendable (Int, CoreMLPrediction) async throws -> Void) async throws {
        try await withThrowingTaskGroup(of: (Int, CoreMLPrediction).self) { group in
            var nextLoad = 0
            var nextConsume = 0
            var exhausted = false
            var completed: [Int: CoreMLPrediction] = [:]
            while true {
                while !exhausted && nextLoad - nextConsume < configuration.maximumInFlightBatches {
                    try Task.checkCancellation()
                    let request = Request(sequence: nextLoad, rowCount: preparation.rowCount,
                                          columnNames: preparation.columnNames)
                    guard let source = try await load(request) else { exhausted = true; break }
                    try Task.checkCancellation()
                    guard source.rowCount == request.rowCount, source.columnNames == request.columnNames else {
                        throw SwiftMLError.invalidParameter("Pipeline batch must match the complete fixed model shape and column order")
                    }
                    try requireBatchCapacity(sourceBytes: sourceCapacity(source), preparationBytes: preparationBytes,
                        configuration: configuration)
                    let sequence = nextLoad
                    guard nextLoad < Int.max else { throw MemoryAdmissionError.invalidCapacity }
                    nextLoad += 1
                    group.addTask {
                        try Task.checkCancellation()
                        let localBudget = try MemoryBudget(limit: preparationBytes)
                        let owner = try await preparation.prepare(source, preprocessing: plan, budget: localBudget)
                        do {
                            let prediction = try await pool.predict(owner)
                            await owner.finishPipelineOwnership()
                            return (sequence, prediction)
                        } catch {
                            await owner.finishPipelineOwnership()
                            throw error
                        }
                    }
                }
                if nextConsume == nextLoad { break }
                if let result = completed.removeValue(forKey: nextConsume) {
                    try Task.checkCancellation()
                    try await consume(nextConsume, result)
                    nextConsume += 1
                } else if let (sequence, result) = try await group.next() {
                    completed[sequence] = result
                }
            }
        }
    }
}
