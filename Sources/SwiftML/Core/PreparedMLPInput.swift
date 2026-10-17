import SwiftDataFrame
import SwiftPreprocessing

/// Immutable CPU input. Idle owners retain storage without a budget reservation.
package final class PreparedMLPInput: Sendable {
    let batch: IndexedNumericBatch
    let preprocessing: StandardPreprocessingPlan
    let logicalPositions: [Int]?
    package let retainedBytes: Int
    package let preparationPeakBytes: Int
    package var rowCount: Int { batch.rowCount }

    private init(batch: IndexedNumericBatch, preprocessing: StandardPreprocessingPlan,
                 logicalPositions: [Int]?, retainedBytes: Int, preparationPeakBytes: Int) {
        self.batch = batch; self.preprocessing = preprocessing
        self.logicalPositions = logicalPositions; self.retainedBytes = retainedBytes
        self.preparationPeakBytes = preparationPeakBytes
    }

    package static func retainedEstimate(features: IndexedNumericBatch,
        preprocessing: StandardPreprocessingPlan) throws -> Int {
        try preprocessing.validate(features.source)
        return try preparedMLPInputBytes(features, preprocessing: preprocessing, mappingCapacity: 0)
    }

    package static func preparationEstimate(features: IndexedNumericBatch,
        preprocessing: StandardPreprocessingPlan) throws -> MemoryEstimate {
        let input = try retainedEstimate(features: features, preprocessing: preprocessing)
        // Includes input, ordered output, sort work, and capacity growth without partial grants.
        return try MemoryEstimate(capacities: [input, input,
            preparedMLPBytes(features.rowCount, MemoryLayout<Int>.stride, 8), 65_536])
    }

    package static func prepare(features: IndexedNumericBatch, preprocessing: StandardPreprocessingPlan,
                                budget: MemoryBudget) async throws -> PreparedMLPInput {
        let peak = try preparationEstimate(features: features, preprocessing: preprocessing)
        let reservation = try await budget.acquire(peak)
        do {
            let result = try prepareExternallyAdmitted(features: features,
                preprocessing: preprocessing, peak: peak)
            await reservation.finish()
            try Task.checkCancellation()
            return result
        } catch {
            await reservation.finish()
            throw error
        }
    }

    package static func prepareExternallyAdmitted(features: IndexedNumericBatch,
        preprocessing: StandardPreprocessingPlan, peak: MemoryEstimate) throws -> PreparedMLPInput {
        try Task.checkCancellation()
        let ordered = try features.sourceOrdered(prefixCount: mlpSourceOrderPrefix(rowCount: features.rowCount))
        let retained = try preparedMLPInputBytes(ordered.batch, preprocessing: preprocessing,
            mappingCapacity: ordered.logicalPositions?.capacity ?? 0)
        guard retained <= peak.bytes else { throw MemoryAdmissionError.invalidCapacity }
        try Task.checkCancellation()
        return PreparedMLPInput(batch: ordered.batch, preprocessing: preprocessing,
            logicalPositions: ordered.logicalPositions, retainedBytes: retained, preparationPeakBytes: peak.bytes)
    }

    func restoring(_ prediction: [Double]) -> [Double] {
        guard let positions = logicalPositions else { return prediction }
        var result = [Double](repeating: 0, count: prediction.count)
        for r in positions.indices { result[positions[r]] = prediction[r] }
        return result
    }
}
