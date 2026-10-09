import CoreML
import SwiftDataFrame
import SwiftPreprocessing

// Immutable metadata only. It cannot retain a model or its prediction slots.
package struct CoreMLTrialInputContract: Sendable {
    let schema: ArrayFeature
    let names: [String]

    // Called from caller tasks, outside the pool actor. No pointer escapes this operation.
    package func prepareFused(_ source: PreparedNumericBatch, plan: TrialFusedPreprocessingPlan,
                              budget: MemoryBudget) async throws -> CoreMLPreparedMatrix {
        guard #available(macOS 15, *) else {
            throw SwiftMLError.invalidParameter("Core ML Float16 arrays require macOS 15 or later")
        }
        guard schema.dataType == .float16, source.columnNames == names,
              source.rowCount == schema.shape[0] else {
            throw SwiftMLError.invalidParameter("Direct trial input differs from the Float16 model contract")
        }
        try plan.validate(source)
        let allowance = try CoreMLPreparedMatrix.allowance(schema: schema, names: names)
        let reservation = try await budget.acquire(allowance)
        do {
            try Task.checkCancellation()
            let storage = try CoreMLMatrixStorage(schema)
            try storage.array.withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, _ in
                try plan.fillFloat16(source, into: buffer)
                guard buffer.allSatisfy(\.isFinite) else {
                    throw SwiftMLError.invalidParameter("Fused input overflows Float16 or contains nonfinite values")
                }
            }
            return try CoreMLPreparedMatrix(storage: storage, input: source, schema: schema, names: names,
                reservation: reservation, allowance: allowance)
        } catch {
            await reservation.finish()
            throw error
        }
    }
}
