import CoreML
import SwiftDataFrame
import SwiftPreprocessing

/// Immutable input shape and type captured from a matrix pool.
/// Holds no model or prediction slots and may outlive the pool that created it.
/// Prepare in caller tasks to allow preprocessing to run outside the pool actor.
public struct CoreMLMatrixInputPreparation: Sendable {
    let schema: ArrayFeature
    let names: [String]

    /// Required row count for this fixed-shape model.
    public var rowCount: Int { schema.shape[0] }
    /// Required feature order, also required by the fitted preprocessing plan.
    public var columnNames: [String] { names }

    /// Applies training-fitted imputation and scaling into private model-typed storage.
    /// Reserves output and temporary workspace together before allocation, then releases
    /// workspace capacity when preparation finishes. The returned owner retains its own quota.
    /// Keep the source and fitted plan separately accounted for. Acquisition may wait;
    /// provide headroom beyond any pool quota held on the same budget.
    /// Nonfinite or overflowing converted values throw. Float16 requires macOS 15.
    /// Prediction still copies prepared bytes into an exclusive pool slot.
    public func prepare(_ source: PreparedNumericBatch, preprocessing plan: StandardPreprocessingPlan,
                              budget: MemoryBudget) async throws -> CoreMLPreparedMatrix {
        guard source.columnNames == names,
              source.rowCount == schema.shape[0] else {
            throw SwiftMLError.invalidParameter("Preprocessing input differs from the model shape or column order")
        }
        try plan.validate(source)
        let allowance = try CoreMLPreparedMatrix.allowance(schema: schema, names: names)
        let workspace = try plan.workspaceAllowance(for: source)
        let peak = try MemoryEstimate(capacities: [allowance.bytes, workspace.bytes])
        let reservation = try await budget.acquire(peak)
        do {
            try Task.checkCancellation()
            let storage = try CoreMLMatrixStorage(schema)
            try fill(storage, source: source, plan: plan)
            try Task.checkCancellation()
            try await reservation.reduce(to: allowance)
            return try CoreMLPreparedMatrix(storage: storage, input: source, schema: schema, names: names,
                reservation: reservation, allowance: allowance)
        } catch {
            await reservation.finish()
            throw error
        }
    }

    private func fill(_ storage: CoreMLMatrixStorage, source: PreparedNumericBatch,
                      plan: StandardPreprocessingPlan) throws {
        switch schema.dataType {
        case .float16:
            guard #available(macOS 15, *) else {
                throw SwiftMLError.invalidParameter("Core ML Float16 arrays require macOS 15 or later")
            }
            try storage.array.withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, _ in
                try plan.fillFloat16(source, into: buffer)
                try requireFinite(buffer)
            }
        case .float32:
            try storage.array.withUnsafeMutableBufferPointer(ofType: Float.self) { buffer, _ in
                try plan.fillFloat32(source, into: buffer)
                try requireFinite(buffer)
            }
        case .double:
            try storage.array.withUnsafeMutableBufferPointer(ofType: Double.self) { buffer, _ in
                try plan.fillDouble(source, into: buffer)
                try requireFinite(buffer)
            }
        default:
            throw SwiftMLError.invalidParameter("Preprocessing requires a floating-point model input")
        }
    }

    private func requireFinite<T: BinaryFloatingPoint>(_ buffer: UnsafeMutableBufferPointer<T>) throws {
        guard buffer.allSatisfy(\.isFinite) else {
            throw SwiftMLError.invalidParameter("Preprocessed input overflows the model type or contains nonfinite values")
        }
    }
}
