import CoreML
import Foundation
import SwiftDataFrame
import SwiftPreprocessing

/// How prepared rows map to the model's declared input and output tensors.
public enum CoreMLInputLayout: Sendable {
    /// Each row is a separate example, using named scalars or a rank-one vector.
    case examples
    /// One request is a fixed rank-two matrix: rows first, features second.
    /// The model must declare matching input and output row counts.
    case matrix
}

/// Predictions and the logical bytes copied by the SwiftSci adapter.
/// Counts exclude Core ML's internal conversions, allocations, and device transfers.
public struct CoreMLPrediction: Sendable {
    /// Owned Double output columns, in input row order with original row indices preserved.
    public let values: PreparedNumericBatch
    /// Total bytes packed into numeric input arrays. Scalar inputs report zero.
    public let inputPackingBytes: Int
    /// Total Double bytes copied from Core ML outputs into owned result columns.
    public let outputCopyBytes: Int
}

/// Runs compiled Core ML regression, vector, or fixed matrix models on prepared columns.
///
/// The actor serializes synchronous predictions on its privately owned model. Each request
/// bounds vector storage by the requested maximum batch size or reserves one fixed matrix.
/// Missing and nonfinite inputs must be handled by the caller before prediction.
/// Core ML controls internal precision and device placement, including any CPU fallback.
public actor CoreMLPredictor {
    private let session: CoreMLPredictionSession
    private let asynchronousMatrix: Bool

    /// Loads a compiled `.mlmodelc` with an explicit ordered input-column contract.
    ///
    /// Scalar Double models match columns by name. A model with one fixed-size vector
    /// input receives values in `inputColumns` order. Vectors support Double and Float32, plus Float16 on macOS 15 or later.
    /// With `inputLayout: .matrix`, one fixed rank-two input receives all prepared rows
    /// in one call and requires a fixed rank-two output with the same row count.
    /// Example outputs must be a Double scalar or a supported floating-point vector. Integer class labels,
    /// images, sequences, and multiple input tensors need their own calling contract.
    ///
    /// - Parameters:
    ///   - compiledModelURL: Compiled artifact from `MLModel.compileModel(at:)`.
    ///   - inputColumns: Unique prepared-column names in model feature order.
    ///   - outputName: Numeric model output to extract.
    ///   - computeUnits: Devices Core ML may use, not a guarantee of actual placement.
    ///   - inputLayout: Explicit example or fixed matrix calling contract.
    /// - Throws: A model loading error or `SwiftMLError.invalidParameter` for an unsupported schema.
    public init(compiledModelURL: URL, inputColumns: [String], outputName: String,
                computeUnits: MLComputeUnits = .all, inputLayout: CoreMLInputLayout = .examples) throws {
        asynchronousMatrix = false
        session = try CoreMLPredictionSession(compiledModelURL: compiledModelURL, inputColumns: inputColumns,
                                               outputName: outputName, computeUnits: computeUnits, inputLayout: inputLayout)
    }

    // Trial-only selection; the public initializer retains the established synchronous path.
    package init(compiledModelURL: URL, inputColumns: [String], outputName: String,
                 computeUnits: MLComputeUnits, asynchronousMatrix: Bool) throws {
        self.asynchronousMatrix = asynchronousMatrix
        session = try CoreMLPredictionSession(compiledModelURL: compiledModelURL, inputColumns: inputColumns,
            outputName: outputName, computeUnits: computeUnits, inputLayout: .matrix)
    }

    /// Predicts after reserving input, output, bounded batch buffers, and estimated workspace.
    ///
    /// `workspaceBytes` must include Core ML temporary storage and adapter object overhead.
    /// Core ML does not expose a complete allocation bound; this reservation is not an RSS cap.
    /// Persistent model storage and returned results need separate lifetime accounting.
    /// Cancellation is checked while packing and between batches. Submitted work completes before release.
    ///
    /// - Parameters:
    ///   - input: Prepared columns matching the configured names, in any column order.
    ///   - budget: Shared admission budget for participating operations.
    ///   - workspaceBytes: Nonnegative additional peak estimate for this model and workload.
    ///   - maximumBatchSize: Positive limit on examples submitted per Core ML call. The default
    ///     preserves single-example calls. Larger values use Core ML batch prediction, which
    ///     does not guarantee vectorized execution or a particular device. Matrix input
    ///     requires this parameter to remain 1; its rows are submitted in one prediction.
    /// - Returns: Owned results and adapter copy counts. No partial output escapes on failure.
    /// - Throws: Admission, cancellation, schema, input-value, or Core ML prediction errors.
    public func predict(_ input: PreparedNumericBatch, budget: MemoryBudget,
                        workspaceBytes: Int, maximumBatchSize: Int = 1) async throws -> CoreMLPrediction {
        guard maximumBatchSize > 0 else {
            throw SwiftMLError.invalidParameter("Core ML maximum batch size must be positive")
        }
        try session.validateRequest(input, maximumBatchSize: maximumBatchSize)
        let indices = try session.columnIndices(in: input)
        let outputBytes = try coreMLByteCount(input.rowCount, session.outputColumns.count, MemoryLayout<Double>.stride)
        let inputRowBytes = try session.inputArray.map { try coreMLByteCount($0.width, $0.elementBytes) } ?? 0
        let outputRowBytes = try session.outputArray.map { try coreMLByteCount($0.width, $0.elementBytes) } ?? 0
        let packingBytes = try coreMLByteCount(input.rowCount, inputRowBytes)
        let batchRows = session.inputLayout == .matrix ? input.rowCount : min(maximumBatchSize, input.rowCount)
        let inputBatchBytes = try coreMLByteCount(batchRows, inputRowBytes)
        let outputBatchBytes = try coreMLByteCount(batchRows, outputRowBytes)
        let estimate = try MemoryEstimate(capacities: [input.payloadByteCount, outputBytes,
                                                       inputBatchBytes, outputBatchBytes, workspaceBytes])
        let reservation = try await budget.acquire(estimate)
        do {
            let result: CoreMLPrediction
            if asynchronousMatrix {
                result = try await session.runMatrixAsynchronously(input, indices: indices,
                    packingBytes: packingBytes, outputBytes: outputBytes, isolation: self)
            } else {
                // No suspension while the synchronous model or request buffers are in use.
                result = try session.run(input, indices: indices, packingBytes: packingBytes,
                    outputBytes: outputBytes, maximumBatchSize: maximumBatchSize)
            }
            await reservation.finish()
            return result
        } catch {
            await reservation.finish()
            throw error
        }
    }

}
