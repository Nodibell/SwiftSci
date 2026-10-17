import SwiftDataFrame
import SwiftPreprocessing

/// Immutable training-fitted imputation, scaling, and CPU Double regression weights.
/// Create through MLPRegressor.fitStandardized. Concurrent predictions share only
/// immutable storage; each admitted call owns its workspace. Caller code owns caching.
public final class FittedMLPPredictor: Sendable {
    let preprocessing: StandardPreprocessingPlan
    let weights: MLPCPUPredictor
    let context = MLPPredictionContext()
    /// Feature names in training order. Incoming schema must match even for zero rows.
    public var columnNames: [String] { preprocessing.columnNames }
    /// Conservative weights and preprocessing capacity charge for idle retention policy.
    public let retainedBytes: Int

    init(preprocessing: StandardPreprocessingPlan, weights: MLPCPUPredictor) throws {
        self.preprocessing = preprocessing; self.weights = weights
        retainedBytes = try MemoryEstimate(capacities: [weights.retainedBytes,
            preparedMLPPlanBytes(preprocessing), MLPPredictionOverhead.predictor]).bytes
    }

    /// Estimates retained request data, deferred selection and edit buffers, ordering,
    /// finite-value validation, and preparation workspace. It does not allocate those buffers.
    public func preparationEstimate(for request: MLPPredictionRequest) throws -> MemoryEstimate {
        try request.preparationEstimate(preprocessing: preprocessing)
    }

    /// Materializes and validates input under one complete reservation, then releases it.
    /// Dropping the last input reference releases idle storage through Swift ownership.
    public func prepare(_ request: MLPPredictionRequest, budget: MemoryBudget) async throws -> MLPPredictionInput {
        let peak = try preparationEstimate(for: request)
        let lease = try await budget.acquire(peak)
        do {
            let view = try request.materialized()
            try validateStandardizedMLPInput(view, preprocessing: preprocessing)
            let core = try PreparedMLPInput.prepareExternallyAdmitted(features: view,
                preprocessing: preprocessing, peak: peak)
            let result = try MLPPredictionInput(core: core, context: context)
            guard result.retainedBytes <= peak.bytes else { throw MemoryAdmissionError.invalidCapacity }
            await lease.finish()
            try Task.checkCancellation()
            return result
        } catch { await lease.finish(); throw error }
    }

    /// Estimates one prediction, including retained input, captured weights, scratch, and
    /// replacing outputs. Accumulated outputs and unrelated owners need separate allowance.
    /// An input from another fitted context throws before any budget acquisition.
    public func predictionEstimate(for input: MLPPredictionInput) throws -> MemoryEstimate {
        guard input.context === context else { throw SwiftMLError.invalidInput("Input belongs to another fitted predictor") }
        let core = try weights.predictionEstimate(for: input.core)
        return try MemoryEstimate(capacities: [core.bytes, MLPPredictionOverhead.input, MLPPredictionOverhead.predictor])
    }

    /// Predicts in original logical occurrence order. NaN or infinite results throw.
    /// Admission and cancellation release reservations on every exit. Individual BLAS
    /// calls are synchronous and cannot be interrupted. Later actor fits have no effect.
    public func predict(_ input: MLPPredictionInput, budget: MemoryBudget) async throws -> [Double] {
        let estimate = try predictionEstimate(for: input)
        let lease = try await budget.acquire(estimate)
        do {
            let result = try weights.predictExternallyAdmitted(features: input.core)
            guard result.allSatisfy(\.isFinite) else { throw SwiftMLError.invalidInput("Prediction produced nonfinite values") }
            await lease.finish()
            try Task.checkCancellation()
            return result
        } catch { await lease.finish(); throw error }
    }
}
