import SwiftDataFrame

final class MLPPredictionContext: Sendable {}

// Conservative object and collection metadata allowances, not allocator measurements.
enum MLPPredictionOverhead {
    static let input = 128
    static let predictor = 256
    static let editedColumn = 128
}

/// Immutable prepared data bound to the fitted predictor that created it.
/// Idle input holds no reservation. It retains preprocessing and source storage,
/// but does not retain the fitted predictor, weights, or mutable training actor.
public final class MLPPredictionInput: Sendable {
    let core: PreparedMLPInput
    let context: MLPPredictionContext
    /// Number of logical input occurrences, including repeats.
    public var rowCount: Int { core.rowCount }
    /// Conservative retained-capacity charge. Shared storage may be charged repeatedly.
    public let retainedBytes: Int
    /// Complete preparation allowance used before materialization.
    public var preparationPeakBytes: Int { core.preparationPeakBytes }

    init(core: PreparedMLPInput, context: MLPPredictionContext) throws {
        self.core = core; self.context = context
        retainedBytes = try MemoryEstimate(capacities: [core.retainedBytes, MLPPredictionOverhead.input]).bytes
    }
}
