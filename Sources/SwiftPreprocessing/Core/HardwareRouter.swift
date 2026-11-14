#if os(macOS)
import Foundation
import MLX

/// Serializes device selection and MLX default-device changes (global MLX state).
public actor HardwareRouter {
    /// The shared.
    public static let shared = HardwareRouter()

    private init() {}

    /// Resolves `.auto` into a concrete device using algorithm-specific heuristics.
    ///
    /// Thresholds are starting points, not universal optima: they depend on the Apple Silicon
    /// configuration, matrix shape, dtype and GPU dispatch overhead. Current rules:
    /// KMeans `n*p < 500_000` → CPU; PCA `n < 2_000 && p < 500` → CPU; LinearSVC `n*p < 50_000` → CPU;
    /// Linear/LogisticRegression `n < 1_000` → CPU; MLP variants `n < 5_000` → CPU; tree models and others → CPU.
    /// Non-positive dimensions resolve to `.cpu`; overflowing `n*p` is treated as very large without trapping.
    /// - Parameters:
    ///   - algorithm: Optimization or decomposition algorithm choice.
    ///   - sampleCount: Number of sample observations.
    ///   - featureCount: Number of feature dimensions.
    ///   - requestedDevice: Target hardware execution device (.cpu, .gpu, or .auto).
    /// - Returns: Resolved execution hardware device target (`.cpu` or `.gpu`).
    public func resolveDevice(
        for algorithm: String,
        sampleCount: Int,
        featureCount: Int,
        requestedDevice: ExecutionDevice
    ) -> ExecutionDevice {
        guard requestedDevice == .auto else {
            // Direct MLX training uses GPU; deployed ANE execution is handled via CoreMLExporter.
            if requestedDevice == .ane { return .gpu }
            return requestedDevice
        }

        // Contract for `.auto`: non-positive dimensions describe no work and resolve to `.cpu`.
        // A product that overflows `Int` is far above every threshold and counts as "large",
        // so it saturates to `Int.max` instead of trapping.
        guard sampleCount > 0, featureCount > 0 else { return .cpu }
        let (product, overflow) = sampleCount.multipliedReportingOverflow(by: featureCount)
        let cells = overflow ? Int.max : product
        switch algorithm {
        case "KMeans":
            return cells < 500_000 ? .cpu : .gpu
        case "PCA":
            return (sampleCount < 2_000 && featureCount < 500) ? .cpu : .gpu
        case "LinearSVC":
            return cells < 50_000 ? .cpu : .gpu
        case "LinearRegression", "LogisticRegression":
            return sampleCount < 1_000 ? .cpu : .gpu
        case "MLP", "MLPClassifier", "MLPRegressor":
            return sampleCount < 5_000 ? .cpu : .gpu
        case "DBSCAN", "RandomForest", "RandomForestClassifier", "RandomForestRegressor", "GBDT", "GradientBoostedTreesRegressor", "DecisionTree", "DecisionTreeClassifier", "DecisionTreeRegressor", "IsolationForest", "KNNImputer", "TFIDF", "BPETokenizer":
            return .cpu
        default:
            return .cpu
        }
    }

}
#endif // os(macOS)
