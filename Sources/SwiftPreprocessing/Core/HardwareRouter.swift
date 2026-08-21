#if os(macOS)
import Foundation
import MLX

/// Serializes device selection and MLX default-device changes (global MLX state).
public actor HardwareRouter {
    /// The shared.
    public static let shared = HardwareRouter()

    private init() {}

    /// Resolves `.auto` into a concrete device using plan-0.8 heuristics.
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

        let cells = sampleCount * featureCount
        switch algorithm {
        case "KMeans":
            return cells < 500_000 ? .cpu : .gpu
        case "PCA":
            return (sampleCount < 2_000 && featureCount < 500) ? .cpu : .gpu
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
