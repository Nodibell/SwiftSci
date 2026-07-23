import Foundation
import MLX

/// Serializes device selection and MLX default-device changes (global MLX state).
public actor HardwareRouter {
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
            // ANE is a placeholder until CoreML export exists.
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
        case "DBSCAN":
            return .cpu
        default:
            return .cpu
        }
    }

}
