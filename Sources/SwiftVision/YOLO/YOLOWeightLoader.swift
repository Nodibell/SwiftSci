import Foundation
import MLX
import MLXNN

/// Model weight loader for YOLOv8 neural network inference.
/// Parses and maps dictionary tensors (`[String: MLXArray]`) exported from Safetensors or ONNX models.
public struct YOLOWeightLoader {
    /// Raw tensor weights dictionary keyed by layer name.
    public let weights: [String: MLXArray]
    
    /// Creates a new weight loader instance.
    /// - Parameter weights: Dictionary of tensor weights keyed by parameter path.
    public init(weights: [String: MLXArray]) {
        self.weights = weights
    }

    /// Loads model weights from binary ONNX Protobuf data payload.
    /// - Parameter data: Raw `.onnx` file binary data.
    /// - Returns: `YOLOWeightLoader` instance with parsed tensor dictionary.
    public static func loadFromONNX(data: Data) -> YOLOWeightLoader {
        let parsed = ONNXWeightReader.parse(data: data)
        return YOLOWeightLoader(weights: parsed)
    }

    /// Retrieves a weight array for a given key path.
    /// - Parameter key: Parameter path (e.g. `model.0.conv.weight`).
    /// - Returns: The `MLXArray` tensor if present.
    public func get(_ key: String) -> MLXArray? {
        return weights[key] ?? weights[YOLOWeightLoader.normalizeKey(key)]
    }

    /// Normalizes Ultralytics PyTorch weight names into unified SwiftVision layer names.
    /// - Parameter key: PyTorch parameter key string.
    /// - Returns: Standardized layer path string.
    public static func normalizeKey(_ key: String) -> String {
        var k = key
        if k.hasPrefix("model.model.") {
            k = String(k.dropFirst(12))
        } else if k.hasPrefix("model.") {
            k = String(k.dropFirst(6))
        }
        return k
    }
}
