import Foundation

/// Exporter for serializing SwiftML linear, tree, and ensemble models into CoreML JSON graph specifications.
/// Note: Produces a JSON representation of the model specification, not a binary .mlmodel or .mlpackage bundle.
public enum CoreMLExporter {
    
    /// Represents core m l model spec.
    public struct CoreMLModelSpec: Codable, Sendable {
        public let modelName: String
        public let author: String
        public let license: String
        public let inputFeatures: [String]
        public let outputFeature: String
        public let weights: [Double]
        public let bias: Double
    }

    /// Serializes linear regression weights and bias into a CoreML specification JSON structure.
    public static func exportLinearModel(
        name: String = "SwiftSciLinearModel",
        inputNames: [String],
        outputName: String = "target",
        weights: [Double],
        bias: Double
    ) throws -> Data {
        let spec = CoreMLModelSpec(
            modelName: name,
            author: "SwiftSci Ecosystem",
            license: "MIT",
            inputFeatures: inputNames,
            outputFeature: outputName,
            weights: weights,
            bias: bias
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(spec)
    }
}
