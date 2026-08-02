import Foundation

/// Exporter for serializing SwiftML linear, tree, and ensemble models into CoreML JSON graph specifications.
/// Note: Produces a JSON representation of the model specification, not a binary .mlmodel or .mlpackage bundle.
public enum CoreMLExporter {
    
    /// Represents a CoreML model specification metadata and parameter payload.
    public struct CoreMLModelSpec: Codable, Sendable {
        /// Display name of the CoreML model.
        public let modelName: String
        /// Author metadata string.
        public let author: String
        /// License metadata string.
        public let license: String
        /// Array of input feature identifier names.
        public let inputFeatures: [String]
        /// Output target feature identifier name.
        public let outputFeature: String
        /// Feature weight coefficient array.
        public let weights: [Double]
        /// Intercept bias term.
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
