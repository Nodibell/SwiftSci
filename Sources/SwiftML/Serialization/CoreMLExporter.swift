import Foundation

/// Exporter for serializing SwiftML linear regression weights into a CoreML JSON specification payload.
/// Note: Produces a JSON representation of model metadata and coefficients, not a binary .mlmodel or
/// .mlpackage bundle. Tree, forest, logistic, and binary Core ML artifact export are not yet implemented.
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
