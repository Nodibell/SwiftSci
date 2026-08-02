import Foundation

/// Exporter for serializing SwiftML models into ONNX model graph specification JSON format.
/// Note: Produces a JSON representation of the ONNX graph, not a binary .onnx file.
public enum ONNXExporter {

    /// Represents an ONNX computational graph JSON specification structure.
    public struct ONNXGraphSpec: Codable, Sendable {
        /// ONNX IR version number.
        public let irVersion: Int
        /// Framework producer identifier string.
        public let producerName: String
        /// Computational graph identifier name.
        public let graphName: String
        /// Array of input node tensor names.
        public let inputs: [String]
        /// Array of output node tensor names.
        public let outputs: [String]
        /// Operator node type identifier (e.g. "LinearRegressor").
        public let nodeType: String
        /// Linear feature weights tensor payload.
        public let weights: [Double]
        /// Bias term scalar payload.
        public let bias: Double
    }

    /// Serializes linear regression model into ONNX graph JSON bundle.
    public static func exportLinearONNX(
        name: String = "SwiftSciLinearONNX",
        inputs: [String],
        output: String = "output",
        weights: [Double],
        bias: Double
    ) throws -> Data {
        let spec = ONNXGraphSpec(
            irVersion: 8,
            producerName: "SwiftSci ONNX Engine",
            graphName: name,
            inputs: inputs,
            outputs: [output],
            nodeType: "LinearRegressor",
            weights: weights,
            bias: bias
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(spec)
    }
}
