import Foundation

/// Exporter for serializing SwiftML models into ONNX model graph specification JSON format.
/// Note: Produces a JSON representation of the ONNX graph, not a binary .onnx file.
public enum ONNXExporter {

    public struct ONNXGraphSpec: Codable, Sendable {
        public let irVersion: Int
        public let producerName: String
        public let graphName: String
        public let inputs: [String]
        public let outputs: [String]
        public let nodeType: String
        public let weights: [Double]
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
