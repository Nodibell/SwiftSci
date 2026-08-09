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

    /// Serializes linear regression model into binary ONNX Protobuf payload (.onnx format).
    public static func exportBinaryONNX(
        name: String = "SwiftSciLinearONNX",
        inputs: [String],
        output: String = "output",
        weights: [Double],
        bias: Double
    ) -> Data {
        var writer = ProtobufWriter()
        
        // Field 1: ir_version = 8 (varint)
        writer.writeVarintField(fieldNumber: 1, value: 8)
        // Field 2: producer_name = "SwiftSci ONNX Engine" (string)
        writer.writeStringField(fieldNumber: 2, value: "SwiftSci ONNX Engine")
        
        // Field 7: GraphProto
        var graphWriter = ProtobufWriter()
        graphWriter.writeStringField(fieldNumber: 2, value: name) // graph name
        
        // Graph inputs
        for inputName in inputs {
            var valueInfoWriter = ProtobufWriter()
            valueInfoWriter.writeStringField(fieldNumber: 1, value: inputName)
            graphWriter.writeBytesField(fieldNumber: 11, bytes: valueInfoWriter.data)
        }
        
        // Graph outputs
        var outValueInfoWriter = ProtobufWriter()
        outValueInfoWriter.writeStringField(fieldNumber: 1, value: output)
        graphWriter.writeBytesField(fieldNumber: 12, bytes: outValueInfoWriter.data)
        
        // Node: Gemm / Linear
        var nodeWriter = ProtobufWriter()
        let weightName = "\(name)_weight"
        let biasName = "\(name)_bias"
        nodeWriter.writeStringField(fieldNumber: 1, value: inputs.first ?? "X")
        nodeWriter.writeStringField(fieldNumber: 1, value: weightName)
        nodeWriter.writeStringField(fieldNumber: 1, value: biasName)
        nodeWriter.writeStringField(fieldNumber: 2, value: output)
        nodeWriter.writeStringField(fieldNumber: 3, value: "node_\(name)")
        nodeWriter.writeStringField(fieldNumber: 4, value: "Gemm")
        graphWriter.writeBytesField(fieldNumber: 1, bytes: nodeWriter.data)
        
        // Initializer 1: Weights TensorProto
        var weightInitWriter = ProtobufWriter()
        weightInitWriter.writeStringField(fieldNumber: 1, value: weightName)
        weightInitWriter.writeVarintField(fieldNumber: 2, value: 11) // DOUBLE = 11
        for w in weights {
            weightInitWriter.writeDoubleField(fieldNumber: 7, value: w)
        }
        graphWriter.writeBytesField(fieldNumber: 5, bytes: weightInitWriter.data)
        
        // Initializer 2: Bias TensorProto
        var biasInitWriter = ProtobufWriter()
        biasInitWriter.writeStringField(fieldNumber: 1, value: biasName)
        biasInitWriter.writeVarintField(fieldNumber: 2, value: 11) // DOUBLE = 11
        biasInitWriter.writeDoubleField(fieldNumber: 7, value: bias)
        graphWriter.writeBytesField(fieldNumber: 5, bytes: biasInitWriter.data)
        
        writer.writeBytesField(fieldNumber: 7, bytes: graphWriter.data)
        return writer.data
    }
}

// MARK: - Lightweight Binary Protobuf Serialization Utility
private struct ProtobufWriter {
    private(set) var data = Data()
    
    mutating func writeVarint(_ value: UInt64) {
        var v = value
        while v >= 0x80 {
            data.append(UInt8((v & 0x7F) | 0x80))
            v >>= 7
        }
        data.append(UInt8(v & 0x7F))
    }
    
    mutating func writeTag(fieldNumber: Int, wireType: Int) {
        writeVarint(UInt64((fieldNumber << 3) | wireType))
    }
    
    mutating func writeVarintField(fieldNumber: Int, value: UInt64) {
        writeTag(fieldNumber: fieldNumber, wireType: 0)
        writeVarint(value)
    }
    
    mutating func writeDoubleField(fieldNumber: Int, value: Double) {
        writeTag(fieldNumber: fieldNumber, wireType: 1)
        var bitPattern = value.bitPattern
        withUnsafeBytes(of: &bitPattern) { data.append(contentsOf: $0) }
    }
    
    mutating func writeStringField(fieldNumber: Int, value: String) {
        let utf8 = Data(value.utf8)
        writeTag(fieldNumber: fieldNumber, wireType: 2)
        writeVarint(UInt64(utf8.count))
        data.append(utf8)
    }
    
    mutating func writeBytesField(fieldNumber: Int, bytes: Data) {
        writeTag(fieldNumber: fieldNumber, wireType: 2)
        writeVarint(UInt64(bytes.count))
        data.append(bytes)
    }
}
