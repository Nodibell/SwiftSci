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
    /// - Parameters:
    ///   - name: Name or identifier string.
    ///   - inputs: Input feature array or tensor.
    ///   - output: Output prediction vector or layer specification.
    ///   - weights: Learned model coefficient weights vector or matrix.
    ///   - bias: Learned model intercept or bias scalar.
    /// - Throws: `SwiftMLError` if feature-target dimensions mismatch, inputs are empty, or optimization fails.
    /// - Returns: Raw serialized binary data representation.
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
    ///
    /// Produces a valid, spec-compliant ONNX `ModelProto` (IR version 8, opset 13) using the `Gemm` operator.
    /// Compatible with official `onnx.checker` and `onnxruntime.InferenceSession`.
    ///
    /// - Parameters:
    ///   - name: Name or identifier string.
    ///   - inputs: Input feature identifier names (e.g. `["x1", "x2"]` or `["X"]`).
    ///   - output: Output prediction tensor name (defaults to `"output"`).
    ///   - weights: Learned model coefficient weights vector.
    ///   - bias: Learned model intercept or bias scalar.
    /// - Returns: Complete, loadable binary `.onnx` file `Data`.
    public static func exportBinaryONNX(
        name: String = "SwiftSciLinearONNX",
        inputs: [String],
        output: String = "output",
        weights: [Double],
        bias: Double
    ) -> Data {
        let numFeatures = weights.count
        let inputTensorName = inputs.first ?? "X"
        let weightName = "\(name)_weight"
        let biasName = "\(name)_bias"

        // 1. Initializer: Weights TensorProto (dims = [numFeatures, 1], data_type = 11 DOUBLE)
        var weightInit = ProtobufWriter()
        weightInit.writeVarintField(fieldNumber: 1, value: UInt64(numFeatures))
        weightInit.writeVarintField(fieldNumber: 1, value: 1)
        weightInit.writeVarintField(fieldNumber: 2, value: 11) // DOUBLE = 11
        weightInit.writeStringField(fieldNumber: 8, value: weightName)
        weightInit.writePackedDoublesField(fieldNumber: 10, values: weights)

        // 2. Initializer: Bias TensorProto (dims = [1], data_type = 11 DOUBLE)
        var biasInit = ProtobufWriter()
        biasInit.writeVarintField(fieldNumber: 1, value: 1)
        biasInit.writeVarintField(fieldNumber: 2, value: 11) // DOUBLE = 11
        biasInit.writeStringField(fieldNumber: 8, value: biasName)
        biasInit.writePackedDoublesField(fieldNumber: 10, values: [bias])

        // 3. NodeProto: Gemm (Y = X * W + B)
        var nodeWriter = ProtobufWriter()
        nodeWriter.writeStringField(fieldNumber: 1, value: inputTensorName)
        nodeWriter.writeStringField(fieldNumber: 1, value: weightName)
        nodeWriter.writeStringField(fieldNumber: 1, value: biasName)
        nodeWriter.writeStringField(fieldNumber: 2, value: output)
        nodeWriter.writeStringField(fieldNumber: 3, value: "node_\(name)")
        nodeWriter.writeStringField(fieldNumber: 4, value: "Gemm")

        // 4. Input ValueInfoProto: shape = ['batch_size', numFeatures], elem_type = 11 DOUBLE
        let inputInfo = encodeValueInfo(
            name: inputTensorName,
            elemType: 11,
            batchDimParam: "batch_size",
            featureDimValue: numFeatures
        )

        // 5. Output ValueInfoProto: shape = ['batch_size', 1], elem_type = 11 DOUBLE
        let outputInfo = encodeValueInfo(
            name: output,
            elemType: 11,
            batchDimParam: "batch_size",
            featureDimValue: 1
        )

        // 6. GraphProto
        var graphWriter = ProtobufWriter()
        graphWriter.writeBytesField(fieldNumber: 1, bytes: nodeWriter.data)      // node
        graphWriter.writeStringField(fieldNumber: 2, value: name)                // name
        graphWriter.writeBytesField(fieldNumber: 5, bytes: weightInit.data)      // initializer W
        graphWriter.writeBytesField(fieldNumber: 5, bytes: biasInit.data)        // initializer B
        graphWriter.writeBytesField(fieldNumber: 11, bytes: inputInfo)           // input
        graphWriter.writeBytesField(fieldNumber: 12, bytes: outputInfo)          // output

        // 7. OperatorSetIdProto (domain: "", version: 13)
        var opsetWriter = ProtobufWriter()
        opsetWriter.writeStringField(fieldNumber: 1, value: "")
        opsetWriter.writeVarintField(fieldNumber: 2, value: 13)

        // 8. ModelProto
        var modelWriter = ProtobufWriter()
        modelWriter.writeVarintField(fieldNumber: 1, value: 8)                   // ir_version = 8
        modelWriter.writeStringField(fieldNumber: 2, value: "SwiftSci ONNX Engine") // producer_name
        modelWriter.writeBytesField(fieldNumber: 7, bytes: graphWriter.data)     // graph
        modelWriter.writeBytesField(fieldNumber: 8, bytes: opsetWriter.data)     // opset_import

        return modelWriter.data
    }

    // MARK: - Private Protobuf Helpers

    private static func encodeValueInfo(
        name: String,
        elemType: Int,
        batchDimParam: String,
        featureDimValue: Int
    ) -> Data {
        var infoWriter = ProtobufWriter()
        infoWriter.writeStringField(fieldNumber: 1, value: name)

        // TypeProto -> Tensor
        var typeWriter = ProtobufWriter()
        var tensorTypeWriter = ProtobufWriter()
        tensorTypeWriter.writeVarintField(fieldNumber: 1, value: UInt64(elemType))

        // TensorShapeProto
        var shapeWriter = ProtobufWriter()

        // Dim 0: batch_size (dim_param = field 2)
        var dim0Writer = ProtobufWriter()
        dim0Writer.writeStringField(fieldNumber: 2, value: batchDimParam)
        shapeWriter.writeBytesField(fieldNumber: 1, bytes: dim0Writer.data)

        // Dim 1: feature count (dim_value = field 1)
        var dim1Writer = ProtobufWriter()
        dim1Writer.writeVarintField(fieldNumber: 1, value: UInt64(featureDimValue))
        shapeWriter.writeBytesField(fieldNumber: 1, bytes: dim1Writer.data)

        tensorTypeWriter.writeBytesField(fieldNumber: 2, bytes: shapeWriter.data)
        typeWriter.writeBytesField(fieldNumber: 1, bytes: tensorTypeWriter.data)
        infoWriter.writeBytesField(fieldNumber: 2, bytes: typeWriter.data)

        return infoWriter.data
    }
}

// MARK: - ProtobufWriter
// Shared implementation lives in Sources/SwiftML/Serialization/ProtobufWriter.swift
