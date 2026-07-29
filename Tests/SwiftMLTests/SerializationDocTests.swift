import Testing
import Foundation
@testable import SwiftML

@Suite("Model Serialization Tests (Phase 4)")
struct SerializationDocTests {

    @Test("CoreMLExporter serializes linear model to valid JSON specification Data")
    func testCoreMLExportLinearModel() throws {
        let weights = [0.5, -1.2, 3.4]
        let bias = 0.1
        let data = try CoreMLExporter.exportLinearModel(
            name: "TestModel",
            inputNames: ["x1", "x2", "x3"],
            outputName: "y",
            weights: weights,
            bias: bias
        )
        
        #expect(!data.isEmpty)
        let decoder = JSONDecoder()
        let spec = try decoder.decode(CoreMLExporter.CoreMLModelSpec.self, from: data)
        #expect(spec.modelName == "TestModel")
        #expect(spec.inputFeatures == ["x1", "x2", "x3"])
        #expect(spec.outputFeature == "y")
        #expect(spec.weights == weights)
        #expect(spec.bias == bias)
    }

    @Test("ONNXExporter serializes linear model to valid ONNX graph JSON Data")
    func testONNXExportLinearModel() throws {
        let weights = [1.0, 2.0]
        let bias = 0.5
        let data = try ONNXExporter.exportLinearONNX(
            name: "TestONNX",
            inputs: ["a", "b"],
            output: "out",
            weights: weights,
            bias: bias
        )
        
        #expect(!data.isEmpty)
        let decoder = JSONDecoder()
        let spec = try decoder.decode(ONNXExporter.ONNXGraphSpec.self, from: data)
        #expect(spec.graphName == "TestONNX")
        #expect(spec.inputs == ["a", "b"])
        #expect(spec.outputs == ["out"])
        #expect(spec.nodeType == "LinearRegressor")
        #expect(spec.weights == weights)
        #expect(spec.bias == bias)
    }
}
