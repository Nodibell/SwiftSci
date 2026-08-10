import Testing
import Foundation
import MLX
import MLXNN
import SwiftML
@testable import SwiftVision

@Suite("YOLOv8 Real Inference Tests")
struct YOLOv8Tests {

    // MARK: - ONNXWeightReader / YOLOWeightLoader (existing)

    @Test("ONNXWeightReader parses binary ONNX weight initializers")
    func testONNXWeightReader() throws {
        let binaryONNX = ONNXExporter.exportBinaryONNX(name: "yolo_test", inputs: ["X"], output: "Y", weights: [0.5, 0.7], bias: 0.1)
        let loader = YOLOWeightLoader.loadFromONNX(data: binaryONNX)
        let weight = loader.get("yolo_test_weight")
        #expect(weight != nil)
    }

    // MARK: - Backbone / Neck / Head (existing)

    @Test("YOLOBackbone outputs expected multi-scale P3, P4, P5 shapes")
    func testYOLOBackbone() throws {
        let backbone = YOLOBackbone()
        let dummyInput = MLXArray.zeros([1, 640, 640, 3])
        let output = backbone(dummyInput)
        eval(output.p3, output.p4, output.p5)
        #expect(output.p3.shape == [1, 80, 80, 64])
        #expect(output.p4.shape == [1, 40, 40, 128])
        #expect(output.p5.shape == [1, 20, 20, 256])
    }

    @Test("YOLONeck outputs expected feature pyramid shapes")
    func testYOLONeck() throws {
        let backbone = YOLOBackbone()
        let neck = YOLONeck()
        let dummyInput = MLXArray.zeros([1, 640, 640, 3])
        let bbOut = backbone(dummyInput)
        let neckOut = neck(bbOut)
        eval(neckOut.headP3, neckOut.headP4, neckOut.headP5)
        #expect(neckOut.headP3.shape == [1, 80, 80, 64])
        #expect(neckOut.headP4.shape == [1, 40, 40, 128])
        #expect(neckOut.headP5.shape == [1, 20, 20, 256])
    }

    @Test("YOLOHead decodes 8400 anchor predictions and class scores")
    func testYOLOHead() throws {
        let backbone = YOLOBackbone()
        let neck = YOLONeck()
        let head = YOLOHead(numClasses: 80)
        let dummyInput = MLXArray.zeros([1, 640, 640, 3])
        let bbOut = backbone(dummyInput)
        let neckOut = neck(bbOut)
        let headOut = head(neckOut)
        eval(headOut.boxes, headOut.scores)
        #expect(headOut.boxes.shape == [1, 8400, 4])
        #expect(headOut.scores.shape == [1, 8400, 80])
    }

    // MARK: - YOLOPreprocessor (existing + new edge cases)

    @Test("YOLOPreprocessor resizes and pads landscape image to 640x640")
    func testYOLOPreprocessorLandscape() throws {
        let prep = YOLOPreprocessor(targetWidth: 640, targetHeight: 640)
        let img = try ImageDataset(width: 100, height: 50, channels: 1, data: [Double](repeating: 0.8, count: 5000))
        let tensor = prep.preprocess(image: img)
        eval(tensor)
        #expect(tensor.shape == [1, 640, 640, 3])
    }

    @Test("YOLOPreprocessor handles square input without distortion")
    func testYOLOPreprocessorSquare() throws {
        let prep = YOLOPreprocessor(targetWidth: 320, targetHeight: 320)
        let img = try ImageDataset(width: 320, height: 320, channels: 1, data: [Double](repeating: 0.5, count: 320 * 320))
        let tensor = prep.preprocess(image: img)
        eval(tensor)
        #expect(tensor.shape == [1, 320, 320, 3])
    }

    @Test("YOLOPreprocessor handles portrait image with letterbox padding")
    func testYOLOPreprocessorPortrait() throws {
        let prep = YOLOPreprocessor(targetWidth: 640, targetHeight: 640)
        let img = try ImageDataset(width: 100, height: 300, channels: 1, data: [Double](repeating: 0.3, count: 30000))
        let tensor = prep.preprocess(image: img)
        eval(tensor)
        #expect(tensor.shape == [1, 640, 640, 3])
    }

    @Test("YOLOPreprocessor respects custom paddingColor value zero")
    func testYOLOPreprocessorCustomPadding() throws {
        let prep = YOLOPreprocessor(targetWidth: 64, targetHeight: 64, paddingColor: 0.0)
        let img = try ImageDataset(width: 10, height: 10, channels: 1, data: [Double](repeating: 1.0, count: 100))
        let tensor = prep.preprocess(image: img)
        eval(tensor)
        #expect(tensor.shape == [1, 64, 64, 3])
    }

    @Test("YOLOPreprocessor handles 1280x720 HD landscape")
    func testYOLOPreprocessorHD() throws {
        let prep = YOLOPreprocessor()
        let img = try ImageDataset(width: 1280, height: 720, channels: 1, data: [Double](repeating: 0.6, count: 1280 * 720))
        let tensor = prep.preprocess(image: img)
        eval(tensor)
        #expect(tensor.shape == [1, 640, 640, 3])
    }

    // MARK: - YOLOv8Detector (existing)

    @Test("YOLOv8Detector end-to-end forward pass executes successfully")
    func testYOLOv8DetectorForwardPass() async throws {
        let detector = YOLOv8Detector(confidenceThreshold: 0.1, iouThreshold: 0.45)
        let img = try ImageDataset(width: 128, height: 128, channels: 1, data: [Double](repeating: 0.5, count: 128 * 128))
        let boxes = try await detector.detect(image: img)
        #expect(boxes.count >= 0)
    }

    // MARK: - ONNXWeightReader edge cases

    @Test("ONNXWeightReader returns empty dict for empty data")
    func testONNXWeightReaderEmpty() {
        let result = ONNXWeightReader.parse(data: Data())
        #expect(result.isEmpty)
    }

    @Test("ONNXWeightReader does not crash on random garbage bytes")
    func testONNXWeightReaderGarbage() {
        let garbage = Data([0xFF, 0xFE, 0x00, 0x01, 0x80, 0x7F, 0xAA, 0xBB])
        let result = ONNXWeightReader.parse(data: garbage)
        #expect(result.count >= 0) // Must not crash; empty is fine
    }

    @Test("ONNXWeightReader weight+bias tensors are correctly keyed")
    func testONNXWeightReaderPackedFloatPath() {
        let data = ONNXExporter.exportBinaryONNX(
            name: "packed_test",
            inputs: ["in"],
            output: "out",
            weights: [1.0, 2.0, 3.0, 4.0],
            bias: 0.5
        )
        let loader = YOLOWeightLoader.loadFromONNX(data: data)
        #expect(loader.get("packed_test_weight") != nil)
        #expect(loader.get("packed_test_bias") != nil)
    }

    @Test("ONNXWeightReader ignores initializer without name field")
    func testONNXWeightReaderNoName() {
        // Manually encode a minimal GraphProto (field 7, len-prefixed) containing
        // an initializer TensorProto (field 5) that has data_type=1 but NO name field.
        // parseTensor should return nil → result dict stays empty.
        var tensorProto = Data()
        // field 3 (data_type=FLOAT=1), wire 0: tag = (3<<3)|0 = 24, value = 1
        tensorProto.append(contentsOf: [24, 1])

        var graphProto = Data()
        // field 5, wire 2: tag = (5<<3)|2 = 42
        graphProto.append(42)
        graphProto.append(UInt8(tensorProto.count))
        graphProto.append(contentsOf: tensorProto)

        var modelProto = Data()
        // field 7, wire 2: tag = (7<<3)|2 = 58
        modelProto.append(58)
        modelProto.append(UInt8(graphProto.count))
        modelProto.append(contentsOf: graphProto)

        let result = ONNXWeightReader.parse(data: modelProto)
        #expect(result.isEmpty)
    }

    // MARK: - YOLOWeightLoader.normalizeKey

    @Test("normalizeKey strips model.model. double prefix")
    func testNormalizeKeyDoubleModel() {
        #expect(YOLOWeightLoader.normalizeKey("model.model.0.conv.weight") == "0.conv.weight")
    }

    @Test("normalizeKey strips single model. prefix")
    func testNormalizeKeySingleModel() {
        #expect(YOLOWeightLoader.normalizeKey("model.0.conv.bias") == "0.conv.bias")
    }

    @Test("normalizeKey leaves key without known prefix unchanged")
    func testNormalizeKeyNoPrefix() {
        let key = "backbone.layer1.weight"
        #expect(YOLOWeightLoader.normalizeKey(key) == key)
    }

    @Test("YOLOWeightLoader.get falls back to normalizeKey lookup")
    func testWeightLoaderFallbackNormalize() {
        let weights: [String: MLXArray] = ["0.conv.weight": MLXArray([1.0, 2.0])]
        let loader = YOLOWeightLoader(weights: weights)
        // Query with model. prefix → normalizeKey strips it → found
        #expect(loader.get("model.0.conv.weight") != nil)
        // Non-existent key → nil
        #expect(loader.get("nonexistent.key") == nil)
    }
}
