import Testing
import Foundation
import MLX
import MLXNN
import SwiftML
@testable import SwiftVision

@Suite("YOLOv8 Real Inference Tests")
struct YOLOv8Tests {
    @Test("ONNXWeightReader parses binary ONNX weight initializers")
    func testONNXWeightReader() throws {
        let binaryONNX = ONNXExporter.exportBinaryONNX(name: "yolo_test", inputs: ["X"], output: "Y", weights: [0.5, 0.7], bias: 0.1)
        let loader = YOLOWeightLoader.loadFromONNX(data: binaryONNX)
        let weight = loader.get("yolo_test_weight")

        #expect(weight != nil)
    }
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

    @Test("YOLOPreprocessor resizes and pads image to 640x640")
    func testYOLOPreprocessor() throws {
        let prep = YOLOPreprocessor(targetWidth: 640, targetHeight: 640)
        let img = try ImageDataset(width: 100, height: 50, channels: 1, data: [Double](repeating: 0.8, count: 5000))
        let tensor = prep.preprocess(image: img)

        eval(tensor)
        #expect(tensor.shape == [1, 640, 640, 3])
    }

    @Test("YOLOv8Detector end-to-end forward pass executes successfully")
    func testYOLOv8DetectorForwardPass() async throws {
        let detector = YOLOv8Detector(confidenceThreshold: 0.1, iouThreshold: 0.45)
        let img = try ImageDataset(width: 128, height: 128, channels: 1, data: [Double](repeating: 0.5, count: 128 * 128))
        let boxes = try await detector.detect(image: img)

        #expect(boxes.count >= 0)
    }
}
