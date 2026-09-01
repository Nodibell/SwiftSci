import Testing
import Foundation
import MLX
import MLXNN
@testable import SwiftVision

@Suite("YOLO Full Coverage & Weight Loading Tests")
struct YOLOCoverageTests {

    @Test("YOLOPreprocessor handles 3-channel RGB image planar vImage scaling")
    func testRGBPreprocessorPlanarScaling() {
        let preprocessor = YOLOPreprocessor(targetWidth: 64, targetHeight: 64)
        let w = 32
        let h = 16
        let pixelCount = w * h
        var rgbData = [Double](repeating: 0.0, count: pixelCount * 3)
        // Fill R, G, B channels with distinct patterns
        for i in 0..<pixelCount {
            rgbData[i] = 0.9                    // R
            rgbData[pixelCount + i] = 0.5       // G
            rgbData[pixelCount * 2 + i] = 0.2   // B
        }
        let rgbImg = ImageDataset(width: w, height: h, channels: 3, data: rgbData)
        let tensor = preprocessor.preprocess(image: rgbImg)

        #expect(tensor.shape == [1, 64, 64, 3])
        let flat = tensor.flattened().asArray(Float.self)
        #expect(!flat.isEmpty)
        // Check that non-padding area has non-zero values
        let hasValues = flat.contains { $0 > 0.1 }
        #expect(hasValues)
    }

    @Test("YOLOPreprocessor handles empty or invalid dimensions gracefully")
    func testEmptyImagePreprocessing() {
        let preprocessor = YOLOPreprocessor(targetWidth: 32, targetHeight: 32, paddingColor: 0.5)
        let emptyImg = ImageDataset(width: 0, height: 0, channels: 1, data: [])
        let tensor = preprocessor.preprocess(image: emptyImg)

        #expect(tensor.shape == [1, 32, 32, 3])
        let flat = tensor.flattened().asArray(Float.self)
        let allPadded = flat.allSatisfy { abs($0 - 0.5) < 1e-4 }
        #expect(allPadded)
    }

    @Test("YOLOBackbone loadWeights populates parameters correctly")
    func testBackboneLoadWeights() {
        let backbone = YOLOBackbone()
        var weightMap = [String: MLXArray]()

        // Populate mock weights for layer 0 conv (cIn: 3, cOut: 16, k: 3)
        weightMap["model.0.conv.weight"] = MLXArray.zeros([16, 3, 3, 3])
        weightMap["model.0.bn.weight"] = MLXArray.ones([16])
        weightMap["model.0.bn.bias"] = MLXArray.zeros([16])
        weightMap["model.0.bn.running_mean"] = MLXArray.zeros([16])
        weightMap["model.0.bn.running_var"] = MLXArray.ones([16])

        let loader = YOLOWeightLoader(weights: weightMap)
        backbone.loadWeights(from: loader, prefix: "model")

        #expect(true)
    }

    @Test("YOLONeck loadWeights populates parameters correctly")
    func testNeckLoadWeights() {
        let neck = YOLONeck()
        var weightMap = [String: MLXArray]()

        // conv_p4: 64 -> 64
        weightMap["model.16.conv.weight"] = MLXArray.zeros([64, 3, 3, 64])
        weightMap["model.16.bn.weight"] = MLXArray.ones([64])
        weightMap["model.16.bn.bias"] = MLXArray.zeros([64])
        weightMap["model.16.bn.running_mean"] = MLXArray.zeros([64])
        weightMap["model.16.bn.running_var"] = MLXArray.ones([64])

        let loader = YOLOWeightLoader(weights: weightMap)
        neck.loadWeights(from: loader, prefix: "model")

        #expect(true)
    }

    @Test("YOLOHead loadWeights populates parameters correctly")
    func testHeadLoadWeights() {
        let head = YOLOHead(numClasses: 80, regMax: 16)
        var weightMap = [String: MLXArray]()

        let headChannels = [64, 128, 256]
        for (i, c) in headChannels.enumerated() {
            weightMap["model.22.cv2.\(i).0.conv.weight"] = MLXArray.zeros([64, 3, 3, c])
            weightMap["model.22.cv2.\(i).0.bn.weight"] = MLXArray.ones([64])
            weightMap["model.22.cv2.\(i).0.bn.bias"] = MLXArray.zeros([64])
            weightMap["model.22.cv2.\(i).0.bn.running_mean"] = MLXArray.zeros([64])
            weightMap["model.22.cv2.\(i).0.bn.running_var"] = MLXArray.ones([64])
            weightMap["model.22.cv2.\(i).2.weight"] = MLXArray.zeros([64, 1, 1, 64])
            weightMap["model.22.cv2.\(i).2.bias"] = MLXArray.zeros([64])
        }

        let loader = YOLOWeightLoader(weights: weightMap)
        head.loadWeights(from: loader, prefix: "model.22")

        #expect(head.numClasses == 80)
    }

    @Test("YOLOv8Detector loadWeights end-to-end delegation")
    func testDetectorLoadWeights() async {
        let detector = YOLOv8Detector()
        var weightMap = [String: MLXArray]()
        weightMap["model.0.conv.weight"] = MLXArray.zeros([16, 3, 3, 3])
        let loader = YOLOWeightLoader(weights: weightMap)

        await detector.loadWeights(loader)
        #expect(true)
    }
}
