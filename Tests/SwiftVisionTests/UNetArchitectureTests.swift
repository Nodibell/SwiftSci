import Testing
import Foundation
import MLX
@testable import SwiftVision

@Suite("UNet Architecture & CNN Feature Extractor Tests")
struct UNetArchitectureTests {

    @Test("UNetDoubleConv forward pass preserves spatial shape and maps channels")
    func testUNetDoubleConv() {
        let block = UNetDoubleConv(inChannels: 3, outChannels: 16)
        let x = MLXArray.zeros([1, 8, 8, 3])
        let out = block(x)
        #expect(out.shape == [1, 8, 8, 16])
    }

    @Test("UNetDown forward pass halves spatial dimensions")
    func testUNetDown() {
        let down = UNetDown(inChannels: 16, outChannels: 32)
        let x = MLXArray.zeros([1, 8, 8, 16])
        let out = down(x)
        #expect(out.shape == [1, 4, 4, 32])
    }

    @Test("UNetUp forward pass doubles spatial dimensions and concatenates skip features")
    func testUNetUp() {
        let up = UNetUp(inChannels: 32, outChannels: 16)
        let x1 = MLXArray.zeros([1, 4, 4, 32])
        let x2 = MLXArray.zeros([1, 8, 8, 16])
        let out = up(x1: x1, x2: x2)
        #expect(out.shape == [1, 8, 8, 16])
    }

    @Test("UNetOutConv projects features to class logits")
    func testUNetOutConv() {
        let outConv = UNetOutConv(inChannels: 16, outChannels: 2)
        let x = MLXArray.zeros([1, 8, 8, 16])
        let out = outConv(x)
        #expect(out.shape == [1, 8, 8, 2])
    }

    @Test("UNetArchitecture end-to-end forward pass produces correct segmentation logits shape")
    func testUNetArchitectureForwardPass() {
        let unet = UNetArchitecture(inChannels: 3, numClasses: 2, baseChannels: 8)
        #expect(unet.inChannels == 3)
        #expect(unet.numClasses == 2)

        let x = MLXArray.zeros([1, 16, 16, 3])
        let logits = unet(x)
        #expect(logits.shape == [1, 16, 16, 2])
    }

    @Test("CNNFeatureExtractor extracts global average pooling feature vector")
    func testCNNFeatureExtractor() {
        let extractor = CNNFeatureExtractor()
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        let img = ImageDataset(width: 2, height: 2, channels: 2, data: data)
        let features = extractor.extractFeatures(image: img)
        #expect(features.count == 2)
        #expect(features[0] == 2.5) // Mean of 1, 2, 3, 4
        #expect(features[1] == 6.5) // Mean of 5, 6, 7, 8
    }

    @Test("CNNFeatureExtractor handles empty image")
    func testCNNFeatureExtractorEmpty() {
        let extractor = CNNFeatureExtractor()
        let img = ImageDataset(width: 0, height: 0, channels: 0, data: [])
        let features = extractor.extractFeatures(image: img)
        #expect(features.isEmpty)
    }

    @Test("VisionError localizedDescription")
    func testVisionError() {
        let err1 = VisionError.notImplemented("feature X")
        let err2 = VisionError.invalidInput("bad dims")
        #expect(err1.errorDescription?.contains("feature X") == true)
        #expect(err2.errorDescription?.contains("bad dims") == true)
    }
}
