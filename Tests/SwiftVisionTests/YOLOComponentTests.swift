import Testing
import Foundation
import MLX
import MLXNN
@testable import SwiftVision

// MARK: - YOLOPreprocessor Tests

@Suite("YOLOPreprocessor Tests")
struct YOLOPreprocessorTests {

    @Test("Default init produces correct target dimensions")
    func testDefaultInit() {
        let preprocessor = YOLOPreprocessor()
        #expect(preprocessor.targetWidth == 640)
        #expect(preprocessor.targetHeight == 640)
        #expect(abs(preprocessor.paddingColor - 114.0 / 255.0) < 1e-6)
    }

    @Test("Custom init respects parameters")
    func testCustomInit() {
        let preprocessor = YOLOPreprocessor(targetWidth: 320, targetHeight: 320, paddingColor: 0.5)
        #expect(preprocessor.targetWidth == 320)
        #expect(preprocessor.targetHeight == 320)
        #expect(abs(preprocessor.paddingColor - 0.5) < 1e-6)
    }

    @Test("Preprocess returns tensor with correct shape [1, H, W, 3]")
    func testPreprocessOutputShape() {
        let preprocessor = YOLOPreprocessor(targetWidth: 64, targetHeight: 64)
        let img = ImageDataset(width: 32, height: 32, channels: 1, data: Array(repeating: 0.5, count: 32 * 32))
        let tensor = preprocessor.preprocess(image: img)
        #expect(tensor.shape == [1, 64, 64, 3])
    }

    @Test("Preprocess square image same size returns tensor with correct shape")
    func testPreprocessSquareImageSameSize() {
        let preprocessor = YOLOPreprocessor(targetWidth: 32, targetHeight: 32)
        let img = ImageDataset(width: 32, height: 32, channels: 1, data: Array(repeating: 1.0, count: 1024))
        let tensor = preprocessor.preprocess(image: img)
        #expect(tensor.shape == [1, 32, 32, 3])
    }

    @Test("Preprocess tall image returns correct output shape")
    func testPreprocessTallImage() {
        let preprocessor = YOLOPreprocessor(targetWidth: 64, targetHeight: 64)
        let img = ImageDataset(width: 32, height: 64, channels: 1, data: Array(repeating: 0.3, count: 32 * 64))
        let tensor = preprocessor.preprocess(image: img)
        #expect(tensor.shape == [1, 64, 64, 3])
    }

    @Test("Preprocess wide image returns correct output shape")
    func testPreprocessWideImage() {
        let preprocessor = YOLOPreprocessor(targetWidth: 64, targetHeight: 64)
        let img = ImageDataset(width: 64, height: 16, channels: 1, data: Array(repeating: 0.7, count: 64 * 16))
        let tensor = preprocessor.preprocess(image: img)
        #expect(tensor.shape == [1, 64, 64, 3])
    }

    @Test("Padding color fills non-image area")
    func testPaddingColorFillsBackground() {
        let padVal: Float = 0.5
        let preprocessor = YOLOPreprocessor(targetWidth: 64, targetHeight: 64, paddingColor: Double(padVal))
        let img = ImageDataset(width: 32, height: 16, channels: 1, data: Array(repeating: 1.0, count: 32 * 16))
        let tensor = preprocessor.preprocess(image: img)
        let flat = tensor.flattened().asArray(Float.self)
        let hasPad = flat.contains { abs($0 - padVal) < 1e-4 }
        #expect(hasPad)
    }
}

// MARK: - YOLOBackbone Tests

@Suite("YOLOBackbone & Building Block Tests")
struct YOLOBackboneTests {

    @Test("ConvBlock init and forward pass produces correct output channels")
    func testConvBlockForwardShape() {
        let conv = ConvBlock(cIn: 3, cOut: 16, k: 3, s: 1)
        let x = MLXArray.zeros([1, 8, 8, 3])
        let out = conv(x)
        #expect(out.dim(3) == 16)
        #expect(out.dim(0) == 1)
    }

    @Test("ConvBlock stride 2 halves spatial dimensions")
    func testConvBlockStride2() {
        let conv = ConvBlock(cIn: 3, cOut: 16, k: 3, s: 2)
        let x = MLXArray.zeros([1, 16, 16, 3])
        let out = conv(x)
        #expect(out.dim(1) == 8)
        #expect(out.dim(2) == 8)
    }

    @Test("BottleneckBlock with shortcut preserves shape when cIn == cOut")
    func testBottleneckWithShortcut() {
        let block = BottleneckBlock(cIn: 32, cOut: 32, shortcut: true)
        let x = MLXArray.zeros([1, 8, 8, 32])
        let out = block(x)
        #expect(out.shape == [1, 8, 8, 32])
    }

    @Test("BottleneckBlock without shortcut produces correct output shape")
    func testBottleneckNoShortcut() {
        let block = BottleneckBlock(cIn: 16, cOut: 32, shortcut: false)
        let x = MLXArray.zeros([1, 8, 8, 16])
        let out = block(x)
        #expect(out.dim(3) == 32)
    }

    @Test("C2fBlock forward pass produces correct output channel dimension")
    func testC2fBlockOutputChannels() {
        let block = C2fBlock(cIn: 32, cOut: 32, n: 1)
        let x = MLXArray.zeros([1, 8, 8, 32])
        let out = block(x)
        #expect(out.dim(3) == 32)
        #expect(out.shape == [1, 8, 8, 32])
    }

    @Test("SPPFBlock forward pass preserves spatial dimensions")
    func testSPPFBlockShape() {
        let block = SPPFBlock(cIn: 32, cOut: 32, k: 3)
        let x = MLXArray.zeros([1, 8, 8, 32])
        let out = block(x)
        #expect(out.shape == [1, 8, 8, 32])
    }

    @Test("YOLOBackbone forward pass produces 3 multiscale feature maps")
    func testYOLOBackboneForwardPass() {
        let backbone = YOLOBackbone()
        let x = MLXArray.zeros([1, 64, 64, 3])
        let out = backbone(x)
        // P3 stride 8: 64/8 = 8, channels 64
        #expect(out.p3.dim(3) == 64)
        // P4 stride 16: 64/16 = 4, channels 128
        #expect(out.p4.dim(3) == 128)
        // P5 stride 32: 64/32 = 2, channels 256
        #expect(out.p5.dim(3) == 256)
    }

    @Test("YOLOBackboneOutput struct stores feature maps correctly")
    func testYOLOBackboneOutputStruct() {
        let p3 = MLXArray.zeros([1, 8, 8, 64])
        let p4 = MLXArray.zeros([1, 4, 4, 128])
        let p5 = MLXArray.zeros([1, 2, 2, 256])
        let output = YOLOBackboneOutput(p3: p3, p4: p4, p5: p5)
        #expect(output.p3.shape == [1, 8, 8, 64])
        #expect(output.p4.shape == [1, 4, 4, 128])
        #expect(output.p5.shape == [1, 2, 2, 256])
    }
}

// MARK: - YOLONeck Tests

@Suite("YOLONeck Tests")
struct YOLONeckTests {

    @Test("YOLONeck forward pass produces 3 head feature maps with correct output channel counts")
    func testNeckForwardPassChannels() {
        let neck = YOLONeck()
        let backboneOut = YOLOBackboneOutput(
            p3: MLXArray.zeros([1, 8, 8, 64]),
            p4: MLXArray.zeros([1, 4, 4, 128]),
            p5: MLXArray.zeros([1, 2, 2, 256])
        )
        let neckOut = neck(backboneOut)
        #expect(neckOut.headP3.dim(3) == 64)
        #expect(neckOut.headP4.dim(3) == 128)
        #expect(neckOut.headP5.dim(3) == 256)
    }

    @Test("YOLONeckOutput struct stores feature maps correctly")
    func testNeckOutputStruct() {
        let p3 = MLXArray.zeros([1, 8, 8, 64])
        let p4 = MLXArray.zeros([1, 4, 4, 128])
        let p5 = MLXArray.zeros([1, 2, 2, 256])
        let out = YOLONeckOutput(headP3: p3, headP4: p4, headP5: p5)
        #expect(out.headP3.shape == [1, 8, 8, 64])
        #expect(out.headP4.shape == [1, 4, 4, 128])
        #expect(out.headP5.shape == [1, 2, 2, 256])
    }
}

// MARK: - YOLOHead Tests

@Suite("YOLOHead Tests")
struct YOLOHeadTests {

    @Test("YOLOHead default init has correct numClasses and regMax")
    func testDefaultInit() {
        let head = YOLOHead(numClasses: 80, regMax: 16)
        #expect(head.numClasses == 80)
        #expect(head.regMax == 16)
    }

    @Test("YOLOHead custom numClasses init")
    func testCustomNumClasses() {
        let head = YOLOHead(numClasses: 10, regMax: 16)
        #expect(head.numClasses == 10)
    }

    @Test("YOLOHead forward pass produces boxes tensor with 8400 rows for 640x640 input")
    func testHeadForwardPassBoxCount() {
        let head = YOLOHead(numClasses: 80, regMax: 16)
        let neck = YOLONeck()
        let backbone = YOLOBackbone()
        let x = MLXArray.zeros([1, 64, 64, 3])
        let backboneOut = backbone(x)
        let neckOut = neck(backboneOut)
        let headOut = head(neckOut)
        // For a 64x64 input with strides [8,16,32]: 8x8 + 4x4 + 2x2 = 64+16+4 = 84 anchors
        let totalAnchors = headOut.boxes.dim(1)
        #expect(totalAnchors == 84)
        #expect(headOut.boxes.dim(2) == 4) // [x1, y1, x2, y2]
        #expect(headOut.scores.dim(2) == 80) // numClasses
    }

    @Test("YOLOHeadOutput struct init stores tensors correctly")
    func testHeadOutputStruct() {
        let boxes = MLXArray.zeros([1, 84, 4])
        let scores = MLXArray.zeros([1, 84, 80])
        let out = YOLOHeadOutput(boxes: boxes, scores: scores)
        #expect(out.boxes.shape == [1, 84, 4])
        #expect(out.scores.shape == [1, 84, 80])
    }

    @Test("YOLOHead scores are in [0, 1] range (sigmoid activated)")
    func testScoresAreInValidRange() {
        let head = YOLOHead(numClasses: 10, regMax: 16)
        let neck = YOLONeck()
        let backbone = YOLOBackbone()
        let x = MLXArray.zeros([1, 32, 32, 3])
        let backboneOut = backbone(x)
        let neckOut = neck(backboneOut)
        let headOut = head(neckOut)
        let scoreValues = headOut.scores.flattened().asArray(Float.self)
        let allValid = scoreValues.allSatisfy { $0 >= 0.0 && $0 <= 1.0 }
        #expect(allValid)
    }
}
