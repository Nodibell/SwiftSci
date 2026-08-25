#if os(macOS)
import Testing
import Foundation
import MLX
@testable import SwiftVision

@Suite("YOLOv8-Seg Instance Segmentation Tests")
struct YOLOSegTests {

    @Test("YOLOSegHead forward pass and mask decoding")
    func testYOLOSegHeadForward() throws {
        let segHead = YOLOSegHead(numClasses: 10, numMasks: 16, regMax: 16)

        // Mock neck outputs: P3 [1, 20, 20, 64], P4 [1, 10, 10, 128], P5 [1, 5, 5, 256]
        let p3 = MLX.zeros([1, 20, 20, 64])
        let p4 = MLX.zeros([1, 10, 10, 128])
        let p5 = MLX.zeros([1, 5, 5, 256])

        let neckOut = YOLONeckOutput(headP3: p3, headP4: p4, headP5: p5)
        let segOut = segHead(neckOut)

        let totalAnchors = 20 * 20 + 10 * 10 + 5 * 5 // 400 + 100 + 25 = 525

        #expect(segOut.boxes.shape == [1, totalAnchors, 4])
        #expect(segOut.scores.shape == [1, totalAnchors, 10])
        #expect(segOut.maskCoefficients.shape == [1, totalAnchors, 16])
        #expect(segOut.protos.shape == [1, 20, 20, 16])

        // Test mask decoding
        let singleCoeff = MLXArray([Float](repeating: 1.0, count: 16))
        let protoMap = MLX.zeros([20, 20, 16])
        let binaryMask = YOLOSegHead.decodeMask(coeff: singleCoeff, protos: protoMap, threshold: 0.4)

        #expect(binaryMask.shape == [20, 20])
    }
}
#endif
