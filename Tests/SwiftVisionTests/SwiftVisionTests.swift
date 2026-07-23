import Testing
import Foundation
@testable import SwiftVision

@Suite("SwiftVision Tests")
struct SwiftVisionTests {
    @Test("Test Vision Metrics calculation")
    func testVisionMetrics() {
        let pred: [[Double]] = [[1, 0], [1, 1]]
        let truth: [[Double]] = [[1, 0], [0, 1]]

        let dice = VisionMetrics.diceCoefficient(predicted: pred, groundTruth: truth)
        let iou = VisionMetrics.iouScore(predicted: pred, groundTruth: truth)

        #expect(dice > 0.0)
        #expect(iou > 0.0)
    }

    @Test("Test BoundingBox IoU")
    func testBoundingBoxIoU() {
        let box1 = BoundingBox(xMin: 0, yMin: 0, xMax: 10, yMax: 10, confidence: 0.9, classLabel: "cat")
        let box2 = BoundingBox(xMin: 5, yMin: 0, xMax: 15, yMax: 10, confidence: 0.8, classLabel: "cat")

        let iou = box1.iou(with: box2)
        #expect(iou > 0.0)
    }

    @Test("Test UNetSegmentation Model")
    func testUNetModel() async throws {
        let img = ImageDataset(width: 4, height: 4, channels: 1, data: Array(repeating: 0.8, count: 16))
        let unet = UNetSegmentationModel(inputChannels: 1, numClasses: 2)
        let mask = try await unet.predict(image: img)

        #expect(mask.count == 4)
        #expect(mask[0].count == 4)
    }
}
