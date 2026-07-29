import Testing
import Foundation
@testable import SwiftVision

@Suite("SwiftVision Tests (Phase 4)")
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

    @Test("Test BoundingBox IoU and NMS")
    func testBoundingBoxIoU() async {
        let box1 = BoundingBox(xMin: 0, yMin: 0, xMax: 10, yMax: 10, confidence: 0.9, classLabel: "cat")
        let box2 = BoundingBox(xMin: 5, yMin: 0, xMax: 15, yMax: 10, confidence: 0.8, classLabel: "cat")

        let iou = box1.iou(with: box2)
        #expect(iou > 0.0)

        let detector = YOLOv8Detector()
        let nms = await detector.nonMaximumSuppression(boxes: [box1, box2])
        #expect(!nms.isEmpty)
    }

    @Test("Test UNetSegmentation Model predict forward pass")
    func testUNetModelPredict() async throws {
        let img = ImageDataset(width: 8, height: 8, channels: 1, data: Array(repeating: 0.8, count: 64))
        let unet = UNetSegmentationModel(inputChannels: 1, numClasses: 2)
        let mask = try await unet.predict(image: img)

        #expect(mask.count == 8)
        #expect(mask[0].count == 8)
        #expect(mask[0][0] >= 0.0 && mask[0][0] <= 1.0)
    }

    @Test("Test YOLOv8Detector detect forward pass and NMS")
    func testYOLODetectorDetect() async throws {
        let img = ImageDataset(width: 32, height: 32, channels: 3, data: Array(repeating: 0.5, count: 32 * 32 * 3))
        let detector = YOLOv8Detector(confidenceThreshold: 0.25, iouThreshold: 0.45)
        let boxes = try await detector.detect(image: img)

        #expect(!boxes.isEmpty)
        #expect(boxes.first?.confidence ?? 0.0 >= 0.25)
    }
}
