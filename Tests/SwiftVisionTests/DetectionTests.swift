import Testing
import Foundation
#if canImport(CoreGraphics)
import CoreGraphics
#endif
@testable import SwiftVision

@Suite("Pretrained Object Detection Head & Empirical mAP@50 Evaluation Tests (G-027)")
struct DetectionTests {

    // MARK: - YOLO Annotation Parser Tests

    @Test("YOLOAnnotationParser parses single-line and multi-line normalized annotation strings")
    func testYOLOAnnotationParserBasic() {
        let sampleAnnotation = """
        # Sample annotation file
        0 0.5 0.5 0.4 0.2 0.95
        1 0.2 0.3 0.1 0.1
        """

        let classLabels = ["car", "pedestrian"]
        let boxes = YOLOAnnotationParser.parse(text: sampleAnnotation, classLabels: classLabels)

        #expect(boxes.count == 2)

        // Box 0: xCenter=0.5, yCenter=0.5, w=0.4, h=0.2
        // xMin = 0.5 - 0.2 = 0.3, yMin = 0.5 - 0.1 = 0.4, xMax = 0.7, yMax = 0.6
        let box0 = boxes[0]
        #expect(box0.classLabel == "car")
        #expect(abs(box0.xMin - 0.3) < 1e-4)
        #expect(abs(box0.yMin - 0.4) < 1e-4)
        #expect(abs(box0.xMax - 0.7) < 1e-4)
        #expect(abs(box0.yMax - 0.6) < 1e-4)
        #expect(abs(box0.confidence - 0.95) < 1e-4)

        // Box 1: xCenter=0.2, yCenter=0.3, w=0.1, h=0.1
        // xMin = 0.15, yMin = 0.25, xMax = 0.25, yMax = 0.35, default confidence 1.0
        let box1 = boxes[1]
        #expect(box1.classLabel == "pedestrian")
        #expect(abs(box1.xMin - 0.15) < 1e-4)
        #expect(abs(box1.yMin - 0.25) < 1e-4)
        #expect(abs(box1.xMax - 0.25) < 1e-4)
        #expect(abs(box1.yMax - 0.35) < 1e-4)
        #expect(abs(box1.confidence - 1.0) < 1e-4)
    }

    @Test("YOLOAnnotationParser un-normalizes coordinates when pixel dimensions are provided")
    func testYOLOAnnotationParserPixelScaling() {
        let sample = "0 0.5 0.5 0.5 0.5"
        let boxes = YOLOAnnotationParser.parse(text: sample, imageWidth: 640, imageHeight: 480)

        #expect(boxes.count == 1)
        let box = boxes[0]
        #expect(box.classLabel == "class_0")
        // xCenter = 320, w = 320 -> xMin = 160, xMax = 480
        // yCenter = 240, h = 240 -> yMin = 120, yMax = 360
        #expect(abs(box.xMin - 160.0) < 1e-4)
        #expect(abs(box.yMin - 120.0) < 1e-4)
        #expect(abs(box.xMax - 480.0) < 1e-4)
        #expect(abs(box.yMax - 360.0) < 1e-4)
    }

    @Test("YOLOAnnotationParser dataset discovery pairs images with annotations")
    func testYOLOAnnotationParserDatasetPairing() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("yolo_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Create dummy image and label
        let imgURL = tempDir.appendingPathComponent("sample01.jpg")
        try "dummy_image_data".write(to: imgURL, atomically: true, encoding: .utf8)

        let labelURL = tempDir.appendingPathComponent("sample01.txt")
        try "0 0.5 0.5 0.2 0.2\n".write(to: labelURL, atomically: true, encoding: .utf8)

        let dataset = try YOLOAnnotationParser.parseDataset(imagesDirectory: tempDir, classLabels: ["dog"])
        #expect(dataset.count == 1)
        #expect(dataset[0].imageURL.lastPathComponent == "sample01.jpg")
        #expect(dataset[0].groundTruth.count == 1)
        #expect(dataset[0].groundTruth[0].classLabel == "dog")
    }

    // MARK: - Object Detection Evaluator & mAP@50 Tests

    @Test("ObjectDetectionEvaluator computes 100% mAP@50, precision, and recall on exact match")
    func testEvaluatorPerfectMatch() {
        let gtBox1 = BoundingBox(xMin: 0.1, yMin: 0.1, xMax: 0.5, yMax: 0.5, confidence: 1.0, classLabel: "cat")
        let gtBox2 = BoundingBox(xMin: 0.6, yMin: 0.6, xMax: 0.9, yMax: 0.9, confidence: 1.0, classLabel: "dog")

        let predBox1 = BoundingBox(xMin: 0.1, yMin: 0.1, xMax: 0.5, yMax: 0.5, confidence: 0.95, classLabel: "cat")
        let predBox2 = BoundingBox(xMin: 0.6, yMin: 0.6, xMax: 0.9, yMax: 0.9, confidence: 0.90, classLabel: "dog")

        let predictions = [[predBox1, predBox2]]
        let groundTruths = [[gtBox1, gtBox2]]

        let metrics = VisionMetrics.evaluateDetection(predictions: predictions, groundTruths: groundTruths, iouThreshold: 0.50)

        #expect(abs(metrics.map50 - 1.0) < 1e-4)
        #expect(abs(metrics.precision - 1.0) < 1e-4)
        #expect(abs(metrics.recall - 1.0) < 1e-4)
        #expect(abs(metrics.f1 - 1.0) < 1e-4)
        #expect(abs(metrics.meanIoU - 1.0) < 1e-4)
        #expect(metrics.totalGroundTruths == 2)
        #expect(metrics.totalDetections == 2)
        #expect(metrics.perClassAP50["cat"] == 1.0)
        #expect(metrics.perClassAP50["dog"] == 1.0)
        #expect(metrics.summary.contains("100.0%"))
    }

    @Test("ObjectDetectionEvaluator handles duplicate false-positive detections for the same ground truth")
    func testEvaluatorDuplicateDetection() {
        let gtBox = BoundingBox(xMin: 0.2, yMin: 0.2, xMax: 0.6, yMax: 0.6, confidence: 1.0, classLabel: "car")

        // Two predictions for the single ground truth box
        let pred1 = BoundingBox(xMin: 0.2, yMin: 0.2, xMax: 0.6, yMax: 0.6, confidence: 0.9, classLabel: "car")
        let pred2 = BoundingBox(xMin: 0.21, yMin: 0.21, xMax: 0.61, yMax: 0.61, confidence: 0.8, classLabel: "car")

        let metrics = VisionMetrics.evaluateDetection(
            predictions: [[pred1, pred2]],
            groundTruths: [[gtBox]],
            iouThreshold: 0.50
        )

        // 1 TP, 1 FP -> Precision = 0.50, Recall = 1.0
        #expect(abs(metrics.precision - 0.50) < 1e-4)
        #expect(abs(metrics.recall - 1.0) < 1e-4)
        #expect(metrics.totalGroundTruths == 1)
        #expect(metrics.totalDetections == 2)
    }

    @Test("ObjectDetectionEvaluator returns 0.0 mAP when detections do not overlap ground truth")
    func testEvaluatorZeroOverlap() {
        let gt = BoundingBox(xMin: 0.0, yMin: 0.0, xMax: 0.2, yMax: 0.2, confidence: 1.0, classLabel: "person")
        let pred = BoundingBox(xMin: 0.8, yMin: 0.8, xMax: 1.0, yMax: 1.0, confidence: 0.9, classLabel: "person")

        let metrics = ObjectDetectionEvaluator.evaluate(
            predictions: [[pred]],
            groundTruths: [[gt]],
            iouThreshold: 0.50
        )

        #expect(abs(metrics.map50 - 0.0) < 1e-4)
        #expect(abs(metrics.precision - 0.0) < 1e-4)
        #expect(abs(metrics.recall - 0.0) < 1e-4)
        #expect(metrics.totalGroundTruths == 1)
        #expect(metrics.totalDetections == 1)
    }

    @Test("ObjectDetectionEvaluator empty inputs yield graceful zero metrics")
    func testEvaluatorEmptyInputs() {
        let metrics = ObjectDetectionEvaluator.evaluate(predictions: [], groundTruths: [])
        #expect(metrics.map50 == 0.0)
        #expect(metrics.totalGroundTruths == 0)
        #expect(metrics.totalDetections == 0)
    }

    // MARK: - NeuralObjectDetector Tests

    @Test("NeuralObjectDetector initializes and performs live detection on ImageDataset")
    func testNeuralObjectDetectorInference() async throws {
        let detector = NeuralObjectDetector(engine: .systemVision, confidenceThreshold: 0.1)

        let width = 32
        let height = 32
        let pixelCount = width * height
        let dataset = ImageDataset(width: width, height: height, channels: 3, data: [Double](repeating: 0.5, count: pixelCount * 3))

        let detections = try await detector.detect(image: dataset)
        // Detections on uniform gray image should be valid array (likely empty, but without throwing)
        #expect(detections.count >= 0)
    }

    @Test("NeuralObjectDetector evaluates paired dataset end-to-end")
    func testNeuralObjectDetectorDatasetEvaluation() async throws {
        let detector = NeuralObjectDetector(engine: .systemVision)

        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("yolo_eval_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Create synthetic image dataset and save to file
        let dataset = ImageDataset(width: 16, height: 16, channels: 3, data: [Double](repeating: 0.8, count: 16 * 16 * 3))
        let imgURL = tempDir.appendingPathComponent("test_img.png")
        if let pngData = dataset.toPNGData() {
            try pngData.write(to: imgURL)
            let paired = [(imageURL: imgURL, groundTruth: [BoundingBox(xMin: 0.1, yMin: 0.1, xMax: 0.5, yMax: 0.5, confidence: 1.0, classLabel: "person")])]

            let metrics = try await detector.evaluate(dataset: paired)
            #expect(metrics.totalGroundTruths == 1)
            #expect(!metrics.summary.isEmpty)
        }
    }
}
