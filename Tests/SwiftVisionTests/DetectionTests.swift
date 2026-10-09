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
    func testEvaluatorPerfectMatch() throws {
        let gtBox1 = BoundingBox(xMin: 0.1, yMin: 0.1, xMax: 0.5, yMax: 0.5, confidence: 1.0, classLabel: "cat")
        let gtBox2 = BoundingBox(xMin: 0.6, yMin: 0.6, xMax: 0.9, yMax: 0.9, confidence: 1.0, classLabel: "dog")

        let predBox1 = BoundingBox(xMin: 0.1, yMin: 0.1, xMax: 0.5, yMax: 0.5, confidence: 0.95, classLabel: "cat")
        let predBox2 = BoundingBox(xMin: 0.6, yMin: 0.6, xMax: 0.9, yMax: 0.9, confidence: 0.90, classLabel: "dog")

        let predictions = [[predBox1, predBox2]]
        let groundTruths = [[gtBox1, gtBox2]]

        let metrics = try VisionMetrics.evaluateDetection(predictions: predictions, groundTruths: groundTruths, iouThreshold: 0.50)

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
    func testEvaluatorDuplicateDetection() throws {
        let gtBox = BoundingBox(xMin: 0.2, yMin: 0.2, xMax: 0.6, yMax: 0.6, confidence: 1.0, classLabel: "car")

        // Two predictions for the single ground truth box
        let pred1 = BoundingBox(xMin: 0.2, yMin: 0.2, xMax: 0.6, yMax: 0.6, confidence: 0.9, classLabel: "car")
        let pred2 = BoundingBox(xMin: 0.21, yMin: 0.21, xMax: 0.61, yMax: 0.61, confidence: 0.8, classLabel: "car")

        let metrics = try VisionMetrics.evaluateDetection(
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
    func testEvaluatorZeroOverlap() throws {
        let gt = BoundingBox(xMin: 0.0, yMin: 0.0, xMax: 0.2, yMax: 0.2, confidence: 1.0, classLabel: "person")
        let pred = BoundingBox(xMin: 0.8, yMin: 0.8, xMax: 1.0, yMax: 1.0, confidence: 0.9, classLabel: "person")

        let metrics = try ObjectDetectionEvaluator.evaluate(
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
    func testEvaluatorEmptyInputs() throws {
        let metrics = try ObjectDetectionEvaluator.evaluate(predictions: [], groundTruths: [])
        #expect(metrics.map50 == 0.0)
        #expect(metrics.totalGroundTruths == 0)
        #expect(metrics.totalDetections == 0)
    }

    @Test("ObjectDetectionEvaluator throws dimensionMismatch when prediction and ground-truth image counts differ")
    func testEvaluatorDimensionMismatch() {
        let box = BoundingBox(xMin: 0.1, yMin: 0.1, xMax: 0.5, yMax: 0.5, confidence: 1.0, classLabel: "cat")
        #expect(throws: VisionError.self) {
            _ = try ObjectDetectionEvaluator.evaluate(
                predictions: [[box], [box]],
                groundTruths: [[box]]
            )
        }
    }

    @Test("ObjectDetectionEvaluator throws invalidInput when iouThreshold is out of range")
    func testEvaluatorInvalidIoUThreshold() {
        let box = BoundingBox(xMin: 0.1, yMin: 0.1, xMax: 0.5, yMax: 0.5, confidence: 1.0, classLabel: "cat")
        #expect(throws: VisionError.self) {
            _ = try ObjectDetectionEvaluator.evaluate(predictions: [[box]], groundTruths: [[box]], iouThreshold: -0.1)
        }
        #expect(throws: VisionError.self) {
            _ = try ObjectDetectionEvaluator.evaluate(predictions: [[box]], groundTruths: [[box]], iouThreshold: 0.0)
        }
        #expect(throws: VisionError.self) {
            _ = try ObjectDetectionEvaluator.evaluate(predictions: [[box]], groundTruths: [[box]], iouThreshold: 1.5)
        }
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

    @Test("Precision-envelope AP reference fixture with duplicate detections and IoU thresholds")
    func testMAPReferenceFixtureWithEnvelopeAndDuplicates() throws {
        // Ground truth: 5 objects of class "target"
        let gt0 = BoundingBox(xMin: 0.1, yMin: 0.1, xMax: 0.3, yMax: 0.3, confidence: 1.0, classLabel: "target")
        let gt1 = BoundingBox(xMin: 0.4, yMin: 0.4, xMax: 0.6, yMax: 0.6, confidence: 1.0, classLabel: "target")
        let gt2 = BoundingBox(xMin: 0.7, yMin: 0.7, xMax: 0.9, yMax: 0.9, confidence: 1.0, classLabel: "target")
        let gt3 = BoundingBox(xMin: 0.1, yMin: 0.7, xMax: 0.3, yMax: 0.9, confidence: 1.0, classLabel: "target")
        let gt4 = BoundingBox(xMin: 0.7, yMin: 0.1, xMax: 0.9, yMax: 0.3, confidence: 1.0, classLabel: "target")

        // Predictions:
        // Det 0: exact match gt0, IoU = 1.0, conf = 0.95 -> TP
        let pred0 = BoundingBox(xMin: 0.1, yMin: 0.1, xMax: 0.3, yMax: 0.3, confidence: 0.95, classLabel: "target")
        // Det 1: exact match gt1, IoU = 1.0, conf = 0.85 -> TP
        let pred1 = BoundingBox(xMin: 0.4, yMin: 0.4, xMax: 0.6, yMax: 0.6, confidence: 0.85, classLabel: "target")
        // Det 2: duplicate match on gt1, IoU = 0.85, conf = 0.75 -> FP (duplicate)
        let pred2 = BoundingBox(xMin: 0.41, yMin: 0.41, xMax: 0.61, yMax: 0.61, confidence: 0.75, classLabel: "target")
        // Det 3: false positive, no overlap, conf = 0.65 -> FP (spurious)
        let pred3 = BoundingBox(xMin: 0.0, yMin: 0.4, xMax: 0.05, yMax: 0.45, confidence: 0.65, classLabel: "target")
        // Det 4: exact match gt2, IoU = 1.0, conf = 0.55 -> TP
        let pred4 = BoundingBox(xMin: 0.7, yMin: 0.7, xMax: 0.9, yMax: 0.9, confidence: 0.55, classLabel: "target")

        let metrics = try VisionMetrics.evaluateDetection(
            predictions: [[pred0, pred1, pred2, pred3, pred4]],
            groundTruths: [[gt0, gt1, gt2, gt3, gt4]],
            iouThreshold: 0.50
        )

        #expect(metrics.totalGroundTruths == 5)
        #expect(metrics.totalDetections == 5)
        #expect(abs(metrics.precision - 0.60) < 1e-4) // 3 TP / 5 Detections
        #expect(abs(metrics.recall - 0.60) < 1e-4)    // 3 TP / 5 GT
        #expect(abs(metrics.f1 - 0.60) < 1e-4)

        // AP analytically calculated:
        // Rank 1: TP (r=0.2, p=1.0)
        // Rank 2: TP (r=0.4, p=1.0)
        // Rank 3: FP (r=0.4, p=2/3)
        // Rank 4: FP (r=0.4, p=0.5)
        // Rank 5: TP (r=0.6, p=0.6)
        // Continuous envelope: (0.2 - 0.0)*1.0 + (0.4 - 0.2)*1.0 + (0.6 - 0.4)*0.6 = 0.2 + 0.2 + 0.12 = 0.52
        let targetAP = metrics.perClassAP50["target"] ?? 0.0
        #expect(abs(targetAP - 0.52) < 1e-4)
        #expect(abs(metrics.map50 - 0.52) < 1e-4)
        #expect(metrics.map50_95 > 0.0 && metrics.map50_95 <= metrics.map50)
    }

    @Test("Empirical mAP multi-class and edge case evaluation with empty predictions and empty GT")
    func testMAPMultiClassAndEdgeCases() throws {
        // Multi-class evaluation
        let catGT = BoundingBox(xMin: 0.1, yMin: 0.1, xMax: 0.3, yMax: 0.3, confidence: 1.0, classLabel: "cat")
        let dogGT = BoundingBox(xMin: 0.5, yMin: 0.5, xMax: 0.7, yMax: 0.7, confidence: 1.0, classLabel: "dog")

        // Cat is perfectly detected (AP = 1.0), dog is not detected (AP = 0.0)
        let catPred = BoundingBox(xMin: 0.1, yMin: 0.1, xMax: 0.3, yMax: 0.3, confidence: 0.9, classLabel: "cat")

        let metrics = try VisionMetrics.evaluateDetection(
            predictions: [[catPred]],
            groundTruths: [[catGT, dogGT]],
            iouThreshold: 0.50
        )

        #expect(metrics.perClassAP50["cat"] == 1.0)
        #expect(metrics.perClassAP50["dog"] == 0.0)
        #expect(abs(metrics.map50 - 0.50) < 1e-4) // (1.0 + 0.0) / 2

        // Edge case: Image with ground truths but completely empty predictions
        let emptyPredMetrics = try VisionMetrics.evaluateDetection(
            predictions: [[]],
            groundTruths: [[catGT]],
            iouThreshold: 0.50
        )
        #expect(emptyPredMetrics.map50 == 0.0)
        #expect(emptyPredMetrics.recall == 0.0)
        #expect(emptyPredMetrics.totalDetections == 0)
        #expect(emptyPredMetrics.totalGroundTruths == 1)

        // Edge case: Image with predictions but completely empty ground truths
        let emptyGTMetrics = try VisionMetrics.evaluateDetection(
            predictions: [[catPred]],
            groundTruths: [[]],
            iouThreshold: 0.50
        )
        #expect(emptyGTMetrics.map50 == 0.0)
        #expect(emptyGTMetrics.recall == 0.0)
        #expect(emptyGTMetrics.precision == 0.0)
        #expect(emptyGTMetrics.totalDetections == 1)
        #expect(emptyGTMetrics.totalGroundTruths == 0)
    }

    @Test("Prediction-only classes do not deflate mAP denominator but penalize overall precision as false positives")
    func testPredictionOnlyClassesDoNotDeflateMAP() throws {
        // GT contains only "cat" (1 object)
        let catGT = BoundingBox(xMin: 0.1, yMin: 0.1, xMax: 0.3, yMax: 0.3, confidence: 1.0, classLabel: "cat")

        // Model predicts "cat" (matches GT) AND "dog" (spurious prediction, no dog in GT)
        let catPred = BoundingBox(xMin: 0.1, yMin: 0.1, xMax: 0.3, yMax: 0.3, confidence: 0.95, classLabel: "cat")
        let dogPred = BoundingBox(xMin: 0.6, yMin: 0.6, xMax: 0.8, yMax: 0.8, confidence: 0.85, classLabel: "dog")

        let metrics = try VisionMetrics.evaluateDetection(
            predictions: [[catPred, dogPred]],
            groundTruths: [[catGT]],
            iouThreshold: 0.50
        )

        // Cat is perfectly detected -> AP = 1.0
        #expect(metrics.perClassAP50["cat"] == 1.0)
        // Dog has 0 GT instances -> excluded from perClassAP50, tracked in predictionOnlyClasses
        #expect(metrics.perClassAP50["dog"] == nil)
        #expect(metrics.predictionOnlyClasses == ["dog"])

        // Crucial test: mAP is averaged strictly across GT classes (1 class: cat) -> mAP = 1.0 / 1 = 1.0!
        // It must NOT be (1.0 + 0.0) / 2 = 0.5!
        #expect(abs(metrics.map50 - 1.0) < 1e-4)

        // Dog prediction is counted as a False Positive:
        // TP = 1 (cat), FP = 1 (dog) -> Precision = 1 / 2 = 0.50, Recall = 1 / 1 = 1.0
        #expect(abs(metrics.precision - 0.50) < 1e-4)
        #expect(abs(metrics.recall - 1.0) < 1e-4)
        #expect(metrics.totalDetections == 2)
        #expect(metrics.totalGroundTruths == 1)
    }

    @Test("Candidate detections with identical confidence break ties deterministically")
    func testDeterministicTieBreaking() throws {
        let gt = BoundingBox(xMin: 0.1, yMin: 0.1, xMax: 0.3, yMax: 0.3, confidence: 1.0, classLabel: "cat")
        // Two candidate predictions with identical confidence (0.80)
        let p1 = BoundingBox(xMin: 0.1, yMin: 0.1, xMax: 0.3, yMax: 0.3, confidence: 0.80, classLabel: "cat") // IoU = 1.0
        let p2 = BoundingBox(xMin: 0.8, yMin: 0.8, xMax: 0.9, yMax: 0.9, confidence: 0.80, classLabel: "cat") // IoU = 0.0

        let m1 = try ObjectDetectionEvaluator.evaluate(predictions: [[p1, p2]], groundTruths: [[gt]])
        let m2 = try ObjectDetectionEvaluator.evaluate(predictions: [[p1, p2]], groundTruths: [[gt]])

        #expect(m1.map50 == m2.map50)
        #expect(m1.precision == m2.precision)
    }

    @Test("BoundingBox and DetectionMetrics Codable serialization and Equatable checks")
    func testDetectionCodableAndBoundingBoxIoU() throws {
        let boxA = BoundingBox(xMin: 0.1, yMin: 0.2, xMax: 0.5, yMax: 0.6, confidence: 0.95, classLabel: "car")
        let boxB = BoundingBox(xMin: 0.1, yMin: 0.2, xMax: 0.5, yMax: 0.6, confidence: 0.95, classLabel: "car")
        #expect(boxA == boxB)

        let encodedBox = try JSONEncoder().encode(boxA)
        let decodedBox = try JSONDecoder().decode(BoundingBox.self, from: encodedBox)
        #expect(decodedBox == boxA)

        // Zero-area boxes yield 0.0 IoU without division by zero
        let zeroBox1 = BoundingBox(xMin: 0.0, yMin: 0.0, xMax: 0.0, yMax: 0.0, confidence: 1.0, classLabel: "dot")
        let zeroBox2 = BoundingBox(xMin: 0.0, yMin: 0.0, xMax: 0.0, yMax: 0.0, confidence: 1.0, classLabel: "dot")
        #expect(zeroBox1.iou(with: zeroBox2) == 0.0)

        let metrics = DetectionMetrics(
            map50: 0.85,
            map50_95: 0.65,
            meanIoU: 0.78,
            precision: 0.88,
            recall: 0.82,
            f1: 0.85,
            perClassAP50: ["car": 0.85],
            predictionOnlyClasses: ["pedestrian"],
            totalGroundTruths: 10,
            totalDetections: 12,
            summary: "Test summary"
        )
        let encodedMetrics = try JSONEncoder().encode(metrics)
        let decodedMetrics = try JSONDecoder().decode(DetectionMetrics.self, from: encodedMetrics)
        #expect(decodedMetrics == metrics)
    }

    @Test("ObjectDetectionEvaluator empty labels and prediction-only dataset summaries")
    func testEvaluatorEmptyDatasetVariants() throws {
        // Multiple images but zero ground truths and zero detections
        let emptyImagesMetrics = try ObjectDetectionEvaluator.evaluate(
            predictions: [[], []],
            groundTruths: [[], []]
        )
        #expect(emptyImagesMetrics.map50 == 0.0)
        #expect(emptyImagesMetrics.totalGroundTruths == 0)
        #expect(emptyImagesMetrics.totalDetections == 0)
        #expect(emptyImagesMetrics.summary.contains("No labeled objects or detections in dataset across 2 image(s)"))

        // Images with detections but absolutely no ground truths
        let pred = BoundingBox(xMin: 0.1, yMin: 0.1, xMax: 0.5, yMax: 0.5, confidence: 0.9, classLabel: "ghost")
        let noGTMetrics = try ObjectDetectionEvaluator.evaluate(
            predictions: [[pred], []],
            groundTruths: [[], []]
        )
        #expect(noGTMetrics.map50 == 0.0)
        #expect(noGTMetrics.totalGroundTruths == 0)
        #expect(noGTMetrics.totalDetections == 1)
        #expect(noGTMetrics.predictionOnlyClasses == ["ghost"])
        #expect(noGTMetrics.summary.contains("No ground-truth target objects in dataset"))
    }
}
