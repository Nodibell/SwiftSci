import Foundation

/// Comprehensive empirical evaluation metrics for object detection models, including mAP@50 and mAP@50:95.
public struct DetectionMetrics: Sendable, Codable, Equatable {
    /// Mean Average Precision evaluated at IoU threshold = 0.50 (mAP@50), in range [0.0, 1.0].
    public let map50: Double
    /// Mean Average Precision averaged across IoU thresholds [0.50, 0.55, ..., 0.95] (mAP@50:95).
    public let map50_95: Double
    /// Mean Intersection-over-Union across all true-positive detections.
    public let meanIoU: Double
    /// Overall precision across all evaluated classes at IoU = 0.50.
    public let precision: Double
    /// Overall recall across all evaluated classes at IoU = 0.50.
    public let recall: Double
    /// Overall harmonic mean F1-score at IoU = 0.50.
    public let f1: Double
    /// Per-class Average Precision scores at IoU = 0.50 strictly for evaluated ground-truth target classes.
    public let perClassAP50: [String: Double]
    /// Labels predicted by candidate detections that do not exist in ground-truth labels (all treated as false positives).
    public let predictionOnlyClasses: [String]
    /// Total number of ground-truth target bounding boxes across the dataset.
    public let totalGroundTruths: Int
    /// Total number of predicted candidate bounding boxes evaluated.
    public let totalDetections: Int
    /// Descriptive human-readable summary string for reports and UI dashboards.
    public let summary: String

    /// Initializes a new detection metrics record.
    public init(
        map50: Double,
        map50_95: Double,
        meanIoU: Double,
        precision: Double,
        recall: Double,
        f1: Double,
        perClassAP50: [String: Double],
        predictionOnlyClasses: [String] = [],
        totalGroundTruths: Int,
        totalDetections: Int,
        summary: String
    ) {
        self.map50 = map50
        self.map50_95 = map50_95
        self.meanIoU = meanIoU
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.perClassAP50 = perClassAP50
        self.predictionOnlyClasses = predictionOnlyClasses
        self.totalGroundTruths = totalGroundTruths
        self.totalDetections = totalDetections
        self.summary = summary
    }
}

/// Evaluator computing standard object detection benchmarks (mAP@50, mAP@50:95, IoU, precision, recall).
public enum ObjectDetectionEvaluator {

    /// Evaluates object detection predictions against ground-truth bounding boxes across a dataset.
    ///
    /// - Parameters:
    ///   - predictions: Array of predicted bounding boxes per image `[numImages][numBoxes]`.
    ///   - groundTruths: Array of true ground-truth bounding boxes per image `[numImages][numBoxes]`.
    ///   - iouThreshold: Minimum IoU threshold to classify a detection as True Positive (default: 0.50, range: (0.0, 1.0]).
    /// - Throws: `VisionError.dimensionMismatch` if prediction and ground-truth image counts differ, or `VisionError.invalidInput` if iouThreshold is out of bounds.
    /// - Returns: A complete `DetectionMetrics` evaluation summary.
    public static func evaluate(
        predictions: [[BoundingBox]],
        groundTruths: [[BoundingBox]],
        iouThreshold: Double = 0.50
    ) throws -> DetectionMetrics {
        guard predictions.count == groundTruths.count else {
            throw VisionError.dimensionMismatch(
                "Prediction image sets count (\(predictions.count)) does not match ground truth image sets count (\(groundTruths.count))."
            )
        }

        guard iouThreshold > 0.0 && iouThreshold <= 1.0 else {
            throw VisionError.invalidInput(
                "iouThreshold must be strictly positive and at most 1.0 (range: (0.0, 1.0]), received \(iouThreshold)."
            )
        }

        let numImages = predictions.count
        guard numImages > 0 else {
            return DetectionMetrics(
                map50: 0.0, map50_95: 0.0, meanIoU: 0.0, precision: 0.0, recall: 0.0, f1: 0.0,
                perClassAP50: [:], predictionOnlyClasses: [], totalGroundTruths: 0, totalDetections: 0,
                summary: "Empty dataset: 0 images evaluated."
            )
        }

        // Collect all target ground truth classes (defines the canonical benchmark taxonomy)
        var gtClasses = Set<String>()
        var totalGT = 0
        for i in 0..<numImages {
            totalGT += groundTruths[i].count
            for box in groundTruths[i] { gtClasses.insert(box.classLabel) }
        }

        // Collect all predicted classes
        var allPredClasses = Set<String>()
        var totalPred = 0
        for i in 0..<numImages {
            totalPred += predictions[i].count
            for box in predictions[i] { allPredClasses.insert(box.classLabel) }
        }

        guard totalGT > 0 || totalPred > 0 else {
            return DetectionMetrics(
                map50: 0.0, map50_95: 0.0, meanIoU: 0.0, precision: 0.0, recall: 0.0, f1: 0.0,
                perClassAP50: [:], predictionOnlyClasses: [], totalGroundTruths: 0, totalDetections: 0,
                summary: "No labeled objects or detections in dataset across \(numImages) image(s)."
            )
        }

        guard !gtClasses.isEmpty else {
            return DetectionMetrics(
                map50: 0.0, map50_95: 0.0, meanIoU: 0.0, precision: 0.0, recall: 0.0, f1: 0.0,
                perClassAP50: [:], predictionOnlyClasses: Array(allPredClasses).sorted(), totalGroundTruths: 0, totalDetections: totalPred,
                summary: "No ground-truth target objects in dataset (\(totalPred) false-positive detections across \(numImages) image(s))."
            )
        }

        // Evaluate AP@50 per ground-truth target class
        var perClassAP50: [String: Double] = [:]
        var allMatchedIoUs: [Double] = []
        var totalTP50 = 0
        var totalFP50 = 0

        for className in gtClasses.sorted() {
            let evalResult = evaluateClassAP(
                predictions: predictions,
                groundTruths: groundTruths,
                className: className,
                iouThreshold: iouThreshold,
                numImages: numImages
            )
            perClassAP50[className] = evalResult.ap
            totalTP50 += evalResult.tp
            totalFP50 += evalResult.fp
            allMatchedIoUs.append(contentsOf: evalResult.matchedIoUs)
        }

        // Account for prediction-only classes: classes predicted by the model that have zero ground truth instances.
        // Every detection in a prediction-only class is a False Positive, penalizing overall precision without deflating mAP denominator.
        let predictionOnlyClasses = Array(allPredClasses.subtracting(gtClasses)).sorted()
        for className in predictionOnlyClasses {
            var classPredCount = 0
            for preds in predictions {
                classPredCount += preds.filter { $0.classLabel == className }.count
            }
            totalFP50 += classPredCount
        }

        // Mean Average Precision is averaged strictly across ground-truth benchmark classes
        let map50 = perClassAP50.values.reduce(0.0, +) / Double(gtClasses.count)
        let meanIoU = allMatchedIoUs.isEmpty ? 0.0 : allMatchedIoUs.reduce(0.0, +) / Double(allMatchedIoUs.count)

        // Precision & Recall at IoU = 0.50
        let precision = (totalTP50 + totalFP50) > 0 ? Double(totalTP50) / Double(totalTP50 + totalFP50) : 0.0
        let recall = totalGT > 0 ? Double(totalTP50) / Double(totalGT) : 0.0
        let f1 = (precision + recall) > 0 ? (2.0 * precision * recall) / (precision + recall) : 0.0

        // Compute mAP@50:95 across IoU steps 0.50...0.95 (step 0.05) strictly across ground-truth classes
        var mapSteps: [Double] = []
        let iouSteps = stride(from: 0.50, through: 0.95, by: 0.05)
        for thresh in iouSteps {
            var stepAPs: [Double] = []
            for className in gtClasses.sorted() {
                let res = evaluateClassAP(
                    predictions: predictions,
                    groundTruths: groundTruths,
                    className: className,
                    iouThreshold: thresh,
                    numImages: numImages
                )
                stepAPs.append(res.ap)
            }
            let stepMAP = stepAPs.isEmpty ? 0.0 : stepAPs.reduce(0.0, +) / Double(stepAPs.count)
            mapSteps.append(stepMAP)
        }
        let map50_95 = mapSteps.isEmpty ? 0.0 : mapSteps.reduce(0.0, +) / Double(mapSteps.count)

        var summary = "Object Detection mAP@50: \(String(format: "%.1f%%", map50 * 100)) | mAP@50:95: \(String(format: "%.1f%%", map50_95 * 100)) | Precision: \(String(format: "%.1f%%", precision * 100)) | Recall: \(String(format: "%.1f%%", recall * 100)) | Mean IoU: \(String(format: "%.3f", meanIoU)) across \(numImages) image(s) (\(totalPred) detections, \(totalGT) ground truths)."
        if !predictionOnlyClasses.isEmpty {
            summary += " Prediction-only classes: [\(predictionOnlyClasses.joined(separator: ", "))]."
        }

        return DetectionMetrics(
            map50: map50,
            map50_95: map50_95,
            meanIoU: meanIoU,
            precision: precision,
            recall: recall,
            f1: f1,
            perClassAP50: perClassAP50,
            predictionOnlyClasses: predictionOnlyClasses,
            totalGroundTruths: totalGT,
            totalDetections: totalPred,
            summary: summary
        )
    }

    // MARK: - Internal AP Calculation

    private struct ClassEvalResult {
        let ap: Double
        let tp: Int
        let fp: Int
        let matchedIoUs: [Double]
    }

    private static func evaluateClassAP(
        predictions: [[BoundingBox]],
        groundTruths: [[BoundingBox]],
        className: String,
        iouThreshold: Double,
        numImages: Int
    ) -> ClassEvalResult {
        // Collect all predicted boxes for this class across all images, paired with their imageIndex and detectionIndex
        struct CandidateDetection {
            let imageIndex: Int
            let originalDetectionIndex: Int
            let box: BoundingBox
        }

        var candidateDetections: [CandidateDetection] = []
        var totalClassGT = 0

        // Track ground truth boxes and their matched state per image
        var gtPerImage: [[[BoundingBox]]] = []
        var gtMatched: [[[Bool]]] = []

        for imgIdx in 0..<numImages {
            let imgGT = groundTruths[imgIdx].filter { $0.classLabel == className }
            totalClassGT += imgGT.count
            gtPerImage.append([imgGT])
            gtMatched.append([[Bool](repeating: false, count: imgGT.count)])

            for (boxIdx, p) in predictions[imgIdx].enumerated() where p.classLabel == className {
                candidateDetections.append(CandidateDetection(imageIndex: imgIdx, originalDetectionIndex: boxIdx, box: p))
            }
        }

        guard totalClassGT > 0 || !candidateDetections.isEmpty else {
            return ClassEvalResult(ap: 1.0, tp: 0, fp: 0, matchedIoUs: [])
        }

        guard totalClassGT > 0 else {
            return ClassEvalResult(ap: 0.0, tp: 0, fp: candidateDetections.count, matchedIoUs: [])
        }

        // Deterministic sort: confidence descending, then imageIndex ascending, then originalDetectionIndex ascending
        candidateDetections.sort {
            if $0.box.confidence != $1.box.confidence {
                return $0.box.confidence > $1.box.confidence
            }
            if $0.imageIndex != $1.imageIndex {
                return $0.imageIndex < $1.imageIndex
            }
            return $0.originalDetectionIndex < $1.originalDetectionIndex
        }

        var tp = [Double](repeating: 0.0, count: candidateDetections.count)
        var fp = [Double](repeating: 0.0, count: candidateDetections.count)
        var matchedIoUs: [Double] = []

        for (detIdx, candidate) in candidateDetections.enumerated() {
            let imgIdx = candidate.imageIndex
            let candidateBox = candidate.box
            let imgGTs = gtPerImage[imgIdx][0]

            var bestIoU = 0.0
            var bestGTIdx = -1

            for (gtIdx, gtBox) in imgGTs.enumerated() {
                let currentIoU = candidateBox.iou(with: gtBox)
                if currentIoU > bestIoU {
                    bestIoU = currentIoU
                    bestGTIdx = gtIdx
                }
            }

            if bestIoU >= iouThreshold && bestGTIdx >= 0 {
                if !gtMatched[imgIdx][0][bestGTIdx] {
                    tp[detIdx] = 1.0
                    gtMatched[imgIdx][0][bestGTIdx] = true
                    matchedIoUs.append(bestIoU)
                } else {
                    // Duplicate detection for already matched ground-truth
                    fp[detIdx] = 1.0
                }
            } else {
                fp[detIdx] = 1.0
            }
        }

        let totalTPCount = Int(tp.reduce(0.0, +))
        let totalFPCount = Int(fp.reduce(0.0, +))

        // Compute cumulative precision and recall
        var cumTP = 0.0
        var cumFP = 0.0
        var precisions: [Double] = []
        var recalls: [Double] = []

        for i in 0..<candidateDetections.count {
            cumTP += tp[i]
            cumFP += fp[i]
            let p = cumTP / (cumTP + cumFP)
            let r = cumTP / Double(totalClassGT)
            precisions.append(p)
            recalls.append(r)
        }

        // Continuous precision-envelope all-point interpolation
        let ap = computeAreaUnderPRCurve(recalls: recalls, precisions: precisions)

        return ClassEvalResult(ap: ap, tp: totalTPCount, fp: totalFPCount, matchedIoUs: matchedIoUs)
    }

    private static func computeAreaUnderPRCurve(recalls: [Double], precisions: [Double]) -> Double {
        guard !recalls.isEmpty, !precisions.isEmpty else { return 0.0 }
        let mrec = [0.0] + recalls + [1.0]
        var mpre = [0.0] + precisions + [0.0]

        // Monotonic envelope: mpre[i] = max(mpre[i], mpre[i+1]) from right to left
        for i in stride(from: mpre.count - 2, through: 0, by: -1) {
            mpre[i] = max(mpre[i], mpre[i + 1])
        }

        // Sum trapezoidal segments where recall changes
        var ap = 0.0
        for i in 1..<mrec.count {
            if mrec[i] != mrec[i - 1] {
                ap += (mrec[i] - mrec[i - 1]) * mpre[i]
            }
        }
        return max(0.0, min(1.0, ap))
    }
}

// MARK: - VisionMetrics Extension

extension VisionMetrics {
    /// Computes Mean Average Precision at IoU threshold = 0.50 (mAP@50).
    /// - Parameters:
    ///   - predictions: Predicted bounding boxes per image.
    ///   - groundTruths: True ground truth bounding boxes per image.
    /// - Throws: `VisionError` if image counts mismatch or parameters are invalid.
    /// - Returns: mAP@50 in range [0.0, 1.0].
    public static func meanAveragePrecision50(
        predictions: [[BoundingBox]],
        groundTruths: [[BoundingBox]]
    ) throws -> Double {
        try ObjectDetectionEvaluator.evaluate(predictions: predictions, groundTruths: groundTruths, iouThreshold: 0.50).map50
    }

    /// Evaluates object detection predictions and computes a full `DetectionMetrics` report.
    /// - Parameters:
    ///   - predictions: Predicted bounding boxes per image.
    ///   - groundTruths: True ground truth bounding boxes per image.
    ///   - iouThreshold: IoU threshold for matching (default 0.50).
    /// - Throws: `VisionError` if image counts mismatch or parameters are invalid.
    /// - Returns: Detailed detection metrics.
    public static func evaluateDetection(
        predictions: [[BoundingBox]],
        groundTruths: [[BoundingBox]],
        iouThreshold: Double = 0.50
    ) throws -> DetectionMetrics {
        try ObjectDetectionEvaluator.evaluate(predictions: predictions, groundTruths: groundTruths, iouThreshold: iouThreshold)
    }
}
