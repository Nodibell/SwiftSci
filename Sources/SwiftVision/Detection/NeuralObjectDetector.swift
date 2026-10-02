import Foundation
#if canImport(Vision)
import Vision
#endif
#if canImport(CoreGraphics)
import CoreGraphics
#endif

/// High-level neural object detection engine supporting YOLOv8 deep networks and Apple Vision neural detectors.
///
/// Enables out-of-the-box live bounding box prediction and empirical mAP@50 / IoU benchmarking
/// across annotated image datasets without requiring synthetic metrics or manual boilerplate.
public struct NeuralObjectDetector: Sendable {

    /// Underlying detection model engine.
    public enum Engine: Sendable {
        /// Utilizes built-in system neural detectors (Apple Vision framework) requiring no external weights.
        case systemVision
        /// Utilizes YOLOv8 deep convolutional forward pass on Apple Silicon (MLX).
        case yolo(YOLOv8Detector)
    }

    /// The active model engine.
    public let engine: Engine
    /// Confidence threshold filtering out low-probability candidate detections.
    public let confidenceThreshold: Double
    /// Non-Maximum Suppression (NMS) IoU threshold.
    public let iouThreshold: Double

    /// Creates a new NeuralObjectDetector instance.
    /// - Parameters:
    ///   - engine: Desired detection engine (defaults to `.systemVision`).
    ///   - confidenceThreshold: Detection confidence cutoff (default: 0.25).
    ///   - iouThreshold: NMS suppression threshold (default: 0.45).
    public init(
        engine: Engine = .systemVision,
        confidenceThreshold: Double = 0.25,
        iouThreshold: Double = 0.45
    ) {
        self.engine = engine
        self.confidenceThreshold = confidenceThreshold
        self.iouThreshold = iouThreshold
    }

    /// Creates a NeuralObjectDetector backed by a custom configured YOLOv8 instance.
    /// - Parameter yoloDetector: The YOLOv8 detector instance.
    public init(yoloDetector: YOLOv8Detector) {
        self.engine = .yolo(yoloDetector)
        self.confidenceThreshold = 0.25
        self.iouThreshold = 0.45
    }

    // MARK: - Live Detection

    /// Detects object bounding boxes from an image file at a local filesystem URL.
    ///
    /// - Parameter imageURL: Local file URL to the target image.
    /// - Throws: `VisionError.invalidInput` if image cannot be read or processed.
    /// - Returns: List of detected `BoundingBox` instances filtered by confidence and NMS.
    public func detect(imageURL: URL) async throws -> [BoundingBox] {
        switch engine {
        case .systemVision:
            #if canImport(Vision)
            return try await detectWithSystemVision(imageURL: imageURL)
            #else
            let dataset = try ImageDataset.load(from: imageURL)
            return try await detect(image: dataset)
            #endif
        case .yolo(let detector):
            let dataset = try ImageDataset.load(from: imageURL)
            return try await detector.detect(image: dataset)
        }
    }

    /// Detects object bounding boxes from an in-memory `ImageDataset` pixel buffer.
    ///
    /// - Parameter image: Source image dataset.
    /// - Throws: `VisionError.invalidInput` if image dimensions are non-positive.
    /// - Returns: List of detected `BoundingBox` instances.
    public func detect(image: ImageDataset) async throws -> [BoundingBox] {
        switch engine {
        case .yolo(let detector):
            return try await detector.detect(image: image)
        case .systemVision:
            #if canImport(Vision) && canImport(CoreGraphics)
            guard let cgImage = image.toCGImage() else {
                throw VisionError.invalidInput("Failed to render ImageDataset to CGImage")
            }
            return try await detectWithSystemVision(cgImage: cgImage)
            #else
            // Fallback lightweight detector
            let detector = YOLOv8Detector(confidenceThreshold: confidenceThreshold, iouThreshold: iouThreshold)
            return try await detector.detect(image: image)
            #endif
        }
    }

    // MARK: - Empirical Dataset Evaluation (mAP@50)

    /// Evaluates object detection performance across an annotated dataset, returning empirical mAP@50.
    ///
    /// - Parameter dataset: Array of paired image URLs and ground-truth bounding boxes.
    /// - Throws: An error if inference fails on any image.
    /// - Returns: Comprehensive `DetectionMetrics` including mAP@50, mAP@50:95, and mean IoU.
    public func evaluate(
        dataset: [(imageURL: URL, groundTruth: [BoundingBox])]
    ) async throws -> DetectionMetrics {
        guard !dataset.isEmpty else {
            return ObjectDetectionEvaluator.evaluate(predictions: [], groundTruths: [])
        }

        var allPredictions: [[BoundingBox]] = []
        var allGroundTruths: [[BoundingBox]] = []

        for pair in dataset {
            let preds = try await detect(imageURL: pair.imageURL)
            allPredictions.append(preds)
            allGroundTruths.append(pair.groundTruth)
        }

        return ObjectDetectionEvaluator.evaluate(
            predictions: allPredictions,
            groundTruths: allGroundTruths,
            iouThreshold: 0.50
        )
    }

    /// Evaluates an entire annotated dataset directory in standard YOLO format.
    ///
    /// - Parameters:
    ///   - imagesDirectory: Folder containing dataset images.
    ///   - labelsDirectory: Optional custom folder containing `.txt` annotations.
    ///   - classLabels: Mapping of integer class IDs to class name strings.
    /// - Throws: An error if the directory is missing or evaluation fails.
    /// - Returns: Comprehensive `DetectionMetrics` report.
    public func evaluate(
        imagesDirectory: URL,
        labelsDirectory: URL? = nil,
        classLabels: [String] = []
    ) async throws -> DetectionMetrics {
        let pairedDataset = try YOLOAnnotationParser.parseDataset(
            imagesDirectory: imagesDirectory,
            labelsDirectory: labelsDirectory,
            classLabels: classLabels
        )
        return try await evaluate(dataset: pairedDataset)
    }

    // MARK: - Private Helpers

    #if canImport(Vision)
    private func detectWithSystemVision(imageURL: URL) async throws -> [BoundingBox] {
        let animalReq = VNRecognizeAnimalsRequest()
        let humanReq = VNDetectHumanRectanglesRequest()
        let handler = VNImageRequestHandler(url: imageURL, options: [:])

        do {
            try handler.perform([animalReq, humanReq])
        } catch {
            throw VisionError.invalidInput("System Vision request failed: \(error.localizedDescription)")
        }

        return extractBoundingBoxes(from: animalReq, humanReq: humanReq)
    }

    #if canImport(CoreGraphics)
    private func detectWithSystemVision(cgImage: CGImage) async throws -> [BoundingBox] {
        let animalReq = VNRecognizeAnimalsRequest()
        let humanReq = VNDetectHumanRectanglesRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

        do {
            try handler.perform([animalReq, humanReq])
        } catch {
            throw VisionError.invalidInput("System Vision request failed: \(error.localizedDescription)")
        }

        return extractBoundingBoxes(from: animalReq, humanReq: humanReq)
    }
    #endif

    private func extractBoundingBoxes(
        from animalReq: VNRecognizeAnimalsRequest,
        humanReq: VNDetectHumanRectanglesRequest
    ) -> [BoundingBox] {
        var candidates: [BoundingBox] = []

        if let animalResults = animalReq.results {
            for obs in animalResults {
                let conf = Double(obs.labels.first?.confidence ?? obs.confidence)
                guard conf >= confidenceThreshold else { continue }

                let label = obs.labels.first?.identifier ?? "animal"
                // Apple Vision normalized coordinates have origin at bottom-left:
                // Convert to top-left image origin:
                let xMin = Double(obs.boundingBox.minX)
                let yMin = max(0.0, 1.0 - Double(obs.boundingBox.maxY))
                let xMax = Double(obs.boundingBox.maxX)
                let yMax = min(1.0, 1.0 - Double(obs.boundingBox.minY))

                candidates.append(BoundingBox(
                    xMin: xMin, yMin: yMin,
                    xMax: xMax, yMax: yMax,
                    confidence: conf,
                    classLabel: label
                ))
            }
        }

        if let humanResults = humanReq.results {
            for obs in humanResults {
                let conf = Double(obs.confidence)
                guard conf >= confidenceThreshold else { continue }

                let xMin = Double(obs.boundingBox.minX)
                let yMin = max(0.0, 1.0 - Double(obs.boundingBox.maxY))
                let xMax = Double(obs.boundingBox.maxX)
                let yMax = min(1.0, 1.0 - Double(obs.boundingBox.minY))

                candidates.append(BoundingBox(
                    xMin: xMin, yMin: yMin,
                    xMax: xMax, yMax: yMax,
                    confidence: conf,
                    classLabel: "person"
                ))
            }
        }

        // Apply Non-Maximum Suppression
        return nonMaximumSuppression(boxes: candidates, iouThreshold: iouThreshold)
    }

    private func nonMaximumSuppression(boxes: [BoundingBox], iouThreshold: Double) -> [BoundingBox] {
        let sorted = boxes.sorted { $0.confidence > $1.confidence }
        var selected: [BoundingBox] = []

        for box in sorted {
            var keep = true
            for prev in selected where prev.classLabel == box.classLabel {
                if box.iou(with: prev) > iouThreshold {
                    keep = false
                    break
                }
            }
            if keep {
                selected.append(box)
            }
        }
        return selected
    }
    #endif
}

// MARK: - YOLOv8Detector File Loading Extensions

extension YOLOv8Detector {
    /// Detects object bounding boxes from an image file at a local filesystem URL.
    public func detect(imageURL: URL) async throws -> [BoundingBox] {
        let dataset = try ImageDataset.load(from: imageURL)
        return try await detect(image: dataset)
    }

    /// Detects object bounding boxes from raw encoded image byte data.
    public func detect(imageData: Data) async throws -> [BoundingBox] {
        let dataset = try ImageDataset.load(from: imageData)
        return try await detect(image: dataset)
    }
}
