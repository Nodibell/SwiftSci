import Foundation
import Accelerate

/// Bounding box representation for object detection.
public struct BoundingBox: Sendable, Codable, Equatable {
    public let xMin: Double
    public let yMin: Double
    public let xMax: Double
    public let yMax: Double
    public let confidence: Double
    public let classLabel: String

    public init(xMin: Double, yMin: Double, xMax: Double, yMax: Double, confidence: Double, classLabel: String) {
        self.xMin = xMin
        self.yMin = yMin
        self.xMax = xMax
        self.yMax = yMax
        self.confidence = confidence
        self.classLabel = classLabel
    }

    /// Computes Intersection over Union (IoU) with another bounding box.
    public func iou(with other: BoundingBox) -> Double {
        let interXMin = max(self.xMin, other.xMin)
        let interYMin = max(self.yMin, other.yMin)
        let interXMax = min(self.xMax, other.xMax)
        let interYMax = min(self.yMax, other.yMax)

        let interWidth = max(0.0, interXMax - interXMin)
        let interHeight = max(0.0, interYMax - interYMin)
        let interArea = interWidth * interHeight

        let areaA = (self.xMax - self.xMin) * (self.yMax - self.yMin)
        let areaB = (other.xMax - other.xMin) * (other.yMax - other.yMin)
        let unionArea = areaA + areaB - interArea

        guard unionArea > 0 else { return 0.0 }
        return interArea / unionArea
    }
}

/// Evaluation metrics for computer vision tasks.
public enum VisionMetrics {
    /// Calculates the Dice Coefficient between binary masks.
    public static func diceCoefficient(predicted: [[Double]], groundTruth: [[Double]]) -> Double {
        guard !predicted.isEmpty, predicted.count == groundTruth.count else { return 0.0 }
        var intersection = 0.0
        var totalPred = 0.0
        var totalTrue = 0.0

        for r in 0..<predicted.count {
            for c in 0..<predicted[r].count {
                let p = predicted[r][c] > 0.5 ? 1.0 : 0.0
                let t = groundTruth[r][c] > 0.5 ? 1.0 : 0.0
                intersection += p * t
                totalPred += p
                totalTrue += t
            }
        }

        let denominator = totalPred + totalTrue
        guard denominator > 0 else { return 1.0 }
        return (2.0 * intersection) / denominator
    }

    /// Calculates Intersection over Union (IoU) score.
    public static func iouScore(predicted: [[Double]], groundTruth: [[Double]]) -> Double {
        guard !predicted.isEmpty, predicted.count == groundTruth.count else { return 0.0 }
        var intersection = 0.0
        var union = 0.0

        for r in 0..<predicted.count {
            for c in 0..<predicted[r].count {
                let p = predicted[r][c] > 0.5 ? 1.0 : 0.0
                let t = groundTruth[r][c] > 0.5 ? 1.0 : 0.0
                if p == 1.0 || t == 1.0 {
                    union += 1.0
                    if p == 1.0 && t == 1.0 {
                        intersection += 1.0
                    }
                }
            }
        }

        guard union > 0 else { return 1.0 }
        return intersection / union
    }
}

/// Simple image dataset container supporting array representation.
public struct ImageDataset: Sendable {
    public let width: Int
    public let height: Int
    public let channels: Int
    public let data: [Double]

    public init(width: Int, height: Int, channels: Int, data: [Double]) {
        self.width = width
        self.height = height
        self.channels = channels
        self.data = data
    }
}

/// Lightweight CNN Feature Extractor.
public struct CNNFeatureExtractor: Sendable {
    public init() {}

    /// Extracts global average pooling features from flattened image array.
    public func extractFeatures(image: ImageDataset) -> [Double] {
        let pixelCount = image.width * image.height
        guard pixelCount > 0 else { return [] }
        var channelMeans = [Double](repeating: 0.0, count: image.channels)

        for c in 0..<image.channels {
            var sum = 0.0
            for p in 0..<pixelCount {
                let idx = c * pixelCount + p
                if idx < image.data.count {
                    sum += image.data[idx]
                }
            }
            channelMeans[c] = sum / Double(pixelCount)
        }
        return channelMeans
    }
}

/// Errors specific to Computer Vision tasks.
public enum VisionError: Error, LocalizedError, Equatable {
    case notImplemented(String)
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .notImplemented(let msg):
            return "Vision feature not implemented: \(msg)"
        case .invalidInput(let msg):
            return "Invalid vision input: \(msg)"
        }
    }
}

/// Simple U-Net Segmentation model implementation.
public actor UNetSegmentationModel {
    public let inputChannels: Int
    public let numClasses: Int

    public init(inputChannels: Int = 3, numClasses: Int = 2) {
        self.inputChannels = inputChannels
        self.numClasses = numClasses
    }

    public func predict(image: ImageDataset) async throws -> [[Double]] {
        throw VisionError.notImplemented("UNetSegmentationModel: real deep learning weight inference is not yet wired — see ROADMAP v2.3")
    }
}

/// YOLOv8 object detector wrapper.
public actor YOLOv8Detector {
    public let confidenceThreshold: Double
    public let iouThreshold: Double

    public init(confidenceThreshold: Double = 0.25, iouThreshold: Double = 0.45) {
        self.confidenceThreshold = confidenceThreshold
        self.iouThreshold = iouThreshold
    }

    public func detect(image: ImageDataset) async throws -> [BoundingBox] {
        throw VisionError.notImplemented("YOLOv8Detector: real deep learning weight inference is not yet wired — see ROADMAP v2.3")
    }

    public func nonMaximumSuppression(boxes: [BoundingBox]) -> [BoundingBox] {
        let sorted = boxes.sorted { $0.confidence > $1.confidence }
        var selected: [BoundingBox] = []

        for box in sorted {
            var keep = true
            for prev in selected {
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
}
