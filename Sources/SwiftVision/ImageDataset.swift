import Foundation
import MLX
import Accelerate
import SwiftML


/// Bounding box representation for object detection.
public struct BoundingBox: Sendable, Codable, Equatable {
    /// The x min.
    public let xMin: Double
    /// The y min.
    public let yMin: Double
    /// The x max.
    public let xMax: Double
    /// The y max.
    public let yMax: Double
    /// The confidence.
    public let confidence: Double
    /// The class label.
    public let classLabel: String

    /// Creates a new instance.
    /// - Parameters:
    ///   - xMin: The x min.
    ///   - yMin: The y min.
    ///   - xMax: The x max.
    ///   - yMax: The y max.
    ///   - confidence: The confidence.
    ///   - classLabel: The class label.
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
    /// The width.
    public let width: Int
    /// The height.
    public let height: Int
    /// The channels.
    public let channels: Int
    /// The data.
    public let data: [Double]

    /// Creates a new instance.
    /// - Parameters:
    ///   - width: The width.
    ///   - height: The height.
    ///   - channels: The channels.
    ///   - data: The data.
    public init(width: Int, height: Int, channels: Int, data: [Double]) {
        self.width = width
        self.height = height
        self.channels = channels
        self.data = data
    }
}

/// Lightweight CNN Feature Extractor.
public struct CNNFeatureExtractor: Sendable {
    /// Creates a new instance.
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

    /// The error description.
    public var errorDescription: String? {
        switch self {
        case .notImplemented(let msg):
            return "Vision feature not implemented: \(msg)"
        case .invalidInput(let msg):
            return "Invalid vision input: \(msg)"
        }
    }
}

/// Deep Convolutional U-Net semantic segmentation model executing real neural network forward passes.
public actor UNetSegmentationModel {
    /// The input channels.
    public let inputChannels: Int
    /// The number of output segmentation classes.
    public let numClasses: Int
    /// The deep U-Net convolutional neural network.
    private let unet: UNetArchitecture

    /// Creates a new UNetSegmentationModel instance.
    /// - Parameters:
    ///   - inputChannels: Number of image channels (default 3).
    ///   - numClasses: Number of output segmentation classes (default 2).
    ///   - baseChannels: Number of base convolutional filter channels (default 16).
    public init(inputChannels: Int = 3, numClasses: Int = 2, baseChannels: Int = 16) {
        self.inputChannels = inputChannels
        self.numClasses = numClasses
        self.unet = UNetArchitecture(inChannels: inputChannels, numClasses: numClasses, baseChannels: baseChannels)
    }

    /// Predicts 2D class segmentation probability mask from image dataset using U-Net neural network forward pass.
    /// - Parameter image: The input image dataset.
    /// - Throws: `VisionError.invalidInput` if image dimensions are non-positive.
    /// - Returns: A 2D array of class segmentation probability masks [Height, Width].
    public func predict(image: ImageDataset) async throws -> [[Double]] {
        guard image.width > 0, image.height > 0 else {
            throw VisionError.invalidInput("Image dimensions must be positive")
        }
        let h = image.height
        let w = image.width
        let c = min(image.channels, inputChannels)

        // Convert [H * W * C] flat image data into NHWC MLXArray
        var floatData = [Float](repeating: 0.0, count: h * w * inputChannels)
        let pixelCount = h * w
        for p in 0..<pixelCount {
            for ch in 0..<c {
                let srcIdx = ch * pixelCount + p
                let dstIdx = p * inputChannels + ch
                if srcIdx < image.data.count {
                    floatData[dstIdx] = Float(image.data[srcIdx])
                }
            }
        }

        let inputTensor = MLXArray(floatData).reshaped([1, h, w, inputChannels])
        let logits = unet(inputTensor)
        eval(logits)

        // Extract class 1 (foreground) or binary sigmoid probabilities
        let flatLogits = logits[0].asArray(Float.self)
        var mask = [[Double]](repeating: [Double](repeating: 0.0, count: w), count: h)

        for r in 0..<h {
            for col in 0..<w {
                let pixelIdx = (r * w + col) * numClasses
                let logitVal = numClasses > 1 ? flatLogits[pixelIdx + 1] : flatLogits[pixelIdx]
                let prob = Double(1.0 / (1.0 + exp(-logitVal)))
                mask[r][col] = prob
            }
        }

        return mask
    }
}

/// Real YOLOv8n object detector executing deep neural network forward pass on MLX/MLXNN.
public actor YOLOv8Detector {
    /// The confidence threshold.
    public let confidenceThreshold: Double
    /// The iou threshold.
    public let iouThreshold: Double

    private let backbone: YOLOBackbone
    private let neck: YOLONeck
    private let head: YOLOHead
    private let preprocessor: YOLOPreprocessor
    /// Category class label names for predictions.
    public let classLabels: [String]

    /// Creates a new YOLOv8 Detector instance.
    /// - Parameters:
    ///   - confidenceThreshold: Confidence threshold filter.
    ///   - iouThreshold: Non-Maximum Suppression IoU threshold.
    ///   - classLabels: Optional custom class names (defaults to 80 COCO classes).
    public init(
        confidenceThreshold: Double = 0.25,
        iouThreshold: Double = 0.45,
        classLabels: [String]? = nil
    ) {
        self.confidenceThreshold = confidenceThreshold
        self.iouThreshold = iouThreshold
        self.backbone = YOLOBackbone()
        self.neck = YOLONeck()
        self.head = YOLOHead(numClasses: classLabels?.count ?? 80)
        self.preprocessor = YOLOPreprocessor(targetWidth: 640, targetHeight: 640)
        self.classLabels = classLabels ?? [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        ]
    }

    /// Binds pre-trained model weights into the neural network layers.
    /// - Parameter loader: Mapped tensor weight loader.
    public func loadWeights(_ loader: YOLOWeightLoader) {
        backbone.loadWeights(from: loader, prefix: "model")
        neck.loadWeights(from: loader, prefix: "model")
        head.loadWeights(from: loader, prefix: "model.22")
    }

    /// Detects object bounding boxes in an input image dataset using real neural network forward pass.
    /// - Parameters:
    ///   - image: The input image dataset.
    /// - Throws: An error if image dimensions are invalid.
    /// - Returns: A list of detected bounding boxes filtered by NMS.
    public func detect(image: ImageDataset) async throws -> [BoundingBox] {
        guard image.width > 0, image.height > 0 else {
            throw VisionError.invalidInput("Image dimensions must be positive")
        }

        let inputTensor = preprocessor.preprocess(image: image)
        let backboneOut = backbone(inputTensor)
        let neckOut = neck(backboneOut)
        let headOut = head(neckOut)

        eval(headOut.boxes, headOut.scores)

        let rawBoxes = headOut.boxes[0].asArray(Float.self)
        let rawScores = headOut.scores[0].asArray(Float.self)
        let numAnchors = 8400
        let numCls = head.numClasses

        var candidateBoxes: [BoundingBox] = []

        for a in 0..<numAnchors {
            let scoreOffset = a * numCls
            var maxScore: Float = 0.0
            var maxClass = 0
            for c in 0..<numCls {
                let s = rawScores[scoreOffset + c]
                if s > maxScore {
                    maxScore = s
                    maxClass = c
                }
            }

            if Double(maxScore) >= confidenceThreshold {
                let boxOffset = a * 4
                let xMin = Double(rawBoxes[boxOffset + 0])
                let yMin = Double(rawBoxes[boxOffset + 1])
                let xMax = Double(rawBoxes[boxOffset + 2])
                let yMax = Double(rawBoxes[boxOffset + 3])
                let label = maxClass < classLabels.count ? classLabels[maxClass] : "object_\(maxClass)"

                candidateBoxes.append(BoundingBox(
                    xMin: xMin, yMin: yMin,
                    xMax: xMax, yMax: yMax,
                    confidence: Double(maxScore),
                    classLabel: label
                ))
            }
        }

        return nonMaximumSuppression(boxes: candidateBoxes)
    }

    /// Non maximum suppression.
    /// - Parameters:
    ///   - boxes: The boxes.
    /// - Returns: A `[BoundingBox]` result.
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
