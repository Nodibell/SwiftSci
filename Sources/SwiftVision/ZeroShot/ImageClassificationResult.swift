import Foundation

/// Represents a single predicted class label and its associated confidence score from zero-shot inference.
public struct ImageClassificationResult: Sendable, Codable, Equatable {
    /// The class identifier or category name (e.g. "outdoor", "golden_retriever", "sports_car").
    public let identifier: String
    /// The prediction confidence score, in the range [0.0, 1.0].
    public let confidence: Float

    /// Creates a new image classification result.
    /// - Parameters:
    ///   - identifier: The class identifier or label.
    ///   - confidence: Confidence score between 0.0 and 1.0.
    public init(identifier: String, confidence: Float) {
        self.identifier = identifier
        self.confidence = confidence
    }
}

/// Comprehensive structural and semantic profile of an unannotated image dataset directory.
///
/// Designed to eliminate manual folder inspection and synthetic benchmarking by extracting
/// honest dataset metrics (resolution, channel depths, format breakdowns, folder categories)
/// and running zero-shot pre-trained inference across sample images.
public struct ImageFolderProfile: Sendable, Codable, Equatable {
    /// The root directory URL that was analyzed.
    public let folderURL: URL
    /// Total number of recognized image files discovered.
    public let totalImages: Int
    /// Distribution of images by file format extension (e.g. `["jpg": 80, "png": 20]`).
    public let formatCounts: [String: Int]
    /// Breakdown of image counts by subfolder names if structured as class categories.
    public let subfolderClasses: [String: Int]
    /// Average image width in pixels across inspected sample images.
    public let averageWidth: Double
    /// Average image height in pixels across inspected sample images.
    public let averageHeight: Double
    /// Minimum discovered resolution `[width, height]`.
    public let minResolution: [Int]
    /// Maximum discovered resolution `[width, height]`.
    public let maxResolution: [Int]
    /// Detected or estimated color channel depth (e.g. 1 for Grayscale, 3 for RGB, 4 for RGBA).
    public let channelDepth: Int
    /// Top detected semantic categories aggregated across the sampled images.
    public let topDetectedLabels: [ImageClassificationResult]
    /// Sample file predictions mapping relative file paths to their top zero-shot classifications.
    public let samplePredictions: [String: [ImageClassificationResult]]
    /// Formatted textual summary ready for reports, console logs, or UI display.
    public let summary: String

    /// Initializes a new image folder profile.
    public init(
        folderURL: URL,
        totalImages: Int,
        formatCounts: [String: Int],
        subfolderClasses: [String: Int],
        averageWidth: Double,
        averageHeight: Double,
        minResolution: [Int],
        maxResolution: [Int],
        channelDepth: Int,
        topDetectedLabels: [ImageClassificationResult],
        samplePredictions: [String: [ImageClassificationResult]],
        summary: String
    ) {
        self.folderURL = folderURL
        self.totalImages = totalImages
        self.formatCounts = formatCounts
        self.subfolderClasses = subfolderClasses
        self.averageWidth = averageWidth
        self.averageHeight = averageHeight
        self.minResolution = minResolution
        self.maxResolution = maxResolution
        self.channelDepth = channelDepth
        self.topDetectedLabels = topDetectedLabels
        self.samplePredictions = samplePredictions
        self.summary = summary
    }
}
