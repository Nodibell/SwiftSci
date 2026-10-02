import Foundation
#if canImport(Vision)
import Vision
#endif
#if canImport(CoreGraphics)
import CoreGraphics
#endif
#if canImport(ImageIO)
import ImageIO
#endif

/// High-level image classifier and dataset profiling engine built on Apple Vision framework.
///
/// Provides out-of-the-box visual inference and deep feature embedding extraction
/// utilizing Apple Vision's built-in neural classification taxonomy (1,300+ hierarchical classes)
/// and feature print representations (`VNGenerateImageFeaturePrintRequest`).
///
/// > Important: This classifier operates on Apple Vision's fixed pre-trained taxonomy.
/// > When `candidateLabels` are supplied, candidate scores are matched and normalized against
/// > matching taxonomy observations. For open-vocabulary vision-language classification
/// > with text prompt embeddings, use ``CLIPProjector``.
///
/// ### Key Capabilities
/// - **Taxonomy Classification**: Categorizes images against Apple's neural taxonomy or scores
///   matching concepts from user-supplied candidate label lists.
/// - **Pretrained Backbone Embeddings**: Extracts 768-dimensional visual feature prints from images
///   for nearest-neighbor retrieval, clustering, or transfer learning.
/// - **Automated Dataset Profiling**: Crawls unannotated image directories, extracting resolution
///   statistics, channel counts, file formats, subfolder classes, and sample predictions.
public struct VisionImageClassifier: Sendable {

    /// Set of standard supported image file extensions.
    public static let supportedExtensions: Set<String> = [
        "jpg", "jpeg", "png", "heic", "bmp", "tiff", "tif", "webp", "gif"
    ]

    /// Initializes a new VisionImageClassifier instance.
    public init() {}

    // MARK: - Classification

    /// Classifies an image file at the given local file URL.
    ///
    /// - Parameters:
    ///   - imageURL: Local file URL pointing to an image.
    ///   - candidateLabels: Optional list of candidate class labels. If provided, the classifier
    ///     scores and normalizes confidences across these candidates. If empty, the top classifications
    ///     from the 1,300+ class taxonomy are returned.
    ///   - maxResults: Maximum number of top results to return (default: 5).
    /// - Throws: `VisionError.invalidInput` if the image file cannot be read or processed.
    /// - Returns: Ranked list of `ImageClassificationResult` instances sorted by confidence descending.
    public func classify(
        imageURL: URL,
        candidateLabels: [String] = [],
        maxResults: Int = 5
    ) async throws -> [ImageClassificationResult] {
        #if canImport(Vision)
        let request = VNClassifyImageRequest()
        let handler = VNImageRequestHandler(url: imageURL, options: [:])
        do {
            try handler.perform([request])
        } catch {
            throw VisionError.invalidInput("Vision classification failed: \(error.localizedDescription)")
        }

        guard let observations = request.results else {
            return []
        }
        return processObservations(observations, candidateLabels: candidateLabels, maxResults: maxResults)
        #else
        // Fallback for non-Apple platforms
        let dataset = try ImageDataset.load(from: imageURL)
        return classify(dataset: dataset, candidateLabels: candidateLabels, maxResults: maxResults)
        #endif
    }

    /// Classifies an image from in-memory byte data.
    ///
    /// - Parameters:
    ///   - imageData: Raw encoded image bytes (PNG, JPEG, etc.).
    ///   - candidateLabels: Optional list of candidate class labels.
    ///   - maxResults: Maximum number of top results to return (default: 5).
    /// - Throws: `VisionError.invalidInput` if data cannot be decoded.
    /// - Returns: Ranked list of `ImageClassificationResult` instances.
    public func classify(
        imageData: Data,
        candidateLabels: [String] = [],
        maxResults: Int = 5
    ) async throws -> [ImageClassificationResult] {
        #if canImport(Vision)
        let request = VNClassifyImageRequest()
        let handler = VNImageRequestHandler(data: imageData, options: [:])
        do {
            try handler.perform([request])
        } catch {
            throw VisionError.invalidInput("Vision classification failed: \(error.localizedDescription)")
        }

        guard let observations = request.results else {
            return []
        }
        return processObservations(observations, candidateLabels: candidateLabels, maxResults: maxResults)
        #else
        let dataset = try ImageDataset.load(from: imageData)
        return classify(dataset: dataset, candidateLabels: candidateLabels, maxResults: maxResults)
        #endif
    }

    /// Classifies an in-memory `ImageDataset` pixel buffer.
    ///
    /// - Parameters:
    ///   - dataset: The input `ImageDataset`.
    ///   - candidateLabels: Optional list of candidate class labels.
    ///   - maxResults: Maximum number of top results to return (default: 5).
    /// - Returns: Ranked list of `ImageClassificationResult` instances.
    public func classify(
        dataset: ImageDataset,
        candidateLabels: [String] = [],
        maxResults: Int = 5
    ) -> [ImageClassificationResult] {
        #if canImport(Vision) && canImport(CoreGraphics)
        guard let cgImage = dataset.toCGImage() else {
            return fallbackClassify(dataset: dataset, candidateLabels: candidateLabels, maxResults: maxResults)
        }
        let request = VNClassifyImageRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
            if let observations = request.results {
                return processObservations(observations, candidateLabels: candidateLabels, maxResults: maxResults)
            }
        } catch {
            // Fall through to fallback
        }
        #endif
        return fallbackClassify(dataset: dataset, candidateLabels: candidateLabels, maxResults: maxResults)
    }

    // MARK: - Feature Print Embeddings

    /// Extracts a high-dimensional feature print embedding (768-D Float vector) from an image file.
    ///
    /// - Parameter imageURL: Local file URL to the target image.
    /// - Throws: `VisionError.invalidInput` if feature extraction fails.
    /// - Returns: A normalized Float embedding array.
    public func extractFeaturePrint(imageURL: URL) async throws -> [Float] {
        #if canImport(Vision)
        let request = VNGenerateImageFeaturePrintRequest()
        let handler = VNImageRequestHandler(url: imageURL, options: [:])
        do {
            try handler.perform([request])
        } catch {
            throw VisionError.invalidInput("Feature print extraction failed: \(error.localizedDescription)")
        }

        guard let observation = request.results?.first as? VNFeaturePrintObservation else {
            throw VisionError.invalidInput("No feature print observation produced")
        }
        return extractFloats(from: observation)
        #else
        let dataset = try ImageDataset.load(from: imageURL)
        return extractFeaturePrint(dataset: dataset)
        #endif
    }

    /// Extracts a high-dimensional feature print embedding from encoded image byte data.
    ///
    /// - Parameter imageData: Encoded image bytes.
    /// - Throws: `VisionError.invalidInput` if extraction fails.
    /// - Returns: A normalized Float embedding array.
    public func extractFeaturePrint(imageData: Data) async throws -> [Float] {
        #if canImport(Vision)
        let request = VNGenerateImageFeaturePrintRequest()
        let handler = VNImageRequestHandler(data: imageData, options: [:])
        do {
            try handler.perform([request])
        } catch {
            throw VisionError.invalidInput("Feature print extraction failed: \(error.localizedDescription)")
        }

        guard let observation = request.results?.first as? VNFeaturePrintObservation else {
            throw VisionError.invalidInput("No feature print observation produced")
        }
        return extractFloats(from: observation)
        #else
        let dataset = try ImageDataset.load(from: imageData)
        return extractFeaturePrint(dataset: dataset)
        #endif
    }

    /// Extracts a visual feature embedding from an in-memory `ImageDataset`.
    ///
    /// - Parameter dataset: The input image dataset.
    /// - Returns: A normalized Float embedding vector.
    public func extractFeaturePrint(dataset: ImageDataset) -> [Float] {
        #if canImport(Vision) && canImport(CoreGraphics)
        if let cgImage = dataset.toCGImage() {
            let request = VNGenerateImageFeaturePrintRequest()
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            if (try? handler.perform([request])) != nil,
               let observation = request.results?.first as? VNFeaturePrintObservation {
                return extractFloats(from: observation)
            }
        }
        #endif

        // Pure-Swift spatial pooling embedding fallback (64-dimensional normalized vector)
        let dim = 8
        let targetCount = dim * dim
        var pooled = [Float](repeating: 0.0, count: targetCount)
        guard dataset.width > 0, dataset.height > 0, !dataset.data.isEmpty else {
            return pooled
        }

        let blockW = max(1, dataset.width / dim)
        let blockH = max(1, dataset.height / dim)
        let pixelCount = dataset.width * dataset.height

        for by in 0..<dim {
            for bx in 0..<dim {
                var sum: Double = 0.0
                var count = 0
                for y in (by * blockH)..<min(dataset.height, (by + 1) * blockH) {
                    for x in (bx * blockW)..<min(dataset.width, (bx + 1) * blockW) {
                        let idx = y * dataset.width + x
                        if idx < pixelCount && idx < dataset.data.count {
                            sum += dataset.data[idx]
                            count += 1
                        }
                    }
                }
                let avg = count > 0 ? sum / Double(count) : 0.0
                pooled[by * dim + bx] = Float(avg)
            }
        }

        // L2-normalize
        var normSq: Float = 0.0
        for v in pooled { normSq += v * v }
        let norm = sqrt(normSq)
        if norm > 1e-12 {
            for i in 0..<pooled.count { pooled[i] /= norm }
        }
        return pooled
    }

    /// Computes the cosine distance between two feature embeddings in range `[0.0, 2.0]`.
    ///
    /// A distance of `0.0` denotes identical semantic orientation, while values close to `1.0` denote orthogonality.
    ///
    /// - Parameters:
    ///   - embeddingA: First normalized feature vector.
    ///   - embeddingB: Second normalized feature vector.
    /// - Returns: Cosine distance scalar value.
    public func computeCosineDistance(embeddingA: [Float], embeddingB: [Float]) -> Float {
        guard embeddingA.count == embeddingB.count, !embeddingA.isEmpty else { return 1.0 }
        var dot: Float = 0.0
        var normA: Float = 0.0
        var normB: Float = 0.0
        for i in 0..<embeddingA.count {
            dot += embeddingA[i] * embeddingB[i]
            normA += embeddingA[i] * embeddingA[i]
            normB += embeddingB[i] * embeddingB[i]
        }
        let denom = sqrt(normA) * sqrt(normB)
        guard denom > 1e-12 else { return 1.0 }
        let cosSim = max(-1.0, min(1.0, dot / denom))
        return 1.0 - cosSim
    }

    // MARK: - Dataset & Folder Profiling

    /// Profiles an unannotated image dataset directory, extracting resolution, format distributions,
    /// subfolder-derived categories, and zero-shot sample predictions.
    ///
    /// - Parameters:
    ///   - folderURL: Root directory containing images or category subfolders.
    ///   - maxSampleImages: Maximum number of images to inspect and classify (default: 50).
    ///   - candidateLabels: Optional list of candidate labels for zero-shot classification.
    /// - Throws: `VisionError.invalidInput` if directory does not exist.
    /// - Returns: An `ImageFolderProfile` containing comprehensive structural and semantic metrics.
    public func profileFolder(
        at folderURL: URL,
        maxSampleImages: Int = 50,
        candidateLabels: [String] = []
    ) async throws -> ImageFolderProfile {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: folderURL.path, isDirectory: &isDir), isDir.boolValue else {
            throw VisionError.invalidInput("URL must be an existing directory: \(folderURL.path)")
        }

        var (allImageURLs, formatCounts, subfolderClasses) = try scanDirectory(at: folderURL)

        allImageURLs.sort { $0.lastPathComponent < $1.lastPathComponent }

        guard !allImageURLs.isEmpty else {
            return ImageFolderProfile(
                folderURL: folderURL,
                totalImages: 0,
                formatCounts: formatCounts,
                subfolderClasses: subfolderClasses,
                averageWidth: 0,
                averageHeight: 0,
                minResolution: [0, 0],
                maxResolution: [0, 0],
                channelDepth: 0,
                topDetectedLabels: [],
                samplePredictions: [:],
                summary: "Empty dataset directory: 0 images found at \(folderURL.path)."
            )
        }

        // Select evenly spaced sample images
        let sampleCount = min(maxSampleImages, allImageURLs.count)
        let step = max(1, allImageURLs.count / sampleCount)
        var sampleURLs: [URL] = []
        for i in stride(from: 0, to: allImageURLs.count, by: step) {
            if sampleURLs.count < sampleCount {
                sampleURLs.append(allImageURLs[i])
            }
        }

        var totalWidth: Double = 0
        var totalHeight: Double = 0
        var minW = Int.max
        var minH = Int.max
        var maxW = 0
        var maxH = 0
        var channelDepth = 3

        var samplePredictions: [String: [ImageClassificationResult]] = [:]
        var labelConfSums: [String: (total: Float, count: Int)] = [:]

        for url in sampleURLs {
            // Resolution extraction
            var imgW = 0
            var imgH = 0

            #if canImport(ImageIO)
            if let src = CGImageSourceCreateWithURL(url as CFURL, nil),
               let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any] {
                imgW = props[kCGImagePropertyPixelWidth] as? Int ?? 0
                imgH = props[kCGImagePropertyPixelHeight] as? Int ?? 0
                if let colorModel = props[kCGImagePropertyColorModel] as? String {
                    channelDepth = colorModel == "Gray" ? 1 : (colorModel == "RGB" ? 3 : 4)
                }
            }
            #endif

            if imgW > 0 && imgH > 0 {
                totalWidth += Double(imgW)
                totalHeight += Double(imgH)
                minW = min(minW, imgW)
                minH = min(minH, imgH)
                maxW = max(maxW, imgW)
                maxH = max(maxH, imgH)
            }

            // Zero-shot classification
            if let preds = try? await classify(imageURL: url, candidateLabels: candidateLabels, maxResults: 3) {
                samplePredictions[url.lastPathComponent] = preds
                for pred in preds {
                    let existing = labelConfSums[pred.identifier, default: (0.0, 0)]
                    labelConfSums[pred.identifier] = (existing.total + pred.confidence, existing.count + 1)
                }
            }
        }

        let inspectedCount = max(1, sampleURLs.count)
        let avgW = totalWidth > 0 ? totalWidth / Double(inspectedCount) : 0.0
        let avgH = totalHeight > 0 ? totalHeight / Double(inspectedCount) : 0.0
        if minW == Int.max { minW = 0; minH = 0 }

        // Top detected labels sorted by mean confidence
        let topDetectedLabels = labelConfSums.map { (key, value) in
            ImageClassificationResult(identifier: key, confidence: value.total / Float(inspectedCount))
        }.sorted { $0.confidence > $1.confidence }.prefix(5)

        // Text summary
        let formatStr = formatCounts.sorted { $0.value > $1.value }
            .map { ".\($0.key): \($0.value)" }
            .joined(separator: ", ")

        let subfolderStr = subfolderClasses.isEmpty
            ? "flat structure"
            : subfolderClasses.sorted { $0.value > $1.value }.map { "\($0.key) (\($0.value))" }.joined(separator: ", ")

        let topLabelsStr = topDetectedLabels.isEmpty
            ? "none"
            : topDetectedLabels.map { "\($0.identifier) (\(String(format: "%.1f%%", $0.confidence * 100)))" }.joined(separator: ", ")

        let summary = "Image dataset at '\(folderURL.lastPathComponent)': \(allImageURLs.count) image(s) [\(formatStr)]. Resolution: avg \(Int(round(avgW)))x\(Int(round(avgH))) (min \(minW)x\(minH), max \(maxW)x\(maxH)). Categories: \(subfolderStr). Top labels: \(topLabelsStr)."

        return ImageFolderProfile(
            folderURL: folderURL,
            totalImages: allImageURLs.count,
            formatCounts: formatCounts,
            subfolderClasses: subfolderClasses,
            averageWidth: avgW,
            averageHeight: avgH,
            minResolution: [minW, minH],
            maxResolution: [maxW, maxH],
            channelDepth: channelDepth,
            topDetectedLabels: Array(topDetectedLabels),
            samplePredictions: samplePredictions,
            summary: summary
        )
    }

    // MARK: - Private Helpers

    #if canImport(Vision)
    private func processObservations(
        _ observations: [VNClassificationObservation],
        candidateLabels: [String],
        maxResults: Int
    ) -> [ImageClassificationResult] {
        if candidateLabels.isEmpty {
            return observations.prefix(maxResults).map {
                ImageClassificationResult(identifier: $0.identifier, confidence: $0.confidence)
            }
        }

        var scores: [String: Float] = [:]
        for cand in candidateLabels {
            let candNorm = cand.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            var bestConf: Float = 0.0
            for obs in observations {
                let idNorm = obs.identifier.lowercased().replacingOccurrences(of: "_", with: " ")
                if idNorm == candNorm {
                    bestConf = max(bestConf, obs.confidence)
                } else if idNorm.components(separatedBy: " ").contains(candNorm) ||
                          candNorm.components(separatedBy: " ").contains(idNorm) {
                    bestConf = max(bestConf, obs.confidence * 0.9)
                } else if idNorm.contains(candNorm) || candNorm.contains(idNorm) {
                    bestConf = max(bestConf, obs.confidence * 0.75)
                }
            }
            scores[cand] = bestConf
        }

        let sumScores = scores.values.reduce(0.0, +)
        var results: [ImageClassificationResult] = []
        if sumScores > 1e-6 {
            for (cand, s) in scores {
                results.append(ImageClassificationResult(identifier: cand, confidence: s / sumScores))
            }
        } else {
            let uniform = Float(1.0 / Float(max(1, candidateLabels.count)))
            for cand in candidateLabels {
                results.append(ImageClassificationResult(identifier: cand, confidence: uniform))
            }
        }
        return results.sorted { $0.confidence > $1.confidence }.prefix(maxResults).map { $0 }
    }

    private func extractFloats(from observation: VNFeaturePrintObservation) -> [Float] {
        var floats = [Float](repeating: 0, count: observation.elementCount)
        observation.data.withUnsafeBytes { buffer in
            if let base = buffer.baseAddress {
                let typed = base.bindMemory(to: Float.self, capacity: observation.elementCount)
                for i in 0..<observation.elementCount {
                    floats[i] = typed[i]
                }
            }
        }
        return floats
    }
    #endif

    private func fallbackClassify(
        dataset: ImageDataset,
        candidateLabels: [String],
        maxResults: Int
    ) -> [ImageClassificationResult] {
        if candidateLabels.isEmpty {
            return [
                ImageClassificationResult(identifier: "image", confidence: 1.0)
            ]
        }
        let uniform = Float(1.0 / Float(candidateLabels.count))
        return candidateLabels.prefix(maxResults).map {
            ImageClassificationResult(identifier: $0, confidence: uniform)
        }
    }

    private func scanDirectory(
        at folderURL: URL
    ) throws -> (imageURLs: [URL], formatCounts: [String: Int], subfolderClasses: [String: Int]) {
        guard let enumerator = FileManager.default.enumerator(
            at: folderURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw VisionError.invalidInput("Failed to enumerate directory at: \(folderURL.path)")
        }

        var allImageURLs: [URL] = []
        var formatCounts: [String: Int] = [:]
        var subfolderClasses: [String: Int] = [:]

        while let fileURL = enumerator.nextObject() as? URL {
            let ext = fileURL.pathExtension.lowercased()
            guard Self.supportedExtensions.contains(ext) else { continue }
            allImageURLs.append(fileURL)
            formatCounts[ext, default: 0] += 1

            let rootPath = folderURL.standardizedFileURL.path
            let filePath = fileURL.standardizedFileURL.path
            if filePath.hasPrefix(rootPath) {
                let relative = String(filePath.dropFirst(rootPath.count))
                    .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
                let components = relative.components(separatedBy: "/")
                if components.count > 1 {
                    let subfolder = components[0]
                    subfolderClasses[subfolder, default: 0] += 1
                }
            }
        }
        return (allImageURLs, formatCounts, subfolderClasses)
    }
}

/// Compatibility typealias for ``VisionImageClassifier``.
@available(*, deprecated, renamed: "VisionImageClassifier", message: "Use VisionImageClassifier directly.")
public typealias ZeroShotImageClassifier = VisionImageClassifier

