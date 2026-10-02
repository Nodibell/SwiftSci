import Testing
import Foundation
@testable import SwiftVision

@Suite("Zero-Shot Visual Inference & Dataset Profiling Tests (G-021)")
struct ZeroShotTests {

    @Test("ImageDataset CoreGraphics bridging, PNG serialization, and round-trip loading")
    func testImageDatasetSerialization() throws {
        let width = 16
        let height = 16
        let pixelCount = width * height
        var data = [Double](repeating: 0.0, count: pixelCount * 3)
        // Red channel = 0.9, Green = 0.2, Blue = 0.1
        for p in 0..<pixelCount {
            data[p] = 0.9
            data[pixelCount + p] = 0.2
            data[2 * pixelCount + p] = 0.1
        }

        let dataset = ImageDataset(width: width, height: height, channels: 3, data: data)

        #if canImport(CoreGraphics)
        let cgImage = dataset.toCGImage()
        #expect(cgImage != nil)
        #expect(cgImage?.width == 16)
        #expect(cgImage?.height == 16)
        #endif

        #if canImport(CoreGraphics) && canImport(ImageIO)
        let pngData = dataset.toPNGData()
        #expect(pngData != nil)
        #expect((pngData?.count ?? 0) > 50)

        if let pngData = pngData {
            let reloaded = try ImageDataset.load(from: pngData)
            #expect(reloaded.width == 16)
            #expect(reloaded.height == 16)
            #expect(reloaded.channels == 3)
            #expect(reloaded.data.count == pixelCount * 3)
            // Verify red channel dominance preserved
            #expect(reloaded.data[0] > 0.8)
        }
        #endif
    }

    @Test("ZeroShotImageClassifier inference on in-memory ImageDataset")
    func testZeroShotClassification() {
        let classifier = ZeroShotImageClassifier()
        let width = 32
        let height = 32
        let pixelCount = width * height
        let dataset = ImageDataset(width: width, height: height, channels: 3, data: [Double](repeating: 0.5, count: pixelCount * 3))

        // Open-ended taxonomy classification
        let openResults = classifier.classify(dataset: dataset, maxResults: 3)
        #expect(!openResults.isEmpty)
        #expect(openResults.count <= 3)
        #expect(openResults.first?.confidence ?? 0.0 >= 0.0)

        // Constrained candidate label zero-shot scoring
        let candidates = ["outdoor", "indoor", "animal"]
        let candidateResults = classifier.classify(dataset: dataset, candidateLabels: candidates, maxResults: 3)
        #expect(candidateResults.count == 3)
        let totalConfidence = candidateResults.reduce(0.0) { $0 + Double($1.confidence) }
        #expect(abs(totalConfidence - 1.0) < 0.01)
    }

    @Test("ZeroShotImageClassifier feature print extraction and cosine distance")
    func testFeaturePrintAndCosineDistance() {
        let classifier = ZeroShotImageClassifier()
        let width = 32
        let height = 32
        let pixelCount = width * height

        let datasetA = ImageDataset(width: width, height: height, channels: 3, data: [Double](repeating: 0.8, count: pixelCount * 3))
        let datasetB = ImageDataset(width: width, height: height, channels: 3, data: [Double](repeating: 0.8, count: pixelCount * 3))
        let datasetC = ImageDataset(width: width, height: height, channels: 3, data: [Double](repeating: 0.1, count: pixelCount * 3))

        let featA = classifier.extractFeaturePrint(dataset: datasetA)
        let featB = classifier.extractFeaturePrint(dataset: datasetB)
        let featC = classifier.extractFeaturePrint(dataset: datasetC)

        #expect(!featA.isEmpty)
        #expect(featA.count == featB.count)

        // Distance between identical images should be ~0.0
        let distIdentical = classifier.computeCosineDistance(embeddingA: featA, embeddingB: featB)
        #expect(distIdentical < 1e-4)

        // Distance between different images should be non-negative
        let distDifferent = classifier.computeCosineDistance(embeddingA: featA, embeddingB: featC)
        #expect(distDifferent >= 0.0)
    }

    @Test("Automated image folder profiling for unannotated datasets")
    func testImageFolderProfiling() async throws {
        let fileManager = FileManager.default
        let tempDir = fileManager.temporaryDirectory.appendingPathComponent("swiftsci_vision_test_\(UUID().uuidString)")
        let catsDir = tempDir.appendingPathComponent("cats")
        let dogsDir = tempDir.appendingPathComponent("dogs")

        try fileManager.createDirectory(at: catsDir, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: dogsDir, withIntermediateDirectories: true)

        defer {
            try? fileManager.removeItem(at: tempDir)
        }

        // Generate sample test images
        let img1 = ImageDataset(width: 16, height: 16, channels: 3, data: [Double](repeating: 0.9, count: 16 * 16 * 3))
        let img2 = ImageDataset(width: 20, height: 20, channels: 3, data: [Double](repeating: 0.7, count: 20 * 20 * 3))
        let img3 = ImageDataset(width: 24, height: 24, channels: 3, data: [Double](repeating: 0.3, count: 24 * 24 * 3))

        #if canImport(CoreGraphics) && canImport(ImageIO)
        if let png1 = img1.toPNGData(), let png2 = img2.toPNGData(), let png3 = img3.toPNGData() {
            try png1.write(to: catsDir.appendingPathComponent("cat1.png"))
            try png2.write(to: catsDir.appendingPathComponent("cat2.png"))
            try png3.write(to: dogsDir.appendingPathComponent("dog1.png"))
        }
        #endif

        let classifier = ZeroShotImageClassifier()
        let profile = try await classifier.profileFolder(
            at: tempDir,
            maxSampleImages: 10,
            candidateLabels: ["animal", "landscape"]
        )

        #expect(profile.totalImages == 3)
        #expect(profile.formatCounts["png"] == 3)
        #expect(profile.subfolderClasses["cats"] == 2)
        #expect(profile.subfolderClasses["dogs"] == 1)
        #expect(profile.averageWidth > 15.0)
        #expect(profile.averageHeight > 15.0)
        #expect(profile.minResolution == [16, 16])
        #expect(profile.maxResolution == [24, 24])
        #expect(!profile.summary.isEmpty)
        #expect(profile.summary.contains("cats"))
        #expect(profile.summary.contains("dogs"))
        #expect(profile.samplePredictions.count == 3)
    }
}
