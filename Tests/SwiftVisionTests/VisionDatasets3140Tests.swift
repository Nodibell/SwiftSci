import Testing
import Foundation
#if canImport(CoreGraphics)
import CoreGraphics
#endif
@testable import SwiftVision

@Suite("SwiftVision 3.14.0 Dataset Metadata, Cropping & Split ImageFolder Tests (G-039)")
struct VisionDatasets3140Tests {

    // MARK: - YOLODatasetMetadata & YOLOAnnotationParser Tests

    @Test("YOLODatasetMetadata parses dict format YAML with numeric sorting, comments, and quotes")
    func testYOLOMetadataDictFormat() throws {
        let yaml = """
        # Ultralytics dataset config
        path: /datasets/road # dataset root
        train: images/train # train images
        val: 'images/val'
        test: "images/test"
        nc: 4 # number of classes

        names:
          0: car # passenger vehicle
          10: 'traffic #10'
          2: "bicycle"
          1: pedestrian
        """

        let metadata = try #require(YOLODatasetMetadata.parseYAMLContent(yaml))
        #expect(metadata.trainPath == "images/train")
        #expect(metadata.valPath == "images/val")
        #expect(metadata.testPath == "images/test")
        #expect(metadata.nc == 4)
        #expect(metadata.classNames == ["car", "pedestrian", "bicycle", "traffic #10"])
    }

    @Test("YOLODatasetMetadata parses block list, inline list, and valid alias")
    func testYOLOMetadataListFormatsAndValidAlias() throws {
        let blockYAML = """
        train: train/images
        valid: valid/images
        nc: 3
        names:
          - 'cat' # feline
          - "dog"
          - bird
        """
        let blockMeta = try #require(YOLODatasetMetadata.parseYAMLContent(blockYAML))
        #expect(blockMeta.trainPath == "train/images")
        #expect(blockMeta.valPath == "valid/images")
        #expect(blockMeta.testPath == nil)
        #expect(blockMeta.nc == 3)
        #expect(blockMeta.classNames == ["cat", "dog", "bird"])

        let inlineYAML = """
        nc: 2
        names: ['car', "school bus", motorcycle] # inline list
        """
        let inlineMeta = try #require(YOLODatasetMetadata.parseYAMLContent(inlineYAML))
        #expect(inlineMeta.nc == 2)
        #expect(inlineMeta.classNames == ["car", "school bus", "motorcycle"])

        #expect(YOLODatasetMetadata.parseYAMLContent("# comment only\n   \n") == nil)
    }

    @Test("YOLODatasetMetadata.parseYAML and YOLOAnnotationParser.parseDataset auto-discover YAML and labels/train")
    func testYAMLParentDiscoveryAndParseDatasetAutoResolution() throws {
        let fm = FileManager.default
        let rootDir = fm.temporaryDirectory.appendingPathComponent("yolo_meta_test_\(UUID().uuidString)")
        let trainImagesDir = rootDir.appendingPathComponent("images/train")
        let trainLabelsDir = rootDir.appendingPathComponent("labels/train")
        try fm.createDirectory(at: trainImagesDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: trainLabelsDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: rootDir) }

        let yamlContent = """
        train: images/train
        val: images/val
        nc: 2
        names:
          0: vehicle
          1: cyclist
        """
        try yamlContent.write(to: rootDir.appendingPathComponent("dataset.yaml"), atomically: true, encoding: .utf8)

        let imgURL = trainImagesDir.appendingPathComponent("frame01.png")
        try Data("png_placeholder".utf8).write(to: imgURL)
        try "0 0.5 0.5 0.4 0.4\n1 0.2 0.2 0.1 0.1\n".write(
            to: trainLabelsDir.appendingPathComponent("frame01.txt"),
            atomically: true,
            encoding: .utf8
        )

        let discovered = try #require(YOLODatasetMetadata.parseYAML(at: trainImagesDir))
        #expect(discovered.classNames == ["vehicle", "cyclist"])
        #expect(discovered.nc == 2)

        let dataset = try YOLOAnnotationParser.parseDataset(imagesDirectory: trainImagesDir)
        #expect(dataset.count == 1)
        #expect(dataset[0].imageURL.lastPathComponent == "frame01.png")
        #expect(dataset[0].groundTruth.count == 2)
        #expect(dataset[0].groundTruth[0].classLabel == "vehicle")
        #expect(dataset[0].groundTruth[1].classLabel == "cyclist")
    }

    @Test("YOLOAnnotationParser.parseDataset resolves trailing /images to /labels and data.yaml")
    func testParseDatasetTrailingImagesAndDataYAML() throws {
        let fm = FileManager.default
        let rootDir = fm.temporaryDirectory.appendingPathComponent("yolo_trailing_test_\(UUID().uuidString)")
        let imagesDir = rootDir.appendingPathComponent("images")
        let labelsDir = rootDir.appendingPathComponent("labels")
        try fm.createDirectory(at: imagesDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: labelsDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: rootDir) }

        let dataYAML = "names: ['apple', 'orange']\n"
        try dataYAML.write(to: rootDir.appendingPathComponent("data.yaml"), atomically: true, encoding: .utf8)

        try Data("img".utf8).write(to: imagesDir.appendingPathComponent("sample.jpg"))
        try "1 0.5 0.5 0.2 0.2\n".write(
            to: labelsDir.appendingPathComponent("sample.txt"),
            atomically: true,
            encoding: .utf8
        )

        let dataset = try YOLOAnnotationParser.parseDataset(imagesDirectory: imagesDir)
        #expect(dataset.count == 1)
        #expect(dataset[0].groundTruth.count == 1)
        #expect(dataset[0].groundTruth[0].classLabel == "orange")
    }

    // MARK: - Normalized Bounding-Box Cropper Tests

    #if canImport(CoreGraphics)
    private func makeQuadrantCGImage(width: Int = 40, height: Int = 40) throws -> CGImage {
        let pixelCount = width * height
        var data = [Double](repeating: 0.0, count: pixelCount * 3)
        let halfW = width / 2
        let halfH = height / 2

        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                if y < halfH && x < halfW {
                    // Top-left: Red
                    data[idx] = 1.0
                } else if y < halfH && x >= halfW {
                    // Top-right: Green
                    data[pixelCount + idx] = 1.0
                } else if y >= halfH && x < halfW {
                    // Bottom-left: Blue
                    data[2 * pixelCount + idx] = 1.0
                } else {
                    // Bottom-right: White
                    data[idx] = 1.0
                    data[pixelCount + idx] = 1.0
                    data[2 * pixelCount + idx] = 1.0
                }
            }
        }

        let dataset = ImageDataset(width: width, height: height, channels: 3, data: data)
        return try #require(dataset.toCGImage())
    }

    @Test("ImageDataset.crop supports BoundingBox, CGRect, paddingFraction, clamping, and degenerate nil")
    func testNormalizedBoundingBoxCropping() throws {
        let cgImage = try makeQuadrantCGImage(width: 40, height: 40)

        // Exact top-left red quadrant via BoundingBox
        let topLeftBox = BoundingBox(xMin: 0.0, yMin: 0.0, xMax: 0.5, yMax: 0.5, confidence: 1.0, classLabel: "red")
        let topLeftCrop = try #require(ImageDataset.crop(image: cgImage, to: topLeftBox))
        #expect(topLeftCrop.width == 20)
        #expect(topLeftCrop.height == 20)
        let topLeftDataset = try ImageDataset.from(cgImage: topLeftCrop)
        #expect(topLeftDataset.data[0] > 0.9)
        #expect(topLeftDataset.data[20 * 20] < 0.1)

        // Padding via CGRect: center [0.25, 0.25, 0.5, 0.5] (20x20) + 0.25 padding -> 30x30
        let centerRect = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)
        let paddedCrop = try #require(ImageDataset.crop(image: cgImage, to: centerRect, paddingFraction: 0.25))
        #expect(paddedCrop.width == 30)
        #expect(paddedCrop.height == 30)

        // Edge clamping: [0.0, 0.0, 0.25, 0.25] (10x10) with padding 0.5 -> clamped to [0, 15] x [0, 15]
        let cornerRect = CGRect(x: 0.0, y: 0.0, width: 0.25, height: 0.25)
        let clampedCrop = try #require(ImageDataset.crop(image: cgImage, to: cornerRect, paddingFraction: 0.5))
        #expect(clampedCrop.width == 15)
        #expect(clampedCrop.height == 15)

        // Degenerate / sub-pixel / out-of-bounds crops return nil
        let subPixelRect = CGRect(x: 0.0, y: 0.0, width: 0.01, height: 0.01) // 0.4 x 0.4 px on 40x40
        #expect(ImageDataset.crop(image: cgImage, to: subPixelRect) == nil)
        let zeroWidthBox = BoundingBox(xMin: 0.5, yMin: 0.5, xMax: 0.5, yMax: 0.8, confidence: 1.0, classLabel: "zero")
        #expect(ImageDataset.crop(image: cgImage, to: zeroWidthBox) == nil)
        let outOfBoundsRect = CGRect(x: 1.2, y: 1.2, width: 0.2, height: 0.2)
        #expect(ImageDataset.crop(image: cgImage, to: outOfBoundsRect) == nil)

        // cropAll filters out degenerate boxes
        let crops = ImageDataset.cropAll(image: cgImage, boxes: [topLeftBox, zeroWidthBox])
        #expect(crops.count == 1)
        #expect(crops[0].box.classLabel == "red")
        #expect(crops[0].image.width == 20)
    }

    @Test("ImageDataset.sampleBackgroundCrop avoids object boxes deterministically and handles full coverage")
    func testSampleBackgroundCrop() throws {
        let cgImage = try makeQuadrantCGImage(width: 40, height: 40)

        // Empty annotations (negative image)
        let emptyBg = try #require(
            ImageDataset.sampleBackgroundCrop(
                from: cgImage,
                avoiding: [],
                size: CGSize(width: 10, height: 10),
                maxAttempts: 8,
                seed: 42
            )
        )
        #expect(emptyBg.width == 10)
        #expect(emptyBg.height == 10)

        // Occupy left half [0.0, 0.0, 0.5, 1.0]; background must come from right half
        let leftHalfBox = BoundingBox(xMin: 0.0, yMin: 0.0, xMax: 0.5, yMax: 1.0, confidence: 1.0, classLabel: "fg")
        let rightBg1 = try #require(
            ImageDataset.sampleBackgroundCrop(
                from: cgImage,
                avoiding: [leftHalfBox],
                size: CGSize(width: 10, height: 10),
                maxAttempts: 32,
                seed: 42
            )
        )
        let rightBg2 = try #require(
            ImageDataset.sampleBackgroundCrop(
                from: cgImage,
                avoiding: [leftHalfBox],
                size: CGSize(width: 10, height: 10),
                maxAttempts: 32,
                seed: 42
            )
        )
        let ds1 = try ImageDataset.from(cgImage: rightBg1)
        let ds2 = try ImageDataset.from(cgImage: rightBg2)
        #expect(ds1.data == ds2.data)
        // Left half has R=1, G=0, B=0 (top) or R=0, G=0, B=1 (bottom); right half always has G=1.0
        #expect(ds1.data[10 * 10] > 0.9)

        // Full image covered -> nil
        let fullBox = BoundingBox(xMin: 0.0, yMin: 0.0, xMax: 1.0, yMax: 1.0, confidence: 1.0, classLabel: "full")
        #expect(
            ImageDataset.sampleBackgroundCrop(
                from: cgImage,
                avoiding: [fullBox],
                size: CGSize(width: 8, height: 8),
                maxAttempts: 16,
                seed: 42
            ) == nil
        )
    }
    #endif

    // MARK: - Split-Aware ImageFolder Loader Tests

    #if canImport(CoreGraphics) && canImport(ImageIO)
    private func writeSolidPNG(
        to url: URL,
        width: Int,
        height: Int,
        r: Double,
        g: Double,
        b: Double
    ) throws {
        let pixelCount = width * height
        var data = [Double](repeating: 0.0, count: pixelCount * 3)
        for p in 0..<pixelCount {
            data[p] = r
            data[pixelCount + p] = g
            data[2 * pixelCount + p] = b
        }
        let img = ImageDataset(width: width, height: height, channels: 3, data: data)
        let pngData = try #require(img.toPNGData())
        try pngData.write(to: url)
    }

    @Test("ImageDataset.loadSplitFolder aligns class indices globally across train, val, and test splits")
    func testLoadSplitFolderGlobalClassAlignment() throws {
        let fm = FileManager.default
        let rootDir = fm.temporaryDirectory.appendingPathComponent("split_folder_test_\(UUID().uuidString)")
        let trainAppleDir = rootDir.appendingPathComponent("train/apple")
        let trainZebraDir = rootDir.appendingPathComponent("train/zebra")
        let valMangoDir = rootDir.appendingPathComponent("val/mango")
        let valZebraDir = rootDir.appendingPathComponent("val/zebra")
        let testAppleDir = rootDir.appendingPathComponent("test/apple")

        for dir in [trainAppleDir, trainZebraDir, valMangoDir, valZebraDir, testAppleDir] {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        defer { try? fm.removeItem(at: rootDir) }

        try writeSolidPNG(to: trainAppleDir.appendingPathComponent("a1.png"), width: 10, height: 10, r: 1.0, g: 0.0, b: 0.0)
        try writeSolidPNG(to: trainZebraDir.appendingPathComponent("z1.png"), width: 12, height: 12, r: 0.0, g: 0.0, b: 1.0)
        try writeSolidPNG(to: valMangoDir.appendingPathComponent("m1.png"), width: 8, height: 8, r: 0.0, g: 1.0, b: 0.0)
        try writeSolidPNG(to: valZebraDir.appendingPathComponent("z2.png"), width: 14, height: 14, r: 0.0, g: 0.0, b: 1.0)
        try writeSolidPNG(to: testAppleDir.appendingPathComponent("a2.png"), width: 16, height: 16, r: 1.0, g: 0.0, b: 0.0)

        let split = try ImageDataset.loadSplitFolder(at: rootDir, targetSize: CGSize(width: 16, height: 16))
        let expectedClasses = ["apple", "mango", "zebra"]

        #expect(split.classes == expectedClasses)
        #expect(split.train.classes == expectedClasses)
        #expect(split.train.images.count == 2)
        // apple=0, zebra=2 even though mango is absent from train
        #expect(split.train.labels == [0, 2])
        #expect(split.train.images[0].width == 16)
        #expect(split.train.images[0].height == 16)

        let val = try #require(split.validation)
        #expect(val.classes == expectedClasses)
        #expect(val.images.count == 2)
        // mango=1, zebra=2
        #expect(val.labels == [1, 2])

        let test = try #require(split.test)
        #expect(test.classes == expectedClasses)
        #expect(test.images.count == 1)
        // apple=0
        #expect(test.labels == [0])
    }

    @Test("ImageDataset.loadSplitFolder falls back cleanly on flat class/image directory")
    func testLoadSplitFolderFlatFallback() throws {
        let fm = FileManager.default
        let rootDir = fm.temporaryDirectory.appendingPathComponent("flat_folder_test_\(UUID().uuidString)")
        let catsDir = rootDir.appendingPathComponent("cats")
        let dogsDir = rootDir.appendingPathComponent("dogs")
        try fm.createDirectory(at: catsDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: dogsDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: rootDir) }

        try writeSolidPNG(to: catsDir.appendingPathComponent("c1.png"), width: 8, height: 8, r: 0.8, g: 0.2, b: 0.2)
        try writeSolidPNG(to: dogsDir.appendingPathComponent("d1.png"), width: 12, height: 12, r: 0.2, g: 0.2, b: 0.8)

        let split = try ImageDataset.loadSplitFolder(at: rootDir, targetSize: CGSize(width: 12, height: 12))
        #expect(split.classes == ["cats", "dogs"])
        #expect(split.train.classes == ["cats", "dogs"])
        #expect(split.train.images.count == 2)
        #expect(split.train.labels == [0, 1])
        #expect(split.train.images[0].width == 12)
        #expect(split.train.images[0].height == 12)
        #expect(split.validation == nil)
        #expect(split.test == nil)
    }

    @Test("ImageDataset.loadSplitFolder rejects nonexistent directories and split folders missing train")
    func testLoadSplitFolderValidationErrors() throws {
        let fm = FileManager.default
        let missingDir = fm.temporaryDirectory.appendingPathComponent("missing_\(UUID().uuidString)")
        #expect(throws: VisionError.self) {
            _ = try ImageDataset.loadSplitFolder(at: missingDir)
        }

        let noTrainDir = fm.temporaryDirectory.appendingPathComponent("no_train_\(UUID().uuidString)")
        try fm.createDirectory(at: noTrainDir.appendingPathComponent("val/cats"), withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: noTrainDir) }

        #expect(throws: VisionError.self) {
            _ = try ImageDataset.loadSplitFolder(at: noTrainDir)
        }
    }
    #endif
}
