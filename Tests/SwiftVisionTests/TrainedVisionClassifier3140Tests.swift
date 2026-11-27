import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers
@testable import SwiftVision

@Suite("TrainedVisionClassifier & Core ML Vision Pipeline Tests (3.14.0)", .serialized)
struct TrainedVisionClassifier3140Tests {

    // MARK: - End-to-End Synthetic Dataset & Core ML Parity

    @Test("TrainedVisionClassifier end-to-end folder training and Core ML parity for OneVsRest, MLP, and RandomForest")
    func testEndToEndFolderTrainAndCoreMLParity() async throws {
        let fm = FileManager.default
        let rootDir = fm.temporaryDirectory.appendingPathComponent("trained_vision_e2e_\(UUID().uuidString)")
        try fm.createDirectory(at: rootDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: rootDir) }

        let classSpecs: [(name: String, variant: Int)] = [
            ("red_stripes", 0),
            ("green_checker", 1),
            ("blue_gradient", 2)
        ]

        // Create train/<class>/ PNGs (3 variations per class) and held-out test/<class>/ PNGs
        var heldOutEntries: [(label: String, variant: Int, url: URL, cgImage: CGImage, pixelBuffer: CVPixelBuffer)] = []

        for spec in classSpecs {
            let trainClassDir = rootDir.appendingPathComponent("train/\(spec.name)")
            let testClassDir = rootDir.appendingPathComponent("test/\(spec.name)")
            try fm.createDirectory(at: trainClassDir, withIntermediateDirectories: true)
            try fm.createDirectory(at: testClassDir, withIntermediateDirectories: true)

            for sampleIdx in 0..<3 {
                let phase = sampleIdx * 6
                let pb = try makePatternPixelBuffer(width: 360, height: 360, variant: spec.variant, phase: phase)
                let cg = try makeCGImage(from: pb)
                let fileURL = trainClassDir.appendingPathComponent("sample_\(sampleIdx).png")
                try writePNG(cgImage: cg, to: fileURL)
            }

            let heldOutPhase = 15
            let heldOutPB = try makePatternPixelBuffer(width: 360, height: 360, variant: spec.variant, phase: heldOutPhase)
            let heldOutCG = try makeCGImage(from: heldOutPB)
            let heldOutURL = testClassDir.appendingPathComponent("heldout.png")
            try writePNG(cgImage: heldOutCG, to: heldOutURL)

            // Reload from PNG so CGImage and CVPixelBuffer reflect identical disk pixels
            let diskCG = try loadCGImage(from: heldOutURL)
            let diskPB = try makePixelBuffer(from: diskCG)
            heldOutEntries.append((spec.name, spec.variant, heldOutURL, diskCG, diskPB))
        }

        let expectedSortedClasses = ["blue_gradient", "green_checker", "red_stripes"]

        // 1. OneVsRest backend
        let ovrClassifier = TrainedVisionClassifier(
            backend: .oneVsRest(learningRate: 0.5, maxIter: 200, l2Penalty: 1e-4)
        )
        try await ovrClassifier.fit(folderURL: rootDir)
        #expect(await ovrClassifier.classes == expectedSortedClasses)
        #expect(await ovrClassifier.featureDimension == 768)

        let ovrModelURL = rootDir.appendingPathComponent("ovr_pipeline.mlmodel")
        try await ovrClassifier.exportCoreMLPipeline(to: ovrModelURL)
        #expect(fm.fileExists(atPath: ovrModelURL.path))

        try await verifySwiftAndCoreMLParity(
            classifier: ovrClassifier,
            mlmodelURL: ovrModelURL,
            heldOutEntries: heldOutEntries,
            expectedClasses: expectedSortedClasses,
            probabilityTolerance: 1e-3,
            expectExactGroundTruth: true
        )

        // 2. MLP backend
        let mlpClassifier = TrainedVisionClassifier(
            backend: .mlp(
                hiddenLayers: [32],
                activation: .relu,
                learningRate: 0.005,
                maxIter: 80,
                batchSize: 9
            )
        )
        try await mlpClassifier.fit(folderURL: rootDir)
        #expect(await mlpClassifier.classes == expectedSortedClasses)
        #expect(await mlpClassifier.featureDimension == 768)

        let mlpModelURL = rootDir.appendingPathComponent("mlp_pipeline.mlmodel")
        try await mlpClassifier.exportCoreMLPipeline(to: mlpModelURL)
        #expect(fm.fileExists(atPath: mlpModelURL.path))

        try await verifySwiftAndCoreMLParity(
            classifier: mlpClassifier,
            mlmodelURL: mlpModelURL,
            heldOutEntries: heldOutEntries,
            expectedClasses: expectedSortedClasses,
            probabilityTolerance: 1e-2,
            expectExactGroundTruth: true
        )

        // 3. RandomForest backend
        let rfClassifier = TrainedVisionClassifier(
            backend: .randomForest(nEstimators: 15, maxDepth: 6, seed: 42)
        )
        try await rfClassifier.fit(folderURL: rootDir)
        #expect(await rfClassifier.classes == expectedSortedClasses)
        #expect(await rfClassifier.featureDimension == 768)

        let rfModelURL = rootDir.appendingPathComponent("rf_pipeline.mlmodel")
        try await rfClassifier.exportCoreMLPipeline(to: rfModelURL)
        #expect(fm.fileExists(atPath: rfModelURL.path))

        try await verifySwiftAndCoreMLParity(
            classifier: rfClassifier,
            mlmodelURL: rfModelURL,
            heldOutEntries: heldOutEntries,
            expectedClasses: expectedSortedClasses,
            probabilityTolerance: 1e-3,
            expectExactGroundTruth: true
        )
    }

    @Test("TrainedVisionClassifier supports in-memory CGImage training, imageURLs binary LogisticRegression, and DecisionTree export")
    func testInMemoryCGImageDecisionTreeAndBinaryLogisticRegression() async throws {
        let fm = FileManager.default
        let rootDir = fm.temporaryDirectory.appendingPathComponent("trained_vision_extra_\(UUID().uuidString)")
        try fm.createDirectory(at: rootDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: rootDir) }

        var images: [CGImage] = []
        var imageURLs: [URL] = []
        var binaryLabels: [String] = []

        for sampleIdx in 0..<3 {
            let pbA = try makePatternPixelBuffer(width: 360, height: 360, variant: 0, phase: sampleIdx * 4)
            let cgA = try makeCGImage(from: pbA)
            let urlA = rootDir.appendingPathComponent("stripes_\(sampleIdx).png")
            try writePNG(cgImage: cgA, to: urlA)
            images.append(try loadCGImage(from: urlA))
            imageURLs.append(urlA)
            binaryLabels.append("stripes")

            let pbB = try makePatternPixelBuffer(width: 360, height: 360, variant: 1, phase: sampleIdx * 4)
            let cgB = try makeCGImage(from: pbB)
            let urlB = rootDir.appendingPathComponent("checker_\(sampleIdx).png")
            try writePNG(cgImage: cgB, to: urlB)
            images.append(try loadCGImage(from: urlB))
            imageURLs.append(urlB)
            binaryLabels.append("checker")
        }

        // DecisionTree trained from in-memory CGImages
        let dtClassifier = TrainedVisionClassifier(backend: .decisionTree(maxDepth: 5))
        try await dtClassifier.fit(images: images, labels: binaryLabels)
        #expect(await dtClassifier.classes == ["checker", "stripes"])
        #expect(await dtClassifier.featureDimension == 768)

        let dtURL = rootDir.appendingPathComponent("dt_pipeline.mlmodel")
        try await dtClassifier.exportCoreMLPipeline(to: dtURL, inputImageName: "photo", outputLabelName: "category")
        let dtCompiledURL = try await MLModel.compileModel(at: dtURL)
        defer { try? fm.removeItem(at: dtCompiledURL) }

        let dtModel = try MLModel(contentsOf: dtCompiledURL)
        let testPB = try makePixelBuffer(from: images[0])
        let dtSwiftPred = try await dtClassifier.predict(image: images[0])
        let dtProvider = try MLDictionaryFeatureProvider(dictionary: ["photo": MLFeatureValue(pixelBuffer: testPB)])
        let dtOut = try await dtModel.prediction(from: dtProvider)
        #expect(dtOut.featureValue(for: "category")?.stringValue == dtSwiftPred)

        // LogisticRegression trained from imageURLs
        let lrClassifier = TrainedVisionClassifier(backend: .logisticRegression(learningRate: 0.5, maxIter: 150))
        try await lrClassifier.fit(imageURLs: imageURLs, labels: binaryLabels)
        #expect(await lrClassifier.classes == ["checker", "stripes"])

        let lrData = try await lrClassifier.exportCoreMLPipelineData()
        #expect(!lrData.isEmpty)
        let lrURL = rootDir.appendingPathComponent("lr_pipeline.mlmodel")
        try lrData.write(to: lrURL)
        let lrCompiledURL = try await MLModel.compileModel(at: lrURL)
        defer { try? fm.removeItem(at: lrCompiledURL) }

        let lrModel = try MLModel(contentsOf: lrCompiledURL)
        let lrProbName = lrModel.modelDescription.predictedProbabilitiesName ?? "classProbability"
        let lrSwiftPred = try await lrClassifier.predictProbability(imageURL: imageURLs[1])
        #expect(lrSwiftPred.label == "checker")

        let pb1 = try makePixelBuffer(from: images[1])
        let lrProvider = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: pb1)])
        let lrOut = try await lrModel.prediction(from: lrProvider)
        #expect(lrOut.featureValue(for: "label")?.stringValue == lrSwiftPred.label)
        let lrDict = try #require(lrOut.featureValue(for: lrProbName)?.dictionaryValue)
        let coremlCheckerProb = lrDict["checker"]?.doubleValue ?? -1.0
        let swiftCheckerProb = lrSwiftPred.probabilities["checker"] ?? -2.0
        #expect(abs(coremlCheckerProb - swiftCheckerProb) < 1e-3)
    }

    // MARK: - Error Path Tests

    @Test("TrainedVisionClassifier error paths: unfitted, empty, mismatched counts, single-class, and invalid params")
    func testErrorPaths() async throws {
        let fm = FileManager.default
        let tempDir = fm.temporaryDirectory.appendingPathComponent("trained_vision_err_\(UUID().uuidString)")
        try fm.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tempDir) }

        let pb = try makePatternPixelBuffer(width: 64, height: 64, variant: 0, phase: 0)
        let cg = try makeCGImage(from: pb)
        let imgURL = tempDir.appendingPathComponent("sample.png")
        try writePNG(cgImage: cg, to: imgURL)

        // 1. Unfitted predict & export errors
        let unfitted = TrainedVisionClassifier()
        do {
            _ = try await unfitted.predict(imageURL: imgURL)
            Issue.record("Expected modelNotFitted for predict(imageURL:)")
        } catch let err as SwiftMLError {
            #expect(err == .modelNotFitted)
        }
        do {
            _ = try await unfitted.predict(image: cg)
            Issue.record("Expected modelNotFitted for predict(image:)")
        } catch let err as SwiftMLError {
            #expect(err == .modelNotFitted)
        }
        do {
            _ = try await unfitted.predictProbability(imageURL: imgURL)
            Issue.record("Expected modelNotFitted for predictProbability(imageURL:)")
        } catch let err as SwiftMLError {
            #expect(err == .modelNotFitted)
        }
        do {
            _ = try await unfitted.predictProbability(image: cg)
            Issue.record("Expected modelNotFitted for predictProbability(image:)")
        } catch let err as SwiftMLError {
            #expect(err == .modelNotFitted)
        }
        do {
            _ = try await unfitted.exportCoreMLPipelineData()
            Issue.record("Expected modelNotFitted for exportCoreMLPipelineData()")
        } catch let err as SwiftMLError {
            #expect(err == .modelNotFitted)
        }
        do {
            try await unfitted.exportCoreMLPipeline(to: tempDir.appendingPathComponent("unfitted.mlmodel"))
            Issue.record("Expected modelNotFitted for exportCoreMLPipeline(to:)")
        } catch let err as SwiftMLError {
            #expect(err == .modelNotFitted)
        }

        // 2. Empty inputs
        do {
            try await unfitted.fit(imageURLs: [], labels: [])
            Issue.record("Expected emptyInput for empty imageURLs")
        } catch let err as SwiftMLError {
            #expect(err == .emptyInput)
        }
        do {
            try await unfitted.fit(images: [], labels: [])
            Issue.record("Expected emptyInput for empty images")
        } catch let err as SwiftMLError {
            #expect(err == .emptyInput)
        }
        let emptyFolder = tempDir.appendingPathComponent("empty_dataset")
        try fm.createDirectory(at: emptyFolder, withIntermediateDirectories: true)
        do {
            try await unfitted.fit(folderURL: emptyFolder)
            Issue.record("Expected emptyInput for empty folder")
        } catch let err as SwiftMLError {
            #expect(err == .emptyInput)
        }

        // 3. Mismatched counts
        do {
            try await unfitted.fit(imageURLs: [imgURL, imgURL], labels: ["only_one"])
            Issue.record("Expected dimensionMismatch for mismatched imageURLs/labels")
        } catch let err as SwiftMLError {
            #expect(err == .dimensionMismatch(expected: 2, got: 1))
        }
        do {
            try await unfitted.fit(images: [cg], labels: ["a", "b"])
            Issue.record("Expected dimensionMismatch for mismatched images/labels")
        } catch let err as SwiftMLError {
            #expect(err == .dimensionMismatch(expected: 1, got: 2))
        }

        // 4. Single-class dataset rejection
        do {
            try await unfitted.fit(images: [cg, cg], labels: ["same", "same"])
            Issue.record("Expected invalidParameter for single-class dataset")
        } catch let err as SwiftMLError {
            if case .invalidParameter = err {
                // Expected
            } else {
                Issue.record("Unexpected SwiftMLError: \(err)")
            }
        }

        // 5. Binary LogisticRegression with 3 classes
        let lrMulticlass = TrainedVisionClassifier(backend: .logisticRegression())
        do {
            try await lrMulticlass.fit(images: [cg, cg, cg], labels: ["a", "b", "c"])
            Issue.record("Expected invalidParameter for multiclass LogisticRegression")
        } catch let err as SwiftMLError {
            if case .invalidParameter = err {
                // Expected
            } else {
                Issue.record("Unexpected SwiftMLError: \(err)")
            }
        }

        // 6. Missing directory & invalid hyperparameters
        let missingDir = tempDir.appendingPathComponent("does_not_exist")
        do {
            try await unfitted.fit(folderURL: missingDir)
            Issue.record("Expected VisionError.invalidInput for missing directory")
        } catch let err as VisionError {
            if case .invalidInput = err {
                // Expected
            } else {
                Issue.record("Unexpected VisionError: \(err)")
            }
        }

        let badBackend = TrainedVisionClassifier(backend: .mlp(hiddenLayers: [0]))
        do {
            try await badBackend.fit(images: [cg, cg], labels: ["a", "b"])
            Issue.record("Expected invalidParameter for zero-width MLP layer")
        } catch let err as SwiftMLError {
            if case .invalidParameter = err {
                // Expected
            } else {
                Issue.record("Unexpected SwiftMLError: \(err)")
            }
        }
    }

    // MARK: - Parity & Synthetic Image Helpers

    private func verifySwiftAndCoreMLParity(
        classifier: TrainedVisionClassifier,
        mlmodelURL: URL,
        heldOutEntries: [(label: String, variant: Int, url: URL, cgImage: CGImage, pixelBuffer: CVPixelBuffer)],
        expectedClasses: [String],
        probabilityTolerance: Double,
        expectExactGroundTruth: Bool
    ) async throws {
        let compiledURL = try await MLModel.compileModel(at: mlmodelURL)
        defer { try? FileManager.default.removeItem(at: compiledURL) }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        let model = try MLModel(contentsOf: compiledURL, configuration: config)
        let probName = model.modelDescription.predictedProbabilitiesName ?? "classProbability"

        for entry in heldOutEntries {
            let urlPred = try await classifier.predict(imageURL: entry.url)
            let cgProbPred = try await classifier.predictProbability(image: entry.cgImage)
            let urlProbPred = try await classifier.predictProbability(imageURL: entry.url)

            #expect(urlPred == cgProbPred.label)
            #expect(urlProbPred.label == cgProbPred.label)
            if expectExactGroundTruth {
                #expect(urlPred == entry.label)
            }

            let sumProb = cgProbPred.probabilities.values.reduce(0.0, +)
            #expect(abs(sumProb - 1.0) < 1e-6)
            #expect(cgProbPred.confidence == cgProbPred.probabilities[cgProbPred.label])

            // Core ML inference using CVPixelBuffer
            let pbProvider = try MLDictionaryFeatureProvider(dictionary: [
                "image": MLFeatureValue(pixelBuffer: entry.pixelBuffer)
            ])
            let pbOut = try await model.prediction(from: pbProvider)
            let coremlPBLabel = try #require(pbOut.featureValue(for: "label")?.stringValue)
            #expect(coremlPBLabel == cgProbPred.label)

            let probDict = try #require(pbOut.featureValue(for: probName)?.dictionaryValue)
            for className in expectedClasses {
                let coremlP = probDict[className]?.doubleValue ?? -1.0
                let swiftP = cgProbPred.probabilities[className] ?? -2.0
                #expect(abs(coremlP - swiftP) < probabilityTolerance)
            }

            // Core ML inference using CGImage directly
            let cgFeatureValue = try MLFeatureValue(
                cgImage: entry.cgImage,
                pixelsWide: 360,
                pixelsHigh: 360,
                pixelFormatType: kCVPixelFormatType_32BGRA,
                options: nil
            )
            let cgProvider = try MLDictionaryFeatureProvider(dictionary: ["image": cgFeatureValue])
            let cgOut = try await model.prediction(from: cgProvider)
            let coremlCGLabel = try #require(cgOut.featureValue(for: "label")?.stringValue)
            #expect(coremlCGLabel == cgProbPred.label)
        }
    }

    private func makePatternPixelBuffer(width: Int, height: Int, variant: Int, phase: Int) throws -> CVPixelBuffer {
        var pixelBufferOpt: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBufferOpt
        )
        #expect(status == kCVReturnSuccess)
        let pixelBuffer = try #require(pixelBufferOpt)

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        let baseAddr = try #require(CVPixelBufferGetBaseAddress(pixelBuffer))
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let bytes = baseAddr.assumingMemoryBound(to: UInt8.self)

        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytesPerRow + x * 4
                let r: UInt8
                let g: UInt8
                let b: UInt8
                switch variant {
                case 0:
                    // Red horizontal stripes
                    let stripe = ((y + phase) / 20) % 2 == 0
                    r = stripe ? 240 : 120
                    g = stripe ? 20 : 10
                    b = stripe ? 20 : 10
                case 1:
                    // Green high-contrast checkerboard
                    let block = (((x + phase) / 24) + ((y + phase) / 24)) % 2 == 0
                    r = block ? 20 : 10
                    g = block ? 235 : 40
                    b = block ? 30 : 210
                default:
                    // Blue diagonal gradient
                    r = UInt8(min(255, ((x + phase) * 120) / max(1, width)))
                    g = UInt8(min(255, ((y + phase) * 160) / max(1, height)))
                    b = 240
                }
                bytes[offset + 0] = b
                bytes[offset + 1] = g
                bytes[offset + 2] = r
                bytes[offset + 3] = 255
            }
        }
        return pixelBuffer
    }

    private func makeCGImage(from pixelBuffer: CVPixelBuffer) throws -> CGImage {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let baseAddress = try #require(CVPixelBufferGetBaseAddress(pixelBuffer))
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue

        let context = try #require(CGContext(
            data: baseAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ))
        return try #require(context.makeImage())
    }

    private func makePixelBuffer(from cgImage: CGImage) throws -> CVPixelBuffer {
        let width = cgImage.width
        let height = cgImage.height
        var pixelBufferOpt: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBufferOpt
        )
        #expect(status == kCVReturnSuccess)
        let pixelBuffer = try #require(pixelBufferOpt)

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        let baseAddress = try #require(CVPixelBufferGetBaseAddress(pixelBuffer))
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue

        let context = try #require(CGContext(
            data: baseAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ))
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pixelBuffer
    }

    private func writePNG(cgImage: CGImage, to url: URL) throws {
        let dest = try #require(CGImageDestinationCreateWithURL(
            url as CFURL,
            UTType.png.identifier as CFString,
            1,
            nil
        ))
        CGImageDestinationAddImage(dest, cgImage, nil)
        let ok = CGImageDestinationFinalize(dest)
        #expect(ok)
    }

    private func loadCGImage(from url: URL) throws -> CGImage {
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        return try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
    }
}
