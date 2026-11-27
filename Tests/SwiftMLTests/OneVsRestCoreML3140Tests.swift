import CoreML
import CoreVideo
import Foundation
import Testing
import Vision
@testable import SwiftML

@Suite("OneVsRestClassifier & Core ML Vision Pipeline (3.14.0)", .serialized)
struct OneVsRestCoreML3140Tests {

    // MARK: - OneVsRestClassifier API & Parity

    @Test("OneVsRestClassifier getBias, getWeightsAndBias, decisionFunction, and predictProbability validation")
    func testOneVsRestMethodsAndValidation() async throws {
        let unfitted = OneVsRestClassifier(numClasses: 3)
        do {
            _ = try await unfitted.getBias(forClass: 0)
            Issue.record("Expected modelNotFitted for getBias")
        } catch let err as SwiftMLError {
            #expect(err == .modelNotFitted)
        }
        do {
            _ = try await unfitted.decisionFunction(features: [[1.0, 2.0]])
            Issue.record("Expected modelNotFitted for decisionFunction")
        } catch let err as SwiftMLError {
            #expect(err == .modelNotFitted)
        }
        do {
            _ = try await unfitted.exportCoreML(featureNames: ["a", "b"], outputName: "label")
            Issue.record("Expected modelNotFitted for exportCoreML")
        } catch let err as SwiftMLError {
            #expect(err == .modelNotFitted)
        }

        let X: [[Double]] = [
            [3.0, 0.0, 0.0],
            [2.5, 0.2, 0.0],
            [0.0, 3.0, 0.0],
            [0.1, 2.7, 0.1],
            [0.0, 0.0, 3.0],
            [0.0, 0.1, 2.8]
        ]
        let y: [Double] = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]

        let ovr = OneVsRestClassifier(numClasses: 3)
        try await ovr.fit(features: X, targets: y, learningRate: 1.0, epochs: 400)
        #expect(await ovr.classes == [0.0, 1.0, 2.0])

        // Out-of-range class checks
        do {
            _ = try await ovr.getBias(forClass: -1)
            Issue.record("Expected indexOutOfRange for negative class index")
        } catch let err as SwiftMLError {
            #expect(err == .indexOutOfRange(index: -1, count: 3))
        }
        do {
            _ = try await ovr.getWeightsAndBias(forClass: 3)
            Issue.record("Expected indexOutOfRange for class index == count")
        } catch let err as SwiftMLError {
            #expect(err == .indexOutOfRange(index: 3, count: 3))
        }

        // Empty / dimension mismatch / NaN validation on decisionFunction
        do {
            _ = try await ovr.decisionFunction(features: [])
            Issue.record("Expected emptyInput")
        } catch let err as SwiftMLError {
            #expect(err == .emptyInput)
        }
        do {
            _ = try await ovr.decisionFunction(features: [[1.0, 2.0]])
            Issue.record("Expected dimensionMismatch")
        } catch let err as SwiftMLError {
            #expect(err == .dimensionMismatch(expected: 3, got: 2))
        }
        do {
            _ = try await ovr.decisionFunction(features: [[1.0, Double.nan, 0.0]])
            Issue.record("Expected containsNaN")
        } catch let err as SwiftMLError {
            #expect(err == .containsNaN)
        }

        // Verify decisionFunction matches manual w_c^T x + b_c and predictProbability is L1-normalized sigmoid
        let logits = try await ovr.decisionFunction(features: X)
        let probs = try await ovr.predictProbability(features: X)
        #expect(logits.count == X.count)
        #expect(probs.count == X.count)

        for c in 0..<3 {
            let (weights, bias) = try await ovr.getWeightsAndBias(forClass: c)
            let biasOnly = try await ovr.getBias(forClass: c)
            let weightsOnly = try await ovr.getWeights(forClass: c)
            #expect(weights == weightsOnly)
            #expect(bias == biasOnly)
            #expect(weights.count == 3)

            for i in 0..<X.count {
                var expectedZ = bias
                for j in 0..<3 {
                    expectedZ += weights[j] * X[i][j]
                }
                #expect(abs(logits[i][c] - expectedZ) < 1e-12)
            }
        }

        for i in 0..<X.count {
            var marginals: [Double] = []
            var sum = 0.0
            for c in 0..<3 {
                let p = sigmoid(logits[i][c])
                marginals.append(p)
                sum += p
            }
            let rowSum = probs[i].reduce(0.0, +)
            #expect(abs(rowSum - 1.0) < 1e-12)
            for c in 0..<3 {
                #expect(abs(probs[i][c] - marginals[c] / sum) < 1e-12)
            }
        }
    }

    @Test("OneVsRestClassifier binary C=2 parity with LogisticRegression")
    func testBinaryParityWithLogisticRegression() async throws {
        let X: [[Double]] = [
            [-2.0, -1.0],
            [-1.5, -0.5],
            [-1.0, -1.2],
            [1.5, 1.0],
            [2.0, 1.5],
            [2.5, 2.0]
        ]
        let y: [Double] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

        let ovr = OneVsRestClassifier(numClasses: 2)
        let lr = LogisticRegression(device: .auto)

        try await ovr.fit(features: X, targets: y, learningRate: 0.5, epochs: 400)
        try await lr.fit(features: X, targets: y, learningRate: 0.5, epochs: 400)

        let (ovrW, ovrB) = try await ovr.getWeightsAndBias(forClass: 1)
        let (lrWOpt, lrBOpt) = await lr.getWeightsAndBias()
        let lrW = try #require(lrWOpt)
        let lrB = try #require(lrBOpt)

        #expect(ovrW.count == lrW.count)
        for j in 0..<ovrW.count {
            #expect(abs(ovrW[j] - lrW[j]) < 1e-5)
        }
        #expect(abs(ovrB - lrB) < 1e-5)

        let ovrProbs = try await ovr.predictProbability(features: X)
        let lrProbs = try await lr.predictProbability(features: X)
        let ovrPreds = try await ovr.predict(features: X)
        let lrPreds = try await lr.predict(features: X)

        #expect(ovrPreds == lrPreds)
        for i in 0..<X.count {
            #expect(abs(ovrProbs[i][0] - lrProbs[i][0]) < 1e-5)
            #expect(abs(ovrProbs[i][1] - lrProbs[i][1]) < 1e-5)
        }
    }

    @Test("OneVsRestClassifier Core ML compilation and probability parity for C=2 and C=4")
    func testOneVsRestCoreMLCompileAndParity() async throws {
        // Binary C=2
        let X2: [[Double]] = [
            [-2.0, -1.0],
            [-1.0, -2.0],
            [2.0, 1.0],
            [1.0, 2.0]
        ]
        let y2: [Double] = [0.0, 0.0, 1.0, 1.0]
        let ovr2 = OneVsRestClassifier(numClasses: 2)
        try await ovr2.fit(features: X2, targets: y2, learningRate: 1.0, epochs: 300)

        let data2 = try await ovr2.exportCoreML(featureNames: ["f0", "f1"], outputName: "label")
        let swiftPreds2 = try await ovr2.predict(features: X2)
        let swiftProbs2 = try await ovr2.predictProbability(features: X2)

        try await withCompiledCoreML(data2) { (compiledURL: URL) async throws in
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly
            let model = try MLModel(contentsOf: compiledURL, configuration: config)
            let probName = model.modelDescription.predictedProbabilitiesName ?? "classProbability"

            for i in 0..<X2.count {
                let provider = try MLDictionaryFeatureProvider(dictionary: ["f0": X2[i][0], "f1": X2[i][1]])
                let out = try await model.prediction(from: provider)
                let predLabel = try #require(out.featureValue(for: "label")?.int64Value)
                #expect(predLabel == Int64(swiftPreds2[i]))

                let probDict = try #require(out.featureValue(for: probName)?.dictionaryValue)
                let p0 = probDict[NSNumber(value: 0)]?.doubleValue ?? -1.0
                let p1 = probDict[NSNumber(value: 1)]?.doubleValue ?? -1.0
                #expect(abs(p0 - swiftProbs2[i][0]) < 1e-5)
                #expect(abs(p1 - swiftProbs2[i][1]) < 1e-5)
            }
        }

        // Multiclass C=4
        let X4: [[Double]] = [
            [4.0, 0.0, 0.0, 0.0],
            [3.5, 0.2, 0.0, 0.1],
            [0.0, 4.0, 0.0, 0.0],
            [0.1, 3.6, 0.2, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.1, 3.7, 0.2],
            [0.0, 0.0, 0.0, 4.0],
            [0.2, 0.0, 0.1, 3.8]
        ]
        let y4: [Double] = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        let ovr4 = OneVsRestClassifier(numClasses: 4)
        try await ovr4.fit(features: X4, targets: y4, learningRate: 1.0, epochs: 400)

        let featureNames4 = ["f0", "f1", "f2", "f3"]
        let data4 = try await ovr4.exportCoreML(featureNames: featureNames4, outputName: "label")
        let swiftPreds4 = try await ovr4.predict(features: X4)
        let swiftProbs4 = try await ovr4.predictProbability(features: X4)

        try await withCompiledCoreML(data4) { (compiledURL: URL) async throws in
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly
            let model = try MLModel(contentsOf: compiledURL, configuration: config)
            let probName = model.modelDescription.predictedProbabilitiesName ?? "classProbability"

            for i in 0..<X4.count {
                let provider = try MLDictionaryFeatureProvider(dictionary: [
                    "f0": X4[i][0],
                    "f1": X4[i][1],
                    "f2": X4[i][2],
                    "f3": X4[i][3]
                ])
                let out = try await model.prediction(from: provider)
                let predLabel = try #require(out.featureValue(for: "label")?.int64Value)
                #expect(predLabel == Int64(swiftPreds4[i]))

                let probDict = try #require(out.featureValue(for: probName)?.dictionaryValue)
                for c in 0..<4 {
                    let coremlProb = probDict[NSNumber(value: c)]?.doubleValue ?? -1.0
                    #expect(abs(coremlProb - swiftProbs4[i][c]) < 1e-4)
                }
            }
        }
    }

    // MARK: - VisionFeaturePrint & Vision Pipeline Export

    @Test("exportBinaryVisionFeaturePrint v1 and v2 compile cleanly with MLModel.compileModel")
    func testVisionFeaturePrintStandaloneCompilation() async throws {
        let v1Data = CoreMLExporter.exportBinaryVisionFeaturePrint(
            inputImageName: "image",
            outputFeatureName: "scenePrint",
            revision: .v1
        )
        try await withCompiledCoreML(v1Data) { (compiledURL: URL) async throws in
            let model = try MLModel(contentsOf: compiledURL)
            let inputDesc = try #require(model.modelDescription.inputDescriptionsByName["image"])
            let outputDesc = try #require(model.modelDescription.outputDescriptionsByName["scenePrint"])
            #expect(inputDesc.type == .image)
            #expect(inputDesc.imageConstraint?.pixelsWide == 299)
            #expect(inputDesc.imageConstraint?.pixelsHigh == 299)
            #expect(outputDesc.type == .multiArray)
            #expect(outputDesc.multiArrayConstraint?.shape == [NSNumber(value: 2048)])
        }

        let v2Data = CoreMLExporter.exportBinaryVisionFeaturePrint(
            inputImageName: "image",
            outputFeatureName: "scenePrint",
            revision: .v2
        )
        try await withCompiledCoreML(v2Data) { (compiledURL: URL) async throws in
            let model = try MLModel(contentsOf: compiledURL)
            let inputDesc = try #require(model.modelDescription.inputDescriptionsByName["image"])
            let outputDesc = try #require(model.modelDescription.outputDescriptionsByName["scenePrint"])
            #expect(inputDesc.type == .image)
            #expect(inputDesc.imageConstraint?.pixelsWide == 360)
            #expect(inputDesc.imageConstraint?.pixelsHigh == 360)
            #expect(outputDesc.type == .multiArray)
            #expect(outputDesc.multiArrayConstraint?.shape == [NSNumber(value: 768)])
        }
    }

    @Test("exportVisionClassifierPipeline validation errors")
    func testVisionPipelineValidationErrors() async throws {
        let unfitted = OneVsRestClassifier(numClasses: 2)
        do {
            _ = try await CoreMLExporter.exportVisionClassifierPipeline(classifier: unfitted, revision: .v2)
            Issue.record("Expected modelNotFitted")
        } catch let err as SwiftMLError {
            #expect(err == .modelNotFitted)
        }

        let wrongDim = OneVsRestClassifier(numClasses: 2)
        try await wrongDim.fit(features: [[0.0, 0.0], [1.0, 1.0]], targets: [0.0, 1.0])
        do {
            _ = try await CoreMLExporter.exportVisionClassifierPipeline(classifier: wrongDim, revision: .v2)
            Issue.record("Expected dimensionMismatch")
        } catch let err as SwiftMLError {
            #expect(err == .dimensionMismatch(expected: 768, got: 2))
        }

        let reg = LinearRegression()
        try await reg.fit(features: [[1.0], [2.0]], targets: [2.0, 4.0])
        do {
            _ = try await CoreMLExporter.exportVisionClassifierPipeline(classifier: reg, revision: .v2)
            Issue.record("Expected unsupportedOperation")
        } catch let err as SwiftMLError {
            #expect(err == .unsupportedOperation("Vision classifier pipeline export does not support LinearRegression."))
        }
    }

    @Test("End-to-end VisionFeaturePrint v2 CVPixelBuffer parity for OneVsRestClassifier, RandomForestClassifier, and MLPClassifier")
    func testVisionClassifierPipelineEndToEndCVPixelBufferParity() async throws {
        // Generate 3 distinct 360x360 synthetic images (2 samples each with subtle variations)
        let bufferA1 = try makePatternPixelBuffer(width: 360, height: 360, variant: 0, phase: 0)
        let bufferA2 = try makePatternPixelBuffer(width: 360, height: 360, variant: 0, phase: 4)
        let bufferB1 = try makePatternPixelBuffer(width: 360, height: 360, variant: 1, phase: 0)
        let bufferB2 = try makePatternPixelBuffer(width: 360, height: 360, variant: 1, phase: 4)
        let bufferC1 = try makePatternPixelBuffer(width: 360, height: 360, variant: 2, phase: 0)
        let bufferC2 = try makePatternPixelBuffer(width: 360, height: 360, variant: 2, phase: 4)

        let buffers: [CVPixelBuffer] = [bufferA1, bufferA2, bufferB1, bufferB2, bufferC1, bufferC2]
        let targets: [Double] = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
        let classNames: [String] = ["checker", "stripes", "gradient"]

        var embeddings: [[Double]] = []
        embeddings.reserveCapacity(buffers.count)
        for pb in buffers {
            let emb = try extractVisionScenePrintV2(pixelBuffer: pb)
            #expect(emb.count == 768)
            embeddings.append(emb)
        }

        // 1. OneVsRestClassifier (C=3, String labels)
        let ovr = OneVsRestClassifier(numClasses: 3)
        try await ovr.fit(features: embeddings, targets: targets, learningRate: 0.5, epochs: 120)
        let ovrPipelineData = try await CoreMLExporter.exportVisionClassifierPipeline(
            classifier: ovr,
            inputImageName: "image",
            outputLabelName: "label",
            classLabels: classNames,
            revision: .v2
        )
        let ovrSwiftPreds = try await ovr.predict(features: embeddings)
        let ovrSwiftProbs = try await ovr.predictProbability(features: embeddings)

        try await verifyVisionPipelineParity(
            modelData: ovrPipelineData,
            buffers: buffers,
            expectedStringLabels: ovrSwiftPreds.map { (idx: Int) -> String in classNames[idx] },
            expectedProbs: ovrSwiftProbs,
            classNames: classNames,
            tolerance: 1e-3
        )

        // 2. RandomForestClassifier (String labels)
        let rf = RandomForestClassifier(nEstimators: 5, maxDepth: 4, randomState: 42)
        try await rf.fit(features: embeddings, targets: targets)
        let rfPipelineData = try await CoreMLExporter.exportVisionClassifierPipeline(
            classifier: rf,
            inputImageName: "image",
            outputLabelName: "label",
            classLabels: classNames,
            revision: .v2
        )
        let rfSwiftPreds = try await rf.predict(features: embeddings)
        let rfSwiftProbs = try await rf.predictProbability(features: embeddings)

        try await verifyVisionPipelineParity(
            modelData: rfPipelineData,
            buffers: buffers,
            expectedStringLabels: rfSwiftPreds.map { (idx: Int) -> String in classNames[idx] },
            expectedProbs: rfSwiftProbs,
            classNames: classNames,
            tolerance: 1e-3
        )

        // 3. MLPClassifier (NeuralNetworkClassifier field 403, String labels)
        let mlp = MLPClassifier(hiddenLayerSizes: [16], maxIter: 40, batchSize: 6, seed: 42, requestedDevice: .cpu)
        try await mlp.fit(features: embeddings, targets: targets)
        let mlpPipelineData = try await CoreMLExporter.exportVisionClassifierPipeline(
            classifier: mlp,
            inputImageName: "image",
            outputLabelName: "label",
            classLabels: classNames,
            revision: .v2
        )
        let mlpSwiftPreds = try await mlp.predict(features: embeddings)
        let mlpSwiftProbs = try await mlp.predictProbability(features: embeddings)

        try await verifyVisionPipelineParity(
            modelData: mlpPipelineData,
            buffers: buffers,
            expectedStringLabels: mlpSwiftPreds.map { (idx: Int) -> String in classNames[idx] },
            expectedProbs: mlpSwiftProbs,
            classNames: classNames,
            tolerance: 1e-2
        )

        // 4. Int64 labels + DecisionTreeClassifier & LogisticRegression compilation check
        let dt = DecisionTreeClassifier(maxDepth: 3)
        try await dt.fit(features: embeddings, targets: targets)
        let dtPipelineData = try await CoreMLExporter.exportVisionClassifierPipeline(
            classifier: dt,
            inputImageName: "image",
            outputLabelName: "label",
            classLabels: nil,
            revision: .v2
        )
        try await withCompiledCoreML(dtPipelineData) { (compiledURL: URL) async throws in
            let model = try MLModel(contentsOf: compiledURL)
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "image": MLFeatureValue(pixelBuffer: buffers[0])
            ])
            let result = try await model.prediction(from: provider)
            let swiftPred = try await dt.predict(features: [embeddings[0]])[0]
            #expect(result.featureValue(for: "label")?.int64Value == Int64(swiftPred))
        }

        let binaryTargets: [Double] = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        let lr = LogisticRegression(device: .cpu)
        try await lr.fit(features: embeddings, targets: binaryTargets, learningRate: 0.2, epochs: 80)
        let lrPipelineData = try await CoreMLExporter.exportVisionClassifierPipeline(
            classifier: lr,
            inputImageName: "image",
            outputLabelName: "label",
            classLabels: ["neg", "pos"],
            revision: .v2
        )
        try await withCompiledCoreML(lrPipelineData) { (compiledURL: URL) async throws in
            let model = try MLModel(contentsOf: compiledURL)
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "image": MLFeatureValue(pixelBuffer: buffers[0])
            ])
            let result = try await model.prediction(from: provider)
            let swiftPred = try await lr.predict(features: [embeddings[0]])[0]
            let expectedStr = swiftPred == 1 ? "pos" : "neg"
            #expect(result.featureValue(for: "label")?.stringValue == expectedStr)
        }

        // 5. Binary MLPClassifier (sigmoid outDim == 1 expanded to 2-class softmax)
        let mlp2 = MLPClassifier(hiddenLayerSizes: [8], maxIter: 30, batchSize: 6, seed: 42, requestedDevice: .cpu)
        try await mlp2.fit(features: embeddings, targets: binaryTargets)
        let mlp2PipelineData = try await CoreMLExporter.exportVisionClassifierPipeline(
            classifier: mlp2,
            inputImageName: "image",
            outputLabelName: "label",
            classLabels: ["neg", "pos"],
            revision: .v2
        )
        let mlp2SwiftPreds = try await mlp2.predict(features: embeddings)
        let mlp2SwiftProbs = try await mlp2.predictProbability(features: embeddings)
        try await verifyVisionPipelineParity(
            modelData: mlp2PipelineData,
            buffers: buffers,
            expectedStringLabels: mlp2SwiftPreds.map { (cls: Int) -> String in cls == 1 ? "pos" : "neg" },
            expectedProbs: mlp2SwiftProbs,
            classNames: ["neg", "pos"],
            tolerance: 1e-2
        )
    }

    @Test("VisionFeaturePrint v1 pipeline (2048-D) compiles via MLModel.compileModel")
    func testVisionPipelineV1Compilation() async throws {
        let X = [[Double](repeating: 0.1, count: 2048), [Double](repeating: 0.9, count: 2048)]
        let y = [0.0, 1.0]
        let ovr = OneVsRestClassifier(numClasses: 2)
        try await ovr.fit(features: X, targets: y, learningRate: 0.1, epochs: 20)

        let data = try await CoreMLExporter.exportVisionClassifierPipeline(
            classifier: ovr,
            inputImageName: "image",
            outputLabelName: "label",
            classLabels: ["a", "b"],
            revision: .v1
        )
        try await withCompiledCoreML(data) { (compiledURL: URL) async throws in
            let model = try MLModel(contentsOf: compiledURL)
            let inputDesc = try #require(model.modelDescription.inputDescriptionsByName["image"])
            #expect(inputDesc.imageConstraint?.pixelsWide == 299)
            #expect(inputDesc.imageConstraint?.pixelsHigh == 299)
        }
    }

    // MARK: - Helpers

    private func verifyVisionPipelineParity(
        modelData: Data,
        buffers: [CVPixelBuffer],
        expectedStringLabels: [String],
        expectedProbs: [[Double]],
        classNames: [String],
        tolerance: Double
    ) async throws {
        try await withCompiledCoreML(modelData) { (compiledURL: URL) async throws in
            let model = try MLModel(contentsOf: compiledURL)
            let probName = model.modelDescription.predictedProbabilitiesName ?? "classProbability"

            for i in 0..<buffers.count {
                let provider = try MLDictionaryFeatureProvider(dictionary: [
                    "image": MLFeatureValue(pixelBuffer: buffers[i])
                ])
                let result = try await model.prediction(from: provider)
                let label = try #require(result.featureValue(for: "label")?.stringValue)
                #expect(label == expectedStringLabels[i])

                let probDict = try #require(result.featureValue(for: probName)?.dictionaryValue)
                for c in 0..<classNames.count {
                    let coremlProb = probDict[classNames[c]]?.doubleValue ?? -1.0
                    #expect(abs(coremlProb - expectedProbs[i][c]) < tolerance)
                }
            }
        }
    }

    private func extractVisionScenePrintV2(pixelBuffer: CVPixelBuffer) throws -> [Double] {
        let request = VNGenerateImageFeaturePrintRequest()
        if #available(macOS 14.0, *) {
            request.revision = VNGenerateImageFeaturePrintRequestRevision2
        }
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try handler.perform([request])
        let observation = try #require(request.results?.first as? VNFeaturePrintObservation)
        let count = observation.elementCount
        var values = [Double](repeating: 0.0, count: count)
        observation.data.withUnsafeBytes { (rawPtr: UnsafeRawBufferPointer) in
            let floats = rawPtr.bindMemory(to: Float.self)
            for idx in 0..<min(count, floats.count) {
                values[idx] = Double(floats[idx])
            }
        }
        return values
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
                let b: UInt8
                let g: UInt8
                let r: UInt8
                switch variant {
                case 0:
                    // High-frequency checkerboard
                    let block = (((x + phase) / 24) + ((y + phase) / 24)) % 2 == 0
                    r = block ? 240 : 20
                    g = block ? 30 : 200
                    b = block ? 30 : 220
                case 1:
                    // Horizontal stripes
                    let stripe = ((y + phase) / 18) % 2 == 0
                    r = stripe ? 30 : 230
                    g = stripe ? 235 : 40
                    b = stripe ? 60 : 210
                default:
                    // Smooth diagonal gradient
                    r = UInt8(min(255, ((x + phase) * 255) / max(1, width)))
                    g = UInt8(min(255, ((y + phase) * 255) / max(1, height)))
                    b = UInt8(220)
                }
                bytes[offset + 0] = b
                bytes[offset + 1] = g
                bytes[offset + 2] = r
                bytes[offset + 3] = 255
            }
        }
        return pixelBuffer
    }
}
