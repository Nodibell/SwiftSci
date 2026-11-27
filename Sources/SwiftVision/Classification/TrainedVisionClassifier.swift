import Foundation
@_exported import SwiftML
#if canImport(Vision)
import Vision
#endif
#if canImport(CoreGraphics)
import CoreGraphics
#endif
#if canImport(ImageIO)
import ImageIO
#endif

/// Classifier head backend configuration for ``TrainedVisionClassifier``.
///
/// Each backend pairs Apple Vision's `VNGenerateImageFeaturePrintRequest` (768-D ScenePrint v2
/// embedding on macOS 14+) with a supervised `SwiftML` estimator that supports Core ML
/// `PipelineClassifier` export.
public enum VisionClassifierBackend: Sendable, Equatable {
    /// Multi-class One-Vs-Rest logistic regression ensemble.
    ///
    /// - Parameters:
    ///   - learningRate: Gradient descent step size (default: `0.1`).
    ///   - maxIter: Maximum number of gradient descent epochs per binary sub-estimator (default: `500`).
    ///   - l2Penalty: Non-negative L2 regularization configuration parameter (default: `1e-4`).
    case oneVsRest(learningRate: Double = 0.1, maxIter: Int = 500, l2Penalty: Double = 1e-4)

    /// Binary logistic regression classifier (macOS only; requires exactly 2 distinct classes).
    ///
    /// - Parameters:
    ///   - learningRate: Gradient descent step size (default: `0.1`).
    ///   - maxIter: Maximum number of gradient descent epochs (default: `500`).
    case logisticRegression(learningRate: Double = 0.1, maxIter: Int = 500)

    /// Multi-Layer Perceptron neural network classifier trained with Adam optimization.
    ///
    /// - Parameters:
    ///   - hiddenLayers: Hidden layer widths (default: `[128]`).
    ///   - activation: Hidden-layer non-linear activation function (default: `.relu`).
    ///   - learningRate: Adam optimizer learning rate (default: `0.005`).
    ///   - maxIter: Maximum number of training epochs (default: `100`).
    ///   - batchSize: Mini-batch size (default: `32`).
    case mlp(
        hiddenLayers: [Int] = [128],
        activation: ActivationFunction = .relu,
        learningRate: Double = 0.005,
        maxIter: Int = 100,
        batchSize: Int = 32
    )

    /// Random Forest ensemble classifier.
    ///
    /// - Parameters:
    ///   - nEstimators: Number of decision trees in the forest (default: `100`).
    ///   - maxDepth: Maximum depth of each tree (default: `10`).
    ///   - seed: Deterministic PRNG seed for bootstrapping and feature subsampling (default: `42`).
    case randomForest(nEstimators: Int = 100, maxDepth: Int = 10, seed: Int = 42)

    /// Single CART Decision Tree classifier.
    ///
    /// - Parameters:
    ///   - maxDepth: Maximum tree depth (default: `8`).
    case decisionTree(maxDepth: Int = 8)
}

/// Structured prediction returned by ``TrainedVisionClassifier/predictProbability(imageURL:)``
/// and ``TrainedVisionClassifier/predictProbability(image:)``.
public struct VisionClassificationPrediction: Sendable, Equatable {
    /// Predicted class label with the highest probability.
    public let label: String
    /// Probability confidence score in `[0.0, 1.0]` associated with ``label``.
    public let confidence: Double
    /// Class-to-probability distribution across all trained classes.
    public let probabilities: [String: Double]

    /// Creates a structured image classification prediction.
    ///
    /// - Parameters:
    ///   - label: Top predicted class label.
    ///   - confidence: Confidence score for `label`.
    ///   - probabilities: Probability dictionary keyed by class label.
    public init(label: String, confidence: Double, probabilities: [String: Double]) {
        self.label = label
        self.confidence = confidence
        self.probabilities = probabilities
    }
}

/// High-level trainable image classifier combining Apple Vision feature prints with
/// supervised `SwiftML` classifiers and native Core ML `PipelineClassifier` export.
///
/// ``TrainedVisionClassifier`` extracts native-resolution `VNGenerateImageFeaturePrintRequest`
/// embeddings (`768`-dimensional ScenePrint v2 on macOS 14+) and trains the configured
/// ``VisionClassifierBackend``. On macOS, fitted models can be exported directly as a
/// standalone Core ML `.mlmodel` `PipelineClassifier` combining `VisionFeaturePrint.Scene`
/// and the fitted classifier head.
public actor TrainedVisionClassifier {
    /// Configured classifier head backend and hyperparameters.
    public let backend: VisionClassifierBackend

    /// Sorted class labels discovered during the most recent successful ``fit(folderURL:)``
    /// or ``fit(imageURLs:labels:)`` call.
    public private(set) var classes: [String] = []

    /// Feature-print embedding dimensionality observed during training (typically `768` for ScenePrint v2),
    /// or `nil` before fitting.
    public private(set) var featureDimension: Int?

    private enum FittedEstimator: Sendable {
        case oneVsRest(OneVsRestClassifier)
        #if os(macOS)
        case logisticRegression(LogisticRegression)
        #endif
        case mlp(MLPClassifier)
        case randomForest(RandomForestClassifier)
        case decisionTree(DecisionTreeClassifier)

        var classifier: any ClassifierEstimator {
            switch self {
            case .oneVsRest(let model): return model
            #if os(macOS)
            case .logisticRegression(let model): return model
            #endif
            case .mlp(let model): return model
            case .randomForest(let model): return model
            case .decisionTree(let model): return model
            }
        }
    }

    private var fittedEstimator: FittedEstimator?

    /// Creates an unfitted trainable Vision image classifier.
    ///
    /// - Parameter backend: Classifier head configuration (defaults to ``VisionClassifierBackend/oneVsRest(learningRate:maxIter:l2Penalty:)``).
    public init(backend: VisionClassifierBackend = .oneVsRest()) {
        self.backend = backend
    }

    // MARK: - Training

    /// Trains the classifier from a flat `<class>/<image>` directory or a split dataset directory
    /// containing a `train` (or `training`) subdirectory of `<class>/<image>` folders.
    ///
    /// Feature prints are extracted directly from the original image files at native resolution
    /// via `VNGenerateImageFeaturePrintRequest` so embeddings align with Core ML `VisionFeaturePrint.Scene` v2.
    ///
    /// - Parameter folderURL: Root dataset folder URL.
    /// - Throws: ``VisionError/invalidInput(_:)`` if the folder does not exist or a split layout lacks `train`,
    ///   ``SwiftMLError/emptyInput`` if no training images are found,
    ///   or ``SwiftMLError/invalidParameter(_:)`` if fewer than 2 classes are present.
    public func fit(folderURL: URL) async throws {
        let (imageURLs, labels) = try Self.collectTrainingURLsAndLabels(from: folderURL)
        try await fit(imageURLs: imageURLs, labels: labels)
    }

    /// Trains the classifier from local image file URLs and matching string class labels.
    ///
    /// - Parameters:
    ///   - imageURLs: Array of local image file URLs.
    ///   - labels: Class label for each image, matching `imageURLs` one-to-one.
    /// - Throws: ``SwiftMLError/emptyInput`` if `imageURLs` is empty,
    ///   ``SwiftMLError/dimensionMismatch(expected:got:)`` if `imageURLs.count != labels.count`,
    ///   ``SwiftMLError/invalidParameter(_:)`` if fewer than 2 unique classes are supplied or hyperparameters are invalid,
    ///   or ``VisionError/invalidInput(_:)`` if any image cannot be processed.
    public func fit(imageURLs: [URL], labels: [String]) async throws {
        guard !imageURLs.isEmpty else {
            throw SwiftMLError.emptyInput
        }
        guard imageURLs.count == labels.count else {
            throw SwiftMLError.dimensionMismatch(expected: imageURLs.count, got: labels.count)
        }
        try Self.validateBackendConfiguration(backend)

        var embeddings: [[Double]] = []
        embeddings.reserveCapacity(imageURLs.count)
        for url in imageURLs {
            let vector = try Self.extractFeaturePrint(imageURL: url)
            embeddings.append(vector)
        }

        try await fitFromEmbeddings(embeddings, labels: labels)
    }

    #if canImport(CoreGraphics)
    /// Trains the classifier from in-memory `CGImage` instances and matching string class labels.
    ///
    /// - Parameters:
    ///   - images: Array of `CGImage` instances.
    ///   - labels: Class label for each image, matching `images` one-to-one.
    /// - Throws: ``SwiftMLError/emptyInput`` if `images` is empty,
    ///   ``SwiftMLError/dimensionMismatch(expected:got:)`` if `images.count != labels.count`,
    ///   ``SwiftMLError/invalidParameter(_:)`` if fewer than 2 unique classes are supplied or hyperparameters are invalid,
    ///   or ``VisionError/invalidInput(_:)`` if feature extraction fails.
    public func fit(images: [CGImage], labels: [String]) async throws {
        guard !images.isEmpty else {
            throw SwiftMLError.emptyInput
        }
        guard images.count == labels.count else {
            throw SwiftMLError.dimensionMismatch(expected: images.count, got: labels.count)
        }
        try Self.validateBackendConfiguration(backend)

        var embeddings: [[Double]] = []
        embeddings.reserveCapacity(images.count)
        for cgImage in images {
            let vector = try Self.extractFeaturePrint(cgImage: cgImage)
            embeddings.append(vector)
        }

        try await fitFromEmbeddings(embeddings, labels: labels)
    }
    #endif

    // MARK: - Inference

    /// Predicts the top class label for an image file on disk.
    ///
    /// - Parameter imageURL: Local image file URL.
    /// - Throws: ``SwiftMLError/modelNotFitted`` if the classifier has not been trained,
    ///   or ``VisionError/invalidInput(_:)`` if feature extraction fails.
    /// - Returns: Predicted class label string.
    public func predict(imageURL: URL) async throws -> String {
        let prediction = try await predictProbability(imageURL: imageURL)
        return prediction.label
    }

    /// Predicts full class probabilities and top label for an image file on disk.
    ///
    /// - Parameter imageURL: Local image file URL.
    /// - Throws: ``SwiftMLError/modelNotFitted`` if the classifier has not been trained,
    ///   ``SwiftMLError/dimensionMismatch(expected:got:)`` if the extracted feature vector size mismatches the fitted model,
    ///   or ``VisionError/invalidInput(_:)`` if feature extraction fails.
    /// - Returns: A ``VisionClassificationPrediction`` containing top label, confidence, and per-class probabilities.
    public func predictProbability(imageURL: URL) async throws -> VisionClassificationPrediction {
        guard fittedEstimator != nil, !classes.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        let embedding = try Self.extractFeaturePrint(imageURL: imageURL)
        return try await predictFromEmbedding(embedding)
    }

    #if canImport(CoreGraphics)
    /// Predicts the top class label for an in-memory `CGImage`.
    ///
    /// - Parameter image: Input `CGImage`.
    /// - Throws: ``SwiftMLError/modelNotFitted`` if the classifier has not been trained,
    ///   or ``VisionError/invalidInput(_:)`` if feature extraction fails.
    /// - Returns: Predicted class label string.
    public func predict(image: CGImage) async throws -> String {
        let prediction = try await predictProbability(image: image)
        return prediction.label
    }

    /// Predicts full class probabilities and top label for an in-memory `CGImage`.
    ///
    /// - Parameter image: Input `CGImage`.
    /// - Throws: ``SwiftMLError/modelNotFitted`` if the classifier has not been trained,
    ///   ``SwiftMLError/dimensionMismatch(expected:got:)`` if the extracted feature vector size mismatches the fitted model,
    ///   or ``VisionError/invalidInput(_:)`` if feature extraction fails.
    /// - Returns: A ``VisionClassificationPrediction`` containing top label, confidence, and per-class probabilities.
    public func predictProbability(image: CGImage) async throws -> VisionClassificationPrediction {
        guard fittedEstimator != nil, !classes.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        let embedding = try Self.extractFeaturePrint(cgImage: image)
        return try await predictFromEmbedding(embedding)
    }
    #endif

    // MARK: - Core ML Pipeline Export

    #if os(macOS)
    /// Exports the fitted Vision feature-print + classifier head pipeline as binary `.mlmodel` data.
    ///
    /// Uses `VisionScenePrintRevision.v1` (`299x299 -> 2048-D`) when ``featureDimension`` is `2048`
    /// and `VisionScenePrintRevision.v2` (`360x360 -> 768-D`) otherwise, embedding ``classes`` as
    /// string class labels in the resulting Core ML `PipelineClassifier`.
    ///
    /// - Parameters:
    ///   - inputImageName: Input image feature name in the Core ML model (default: `"image"`).
    ///   - outputLabelName: Predicted output class label feature name (default: `"label"`).
    /// - Throws: ``SwiftMLError/modelNotFitted`` if the classifier has not been trained,
    ///   or ``SwiftMLError/exportFailed(_:)`` / ``SwiftMLError/unsupportedOperation(_:)`` on export failure.
    /// - Returns: Binary `.mlmodel` payload ready to write to disk or compile with `MLModel.compileModel(at:)`.
    public func exportCoreMLPipelineData(
        inputImageName: String = "image",
        outputLabelName: String = "label"
    ) async throws -> Data {
        guard let estimator = fittedEstimator, let dim = featureDimension, !classes.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        let revision: VisionScenePrintRevision = (dim == 2048) ? .v1 : .v2
        switch estimator {
        case .oneVsRest(let model):
            return try await CoreMLExporter.exportVisionClassifierPipeline(
                classifier: model,
                inputImageName: inputImageName,
                outputLabelName: outputLabelName,
                classLabels: classes,
                revision: revision
            )
        case .logisticRegression(let model):
            return try await CoreMLExporter.exportVisionClassifierPipeline(
                classifier: model,
                inputImageName: inputImageName,
                outputLabelName: outputLabelName,
                classLabels: classes,
                revision: revision
            )
        case .mlp(let model):
            return try await CoreMLExporter.exportVisionClassifierPipeline(
                classifier: model,
                inputImageName: inputImageName,
                outputLabelName: outputLabelName,
                classLabels: classes,
                revision: revision
            )
        case .randomForest(let model):
            return try await CoreMLExporter.exportVisionClassifierPipeline(
                classifier: model,
                inputImageName: inputImageName,
                outputLabelName: outputLabelName,
                classLabels: classes,
                revision: revision
            )
        case .decisionTree(let model):
            return try await CoreMLExporter.exportVisionClassifierPipeline(
                classifier: model,
                inputImageName: inputImageName,
                outputLabelName: outputLabelName,
                classLabels: classes,
                revision: revision
            )
        }
    }

    /// Exports and writes the fitted Vision feature-print + classifier head pipeline as a `.mlmodel` file.
    ///
    /// - Parameters:
    ///   - url: Destination file URL (`.mlmodel` extension recommended).
    ///   - inputImageName: Input image feature name in the Core ML model (default: `"image"`).
    ///   - outputLabelName: Predicted output class label feature name (default: `"label"`).
    /// - Throws: ``SwiftMLError/modelNotFitted`` if the classifier has not been trained,
    ///   or ``SwiftMLError/exportFailed(_:)`` if serialization or file writing fails.
    public func exportCoreMLPipeline(
        to url: URL,
        inputImageName: String = "image",
        outputLabelName: String = "label"
    ) async throws {
        let data = try await exportCoreMLPipelineData(
            inputImageName: inputImageName,
            outputLabelName: outputLabelName
        )
        do {
            try data.write(to: url, options: .atomic)
        } catch {
            throw SwiftMLError.exportFailed(
                "Failed to write Core ML Vision pipeline to \(url.path): \(error.localizedDescription)"
            )
        }
    }
    #endif

    // MARK: - Private Training & Inference Helpers

    private func fitFromEmbeddings(_ embeddings: [[Double]], labels: [String]) async throws {
        guard let first = embeddings.first, !first.isEmpty else {
            throw SwiftMLError.emptyInput
        }
        let dim = first.count
        for row in embeddings {
            guard row.count == dim else {
                throw SwiftMLError.dimensionMismatch(expected: dim, got: row.count)
            }
            guard row.allSatisfy(\.isFinite) else {
                throw SwiftMLError.invalidParameter("Extracted feature print vectors must be finite.")
            }
        }

        let sortedClasses = Array(Set(labels)).sorted()
        guard sortedClasses.count >= 2 else {
            throw SwiftMLError.invalidParameter(
                "TrainedVisionClassifier requires at least 2 distinct classes, got \(sortedClasses.count)."
            )
        }

        var labelToIndex: [String: Double] = [:]
        labelToIndex.reserveCapacity(sortedClasses.count)
        for (idx, className) in sortedClasses.enumerated() {
            labelToIndex[className] = Double(idx)
        }

        let targets: [Double] = labels.map { (label: String) -> Double in
            labelToIndex[label] ?? 0.0
        }

        let trainedEstimator = try await trainBackend(
            features: embeddings,
            targets: targets,
            numClasses: sortedClasses.count
        )

        self.fittedEstimator = trainedEstimator
        self.classes = sortedClasses
        self.featureDimension = dim
    }

    private func trainBackend(
        features: [[Double]],
        targets: [Double],
        numClasses: Int
    ) async throws -> FittedEstimator {
        switch backend {
        case let .oneVsRest(learningRate, maxIter, _):
            let ovr = OneVsRestClassifier(numClasses: numClasses)
            try await ovr.fit(
                features: features,
                targets: targets,
                learningRate: Float(learningRate),
                epochs: maxIter
            )
            return .oneVsRest(ovr)

        case let .logisticRegression(learningRate, maxIter):
            guard numClasses == 2 else {
                throw SwiftMLError.invalidParameter(
                    "LogisticRegression backend requires exactly 2 classes, got \(numClasses). Use .oneVsRest for multiclass tasks."
                )
            }
            #if os(macOS)
            let lr = LogisticRegression(device: .auto)
            try await lr.fit(
                features: features,
                targets: targets,
                learningRate: Float(learningRate),
                epochs: maxIter
            )
            return .logisticRegression(lr)
            #else
            throw SwiftMLError.unsupportedOperation("LogisticRegression backend is only supported on macOS.")
            #endif

        case let .mlp(hiddenLayers, activation, learningRate, maxIter, batchSize):
            let mlp = MLPClassifier(
                hiddenLayerSizes: hiddenLayers,
                activation: activation,
                solver: .adam,
                maxIter: maxIter,
                learningRate: learningRate,
                batchSize: batchSize,
                seed: 42,
                requestedDevice: .auto
            )
            try await mlp.fit(features: features, targets: targets)
            return .mlp(mlp)

        case let .randomForest(nEstimators, maxDepth, seed):
            let rf = RandomForestClassifier(
                nEstimators: nEstimators,
                maxDepth: maxDepth,
                randomState: seed
            )
            try await rf.fit(features: features, targets: targets)
            return .randomForest(rf)

        case let .decisionTree(maxDepth):
            let dt = DecisionTreeClassifier(maxDepth: maxDepth)
            try await dt.fit(features: features, targets: targets)
            return .decisionTree(dt)
        }
    }

    private func predictFromEmbedding(_ embedding: [Double]) async throws -> VisionClassificationPrediction {
        guard let estimator = fittedEstimator, !classes.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        if let expectedDim = featureDimension, embedding.count != expectedDim {
            throw SwiftMLError.dimensionMismatch(expected: expectedDim, got: embedding.count)
        }

        let probRows = try await estimator.classifier.predictProbability(features: [embedding])
        guard let row = probRows.first, !row.isEmpty else {
            throw SwiftMLError.trainingFailed("Classifier produced empty probability output.")
        }

        var probabilities: [String: Double] = [:]
        probabilities.reserveCapacity(classes.count)
        var bestIdx = 0
        var bestProb = -Double.infinity

        for idx in 0..<classes.count {
            let p = idx < row.count ? row[idx] : 0.0
            probabilities[classes[idx]] = p
            if p > bestProb {
                bestProb = p
                bestIdx = idx
            }
        }

        let predictedLabel = classes[bestIdx]
        return VisionClassificationPrediction(
            label: predictedLabel,
            confidence: max(0.0, bestProb),
            probabilities: probabilities
        )
    }

    private static func validateBackendConfiguration(_ backend: VisionClassifierBackend) throws {
        switch backend {
        case let .oneVsRest(learningRate, maxIter, l2Penalty):
            guard learningRate.isFinite, learningRate > 0.0 else {
                throw SwiftMLError.invalidParameter("learningRate must be finite and positive.")
            }
            guard maxIter >= 1 else {
                throw SwiftMLError.invalidParameter("maxIter must be at least 1.")
            }
            guard l2Penalty.isFinite, l2Penalty >= 0.0 else {
                throw SwiftMLError.invalidParameter("l2Penalty must be finite and non-negative.")
            }

        case let .logisticRegression(learningRate, maxIter):
            guard learningRate.isFinite, learningRate > 0.0 else {
                throw SwiftMLError.invalidParameter("learningRate must be finite and positive.")
            }
            guard maxIter >= 1 else {
                throw SwiftMLError.invalidParameter("maxIter must be at least 1.")
            }

        case let .mlp(hiddenLayers, _, learningRate, maxIter, batchSize):
            guard hiddenLayers.allSatisfy({ (width: Int) -> Bool in width > 0 }) else {
                throw SwiftMLError.invalidParameter("All MLP hidden layer widths must be positive.")
            }
            guard learningRate.isFinite, learningRate > 0.0 else {
                throw SwiftMLError.invalidParameter("learningRate must be finite and positive.")
            }
            guard maxIter >= 1 else {
                throw SwiftMLError.invalidParameter("maxIter must be at least 1.")
            }
            guard batchSize >= 1 else {
                throw SwiftMLError.invalidParameter("batchSize must be at least 1.")
            }

        case let .randomForest(nEstimators, maxDepth, _):
            guard nEstimators >= 1 else {
                throw SwiftMLError.invalidParameter("nEstimators must be at least 1.")
            }
            guard maxDepth >= 1 else {
                throw SwiftMLError.invalidParameter("maxDepth must be at least 1.")
            }

        case let .decisionTree(maxDepth):
            guard maxDepth >= 1 else {
                throw SwiftMLError.invalidParameter("maxDepth must be at least 1.")
            }
        }
    }

    // MARK: - Private Vision FeaturePrint Helpers

    private static func extractFeaturePrint(imageURL: URL) throws -> [Double] {
        #if canImport(Vision)
        let request = VNGenerateImageFeaturePrintRequest()
        if #available(macOS 14.0, *) {
            request.revision = VNGenerateImageFeaturePrintRequestRevision2
        }
        let handler = VNImageRequestHandler(url: imageURL, options: [:])
        do {
            try handler.perform([request])
        } catch {
            throw VisionError.invalidInput(
                "Feature print extraction failed for '\(imageURL.lastPathComponent)': \(error.localizedDescription)"
            )
        }
        guard let observation = request.results?.first as? VNFeaturePrintObservation else {
            throw VisionError.invalidInput(
                "No feature print observation produced for '\(imageURL.lastPathComponent)'."
            )
        }
        return try extractDoubles(from: observation)
        #elseif canImport(CoreGraphics) && canImport(ImageIO)
        let dataset = try ImageDataset.load(from: imageURL)
        let floats = VisionImageClassifier().extractFeaturePrint(dataset: dataset)
        return floats.map { (value: Float) -> Double in Double(value) }
        #else
        throw VisionError.notImplemented("Image feature print extraction requires Vision or CoreGraphics.")
        #endif
    }

    #if canImport(CoreGraphics)
    private static func extractFeaturePrint(cgImage: CGImage) throws -> [Double] {
        guard cgImage.width > 0, cgImage.height > 0 else {
            throw VisionError.invalidInput("Image dimensions must be non-zero.")
        }
        #if canImport(Vision)
        let request = VNGenerateImageFeaturePrintRequest()
        if #available(macOS 14.0, *) {
            request.revision = VNGenerateImageFeaturePrintRequestRevision2
        }
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            throw VisionError.invalidInput(
                "Feature print extraction failed for CGImage: \(error.localizedDescription)"
            )
        }
        guard let observation = request.results?.first as? VNFeaturePrintObservation else {
            throw VisionError.invalidInput("No feature print observation produced for CGImage.")
        }
        return try extractDoubles(from: observation)
        #elseif canImport(ImageIO)
        let dataset = try ImageDataset.from(cgImage: cgImage)
        let floats = VisionImageClassifier().extractFeaturePrint(dataset: dataset)
        return floats.map { (value: Float) -> Double in Double(value) }
        #else
        throw VisionError.notImplemented("Image feature print extraction requires Vision or ImageIO.")
        #endif
    }
    #endif

    #if canImport(Vision)
    private static func extractDoubles(from observation: VNFeaturePrintObservation) throws -> [Double] {
        let count = observation.elementCount
        guard count > 0 else {
            throw VisionError.invalidInput("Feature print observation produced zero elements.")
        }
        switch observation.elementType {
        case .float:
            var values = [Double](repeating: 0.0, count: count)
            observation.data.withUnsafeBytes { (rawPtr: UnsafeRawBufferPointer) in
                let floats = rawPtr.bindMemory(to: Float.self)
                let limit = min(count, floats.count)
                for idx in 0..<limit {
                    values[idx] = Double(floats[idx])
                }
            }
            return values
        case .double:
            var values = [Double](repeating: 0.0, count: count)
            observation.data.withUnsafeBytes { (rawPtr: UnsafeRawBufferPointer) in
                let doubles = rawPtr.bindMemory(to: Double.self)
                let limit = min(count, doubles.count)
                for idx in 0..<limit {
                    values[idx] = doubles[idx]
                }
            }
            return values
        case .unknown:
            throw VisionError.invalidInput("Feature print observation has unknown element type.")
        @unknown default:
            throw VisionError.invalidInput("Unsupported feature print element type.")
        }
    }
    #endif

    // MARK: - Private Folder Resolution Helpers

    private static func collectTrainingURLsAndLabels(from folderURL: URL) throws -> (urls: [URL], labels: [String]) {
        let resolvedRoot = folderURL.resolvingSymlinksInPath()
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: resolvedRoot.path, isDirectory: &isDir), isDir.boolValue else {
            throw VisionError.invalidInput("Directory not found: \(folderURL.path)")
        }

        let topSubdirs = try immediateSubdirectories(in: resolvedRoot)
        let trainAliases = ["train", "training"]
        let valAliases = ["val", "valid", "validation"]
        let testAliases = ["test", "testing"]

        let trainDir = resolveSplitDirectory(from: topSubdirs, aliases: trainAliases)
        let valDir = resolveSplitDirectory(from: topSubdirs, aliases: valAliases)
        let testDir = resolveSplitDirectory(from: topSubdirs, aliases: testAliases)

        let hasAnySplitDir = (trainDir != nil || valDir != nil || testDir != nil)
        let trainingRoot: URL
        if hasAnySplitDir {
            guard let resolvedTrainDir = trainDir else {
                throw VisionError.invalidInput(
                    "Split dataset folder is missing a 'train' or 'training' subdirectory: \(folderURL.path)"
                )
            }
            trainingRoot = resolvedTrainDir
        } else {
            trainingRoot = resolvedRoot
        }

        let classDirs = try immediateSubdirectories(in: trainingRoot)
        var imageURLs: [URL] = []
        var labels: [String] = []

        for classDir in classDirs {
            let className = classDir.lastPathComponent
            let classImages = try collectSortedImageURLs(in: classDir)
            for imgURL in classImages {
                imageURLs.append(imgURL)
                labels.append(className)
            }
        }

        guard !imageURLs.isEmpty else {
            throw SwiftMLError.emptyInput
        }

        return (imageURLs, labels)
    }

    private static func immediateSubdirectories(in directoryURL: URL) throws -> [URL] {
        let contents = try FileManager.default.contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )
        var dirs: [URL] = []
        for item in contents {
            let values = try item.resourceValues(forKeys: [.isDirectoryKey])
            if values.isDirectory == true {
                dirs.append(item)
            }
        }
        dirs.sort { (lhs: URL, rhs: URL) -> Bool in
            lhs.lastPathComponent < rhs.lastPathComponent
        }
        return dirs
    }

    private static func resolveSplitDirectory(from subdirectories: [URL], aliases: [String]) -> URL? {
        for alias in aliases {
            if let match = subdirectories.first(where: { (dir: URL) -> Bool in
                dir.lastPathComponent.lowercased() == alias
            }) {
                return match
            }
        }
        return nil
    }

    private static func collectSortedImageURLs(in directoryURL: URL) throws -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: directoryURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw VisionError.invalidInput("Failed to enumerate directory at: \(directoryURL.path)")
        }

        var urls: [URL] = []
        while let fileURL = enumerator.nextObject() as? URL {
            let ext = fileURL.pathExtension.lowercased()
            if VisionImageClassifier.supportedExtensions.contains(ext) {
                urls.append(fileURL)
            }
        }

        urls.sort { (lhs: URL, rhs: URL) -> Bool in
            lhs.standardizedFileURL.path < rhs.standardizedFileURL.path
        }
        return urls
    }
}
