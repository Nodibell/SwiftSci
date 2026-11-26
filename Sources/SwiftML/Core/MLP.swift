import Foundation
import Accelerate
import SwiftPreprocessing
import SwiftDataFrame

/// Activation functions supported by Multi-Layer Perceptrons.
public enum ActivationFunction: String, Sendable, Codable {
    case relu
    case sigmoid
    case tanh
}

/// Optimization solver algorithms for Multi-Layer Perceptrons.
public enum MLPSolver: String, Sendable, Codable {
    case adam
    case sgd
}

/// A flat row-major layer weight representation for high-performance matrix operations.
public struct LayerWeights: Sendable, Codable {
    /// The w.
    public var W: [Double]   // row-major flat buffer: inDim x outDim
    /// The b.
    public var b: [Double]   // outDim
    /// The in dim.
    public let inDim: Int
    /// The out dim.
    public let outDim: Int

    /// Creates a new instance.
    /// - Parameters:
    ///   - W: The w.
    ///   - b: The b.
    ///   - inDim: The in dim.
    ///   - outDim: The out dim.
    public init(W: [Double], b: [Double], inDim: Int, outDim: Int) {
        self.W = W
        self.b = b
        self.inDim = inDim
        self.outDim = outDim
    }
}

private final class LayerAdamState {
    var mW: [Double]
    var vW: [Double]
    var mB: [Double]
    var vB: [Double]
    var t: Int = 0

    init(inDim: Int, outDim: Int) {
        self.mW = [Double](repeating: 0.0, count: inDim * outDim)
        self.vW = [Double](repeating: 0.0, count: inDim * outDim)
        self.mB = [Double](repeating: 0.0, count: outDim)
        self.vB = [Double](repeating: 0.0, count: outDim)
        self.t = 0
    }
}

/// Multi-Layer Perceptron Classifier.
public actor MLPClassifier: ClassifierEstimator {
    /// The hidden layer sizes.
    public let hiddenLayerSizes: [Int]
    /// The activation.
    public let activation: ActivationFunction
    /// The solver.
    public let solver: MLPSolver
    /// The max iter.
    public let maxIter: Int
    /// The learning rate.
    public let learningRate: Double
    /// The beta1.
    public let beta1: Double
    /// The beta2.
    public let beta2: Double
    /// The epsilon.
    public let epsilon: Double
    /// The batch size.
    public let batchSize: Int
    /// The seed.
    public let seed: Int
    /// The requested device.
    public let requestedDevice: ExecutionDevice
    /// The resolved device.
    public private(set) var resolvedDevice: ExecutionDevice = .cpu

    private var layers: [LayerWeights]?
    private var classes: [Double]?

    /// The trained layer weights and biases (nil if not fitted).
    public var trainedLayers: [LayerWeights]? { layers }
    /// The trained class labels (nil if not fitted).
    public var trainedClasses: [Double]? { classes }

    /// Creates a new instance.
    /// - Parameters:
    ///   - hiddenLayerSizes: The hidden layer sizes.
    ///   - activation: The activation.
    ///   - solver: The solver.
    ///   - maxIter: The max iter.
    ///   - learningRate: The learning rate.
    ///   - beta1: The beta1.
    ///   - beta2: The beta2.
    ///   - epsilon: The epsilon.
    ///   - batchSize: The batch size.
    ///   - seed: The seed.
    ///   - requestedDevice: The requested device.
    public init(
        hiddenLayerSizes: [Int] = [100],
        activation: ActivationFunction = .relu,
        solver: MLPSolver = .adam,
        maxIter: Int = 200,
        learningRate: Double = 1e-3,
        beta1: Double = 0.9,
        beta2: Double = 0.999,
        epsilon: Double = 1e-8,
        batchSize: Int = 32,
        seed: Int = 42,
        requestedDevice: ExecutionDevice = .auto
    ) {
        self.hiddenLayerSizes = hiddenLayerSizes
        self.activation = activation
        self.solver = solver
        self.maxIter = maxIter
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batchSize = batchSize
        self.seed = seed
        self.requestedDevice = requestedDevice
    }

    /// Fits the multi-layer perceptron classifier on training data.
    /// - Parameters:
    ///   - features: The features matrix.
    ///   - targets: The target labels.
    /// - Throws: An error if inputs are invalid.
    public func fit(features: [[Double]], targets: [Double]) async throws {
        try await fit(features: features, targets: targets, validationFeatures: nil, validationTargets: nil, earlyStopping: nil)
    }

    /// Fits the multi-layer perceptron classifier with optional validation monitoring and early stopping.
    ///
    /// - Parameters:
    ///   - features: 2D feature matrix of shape `[N, P]`.
    ///   - targets: 1D class labels of length `N`.
    ///   - validationFeatures: Optional validation features matrix.
    ///   - validationTargets: Optional validation targets array.
    ///   - earlyStopping: Optional `EarlyStopping` callback.
    /// - Throws: `SwiftMLError` if inputs are invalid.
    ///
    /// ## Thread Safety
    /// Thread-safe via actor isolation.
    public func fit(
        features: [[Double]],
        targets: [Double],
        validationFeatures: [[Double]]? = nil,
        validationTargets: [Double]? = nil,
        earlyStopping: EarlyStopping? = nil
    ) async throws {
        guard !features.isEmpty, !features[0].isEmpty, !targets.isEmpty else {
            throw SwiftMLError.emptyInput
        }
        let numSamples = features.count
        let numFeatures = try validateMLPTrainingInputs(
            features: features,
            targets: targets,
            validationFeatures: validationFeatures,
            validationTargets: validationTargets
        )

        self.resolvedDevice = await HardwareRouter.shared.resolveDevice(
            for: "MLPClassifier",
            sampleCount: numSamples,
            featureCount: numFeatures,
            requestedDevice: requestedDevice
        )

        let uniqueClasses = Array(Set(targets)).sorted()
        self.classes = uniqueClasses
        let numClasses = max(1, uniqueClasses.count)
        let lossKind: MLPLossKind = numClasses <= 2
            ? .binaryCrossEntropy(positiveClass: uniqueClasses.last ?? 1.0)
            : .softmaxCrossEntropy(classes: uniqueClasses)

        let layerSizes = [numFeatures] + hiddenLayerSizes + [numClasses > 2 ? numClasses : 1]
        var rng = SeededRandom(seed: seed)

        var layers = [LayerWeights]()
        var adamStates = [LayerAdamState]()

        for l in 0..<(layerSizes.count - 1) {
            let inDim = layerSizes[l]
            let outDim = layerSizes[l + 1]
            let limit = sqrt(6.0 / Double(inDim + outDim))
            var wFlat = [Double]()
            wFlat.reserveCapacity(inDim * outDim)
            for _ in 0..<(inDim * outDim) {
                wFlat.append(rng.nextDouble() * 2.0 * limit - limit)
            }
            let bFlat = [Double](repeating: 0.0, count: outDim)
            layers.append(LayerWeights(W: wFlat, b: bFlat, inDim: inDim, outDim: outDim))
            adamStates.append(LayerAdamState(inDim: inDim, outDim: outDim))
        }

        layers = mlpRunMiniBatchTraining(
            initialLayers: layers,
            adamStates: &adamStates,
            features: features,
            targets: targets,
            validationFeatures: validationFeatures,
            validationTargets: validationTargets,
            earlyStopping: earlyStopping,
            activation: activation,
            solver: solver,
            maxIter: maxIter,
            learningRate: learningRate,
            beta1: beta1,
            beta2: beta2,
            epsilon: epsilon,
            batchSize: batchSize,
            lossKind: lossKind,
            rng: &rng
        )

        self.layers = layers
    }

    /// Predict.
    /// - Parameters:
    ///   - features: The features.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[Int]` result.
    public func predict(features: [[Double]]) async throws -> [Int] {
        let probs = try await predictProbability(features: features)
        guard let classes = self.classes else { return [] }
        if classes.count <= 2 {
            return probs.map { $0[1] >= 0.5 ? Int(classes.last ?? 1.0) : Int(classes.first ?? 0.0) }
        } else {
            return probs.map { p in
                let maxIdx = p.argmax()
                return Int(classes[maxIdx])
            }
        }
    }

    /// Predicts labels from a prepared matrix in either storage order.
    /// Columns must follow the positional feature order used during fitting.
    /// Uses the same CPU Double-precision calculation as row-array prediction.
    public func predict(features: PreparedNumericMatrix) async throws -> [Int] {
        let probabilities = try await predictProbability(features: features)
        guard let classes = classes else { return [] }
        if classes.count <= 2 {
            return probabilities.map { $0[1] >= 0.5 ? Int(classes.last ?? 1) : Int(classes.first ?? 0) }
        }
        return probabilities.map { Int(classes[$0.argmax()]) }
    }

    /// Predicts class probabilities without materializing a complete row-array input.
    /// Columns are positional; names do not reorder the model's features.
    /// Empty input returns an empty result after the fitted-model check.
    /// Throws if the model is not fitted or a nonempty matrix has the wrong width.
    public func predictProbability(features: PreparedNumericMatrix) async throws -> [[Double]] {
        guard let layers = layers, let classes = classes else { throw SwiftMLError.modelNotFitted }
        guard features.rowCount > 0 else { return [] }
        try mlpValidateMatrix(features, layers: layers)
        if let batched = mlpBatchedPrediction(matrix: features, layers: layers,
            activate: { applyActivation($0, activation: activation) },
            finish: { mlpProbabilities($0, classCount: classes.count) }) { return batched }
        var results: [[Double]] = []
        results.reserveCapacity(features.rowCount)
        for row in 0..<features.rowCount {
            let logits = mlpPredictRow(mlpMatrixRow(features, row: row), layers: layers,
                activate: { applyActivation($0, activation: activation) })
            results.append(mlpProbabilities(logits, classCount: classes.count))
        }
        return results
    }

    /// Predict probability.
    /// - Parameters:
    ///   - features: The features.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[[Double]]` result.
    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        guard let layers = layers, let classes = classes else {
            throw SwiftMLError.modelNotFitted
        }
        guard !features.isEmpty else { return [] }

        if let batched = try mlpBatchedPrediction(features: features, layers: layers,
            activate: { applyActivation($0, activation: activation) },
            finish: { mlpProbabilities($0, classCount: classes.count) }) { return batched }

        var results = [[Double]]()
        results.reserveCapacity(features.count)

        for x in features {
            let curr = try mlpPredictRowValidated(x, layers: layers,
                activate: { applyActivation($0, activation: activation) })
            results.append(mlpProbabilities(curr, classCount: classes.count))
        }

        return results
    }
}

/// Multi-Layer Perceptron Regressor.
public actor MLPRegressor: RegressorEstimator {
    /// The hidden layer sizes.
    public let hiddenLayerSizes: [Int]
    /// The activation.
    public let activation: ActivationFunction
    /// The solver.
    public let solver: MLPSolver
    /// The max iter.
    public let maxIter: Int
    /// The learning rate.
    public let learningRate: Double
    /// The beta1.
    public let beta1: Double
    /// The beta2.
    public let beta2: Double
    /// The epsilon.
    public let epsilon: Double
    /// The batch size.
    public let batchSize: Int
    /// The seed.
    public let seed: Int
    /// The requested device.
    public let requestedDevice: ExecutionDevice
    /// The resolved device.
    public private(set) var resolvedDevice: ExecutionDevice = .cpu

    private var layers: [LayerWeights]?

    /// The trained layer weights and biases (nil if not fitted).
    public var trainedLayers: [LayerWeights]? { layers }

    /// Creates a new instance.
    /// - Parameters:
    ///   - hiddenLayerSizes: The hidden layer sizes.
    ///   - activation: The activation.
    ///   - solver: The solver.
    ///   - maxIter: The max iter.
    ///   - learningRate: The learning rate.
    ///   - beta1: The beta1.
    ///   - beta2: The beta2.
    ///   - epsilon: The epsilon.
    ///   - batchSize: The batch size.
    ///   - seed: The seed.
    ///   - requestedDevice: The requested device.
    public init(
        hiddenLayerSizes: [Int] = [100],
        activation: ActivationFunction = .relu,
        solver: MLPSolver = .adam,
        maxIter: Int = 200,
        learningRate: Double = 1e-3,
        beta1: Double = 0.9,
        beta2: Double = 0.999,
        epsilon: Double = 1e-8,
        batchSize: Int = 32,
        seed: Int = 42,
        requestedDevice: ExecutionDevice = .auto
    ) {
        self.hiddenLayerSizes = hiddenLayerSizes
        self.activation = activation
        self.solver = solver
        self.maxIter = maxIter
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batchSize = batchSize
        self.seed = seed
        self.requestedDevice = requestedDevice
    }

    /// Fits the multi-layer perceptron regressor on training data.
    /// - Parameters:
    ///   - features: The features matrix.
    ///   - targets: The continuous target values.
    /// - Throws: An error if inputs are invalid.
    public func fit(features: [[Double]], targets: [Double]) async throws {
        try await fit(features: features, targets: targets, validationFeatures: nil, validationTargets: nil, earlyStopping: nil)
    }

    /// Fits the multi-layer perceptron regressor with optional validation monitoring and early stopping.
    ///
    /// - Parameters:
    ///   - features: 2D feature matrix of shape `[N, P]`.
    ///   - targets: 1D continuous targets of length `N`.
    ///   - validationFeatures: Optional validation features matrix.
    ///   - validationTargets: Optional validation targets array.
    ///   - earlyStopping: Optional `EarlyStopping` callback.
    /// - Throws: `SwiftMLError` if inputs are invalid.
    ///
    /// ## Thread Safety
    /// Thread-safe via actor isolation.
    public func fit(
        features: [[Double]],
        targets: [Double],
        validationFeatures: [[Double]]? = nil,
        validationTargets: [Double]? = nil,
        earlyStopping: EarlyStopping? = nil
    ) async throws {
        self.layers = try await trainLayers(features: features, targets: targets,
            validationFeatures: validationFeatures, validationTargets: validationTargets,
            earlyStopping: earlyStopping)
    }

    private func trainLayers(features: [[Double]], targets: [Double],
        validationFeatures: [[Double]]?, validationTargets: [Double]?,
        earlyStopping: EarlyStopping?) async throws -> [LayerWeights] {
        guard !features.isEmpty, !features[0].isEmpty, !targets.isEmpty else {
            throw SwiftMLError.emptyInput
        }
        let numSamples = features.count
        let numFeatures = try validateMLPTrainingInputs(
            features: features,
            targets: targets,
            validationFeatures: validationFeatures,
            validationTargets: validationTargets
        )

        self.resolvedDevice = await HardwareRouter.shared.resolveDevice(
            for: "MLPRegressor",
            sampleCount: numSamples,
            featureCount: numFeatures,
            requestedDevice: requestedDevice
        )

        let layerSizes = [numFeatures] + hiddenLayerSizes + [1]
        var rng = SeededRandom(seed: seed)

        var layers = [LayerWeights]()
        var adamStates = [LayerAdamState]()

        for l in 0..<(layerSizes.count - 1) {
            let inDim = layerSizes[l]
            let outDim = layerSizes[l + 1]
            let limit = sqrt(6.0 / Double(inDim + outDim))
            var wFlat = [Double]()
            wFlat.reserveCapacity(inDim * outDim)
            for _ in 0..<(inDim * outDim) {
                wFlat.append(rng.nextDouble() * 2.0 * limit - limit)
            }
            let bFlat = [Double](repeating: 0.0, count: outDim)
            layers.append(LayerWeights(W: wFlat, b: bFlat, inDim: inDim, outDim: outDim))
            adamStates.append(LayerAdamState(inDim: inDim, outDim: outDim))
        }

        return mlpRunMiniBatchTraining(
            initialLayers: layers,
            adamStates: &adamStates,
            features: features,
            targets: targets,
            validationFeatures: validationFeatures,
            validationTargets: validationTargets,
            earlyStopping: earlyStopping,
            activation: activation,
            solver: solver,
            maxIter: maxIter,
            learningRate: learningRate,
            beta1: beta1,
            beta2: beta2,
            epsilon: epsilon,
            batchSize: batchSize,
            lossKind: .mse,
            rng: &rng
        )
    }

    /// Fits training-only imputation and scaling, then captures immutable CPU Double prediction.
    /// Validation uses the training plan. Later fits cannot change the returned predictor.
    /// Ordinary prediction on this actor expects already-standardized features afterward.
    /// Training allocations are outside inference budgets; callers can reserve a measured
    /// whole-training allowance with MemoryBudget.withReservation. Cancellation is observed
    /// before and after synchronous training, not inside individual training iterations.
    public func fitStandardized(features: PreparedNumericBatch, targets: [Double],
        validation: PreparedSupervisedBatch? = nil, imputationStrategy: Imputer.Strategy = .mean,
        earlyStopping: EarlyStopping? = nil) async throws -> FittedMLPPredictor {
        try Task.checkCancellation()
        guard features.rowCount > 0, features.columnCount > 0 else { throw SwiftMLError.emptyInput }
        try validateStandardizedConfiguration(width: features.columnCount)
        guard earlyStopping == nil || validation != nil else {
            throw SwiftMLError.invalidParameter("Early stopping requires validation data")
        }
        let training = try StandardizedMLPTraining(features: features, targets: targets,
            validation: validation, strategy: imputationStrategy)
        let trained = try await trainLayers(features: training.rows, targets: targets,
            validationFeatures: training.validation?.rows, validationTargets: training.validation?.targets,
            earlyStopping: earlyStopping)
        try Task.checkCancellation()
        guard trained.allSatisfy({ $0.W.allSatisfy(\.isFinite) && $0.b.allSatisfy(\.isFinite) }) else {
            throw SwiftMLError.trainingFailed("Standardized MLP produced nonfinite weights")
        }
        let weights = try MLPCPUPredictor(layers: trained, activation: activation)
        let result = try FittedMLPPredictor(preprocessing: training.plan, weights: weights)
        self.layers = trained
        return result
    }

    private func validateStandardizedConfiguration(width: Int) throws {
        let sizes = [width] + hiddenLayerSizes + [1]
        guard maxIter >= 0, batchSize > 0, learningRate.isFinite, learningRate > 0,
              (0..<1).contains(beta1), (0..<1).contains(beta2), epsilon.isFinite, epsilon > 0,
              sizes.allSatisfy({ $0 > 0 && $0 <= Int(Int32.max) }) else {
            throw SwiftMLError.invalidParameter("Invalid standardized MLP configuration")
        }
        for index in 1..<sizes.count {
            _ = try preparedMLPBytes(sizes[index - 1], sizes[index], MemoryLayout<Double>.stride)
        }
    }

    /// Predicts finite compact columns without materializing nested row arrays.
    /// Feature positions follow fitting order. Empty input returns an empty result
    /// after fitted-state validation. Nonempty input must match the fitted width.
    public func predict(features: PreparedNumericBatch) async throws -> [Double] {
        try features.requireFinite()
        guard let layers else { throw SwiftMLError.modelNotFitted }
        guard features.rowCount > 0 else { return [] }
        guard features.columnCount == layers[0].inDim else {
            throw SwiftMLError.dimensionMismatch(expected: layers[0].inDim, got: features.columnCount)
        }
        return mlpCompactPrediction(features: features, layers: layers,
            activate: { applyActivation($0, activation: activation) })
    }

    /// Preprocesses compact columns directly into bounded CPU Double buffers.
    package func predict(features: PreparedNumericBatch,
                         preprocessing: StandardPreprocessingPlan) async throws -> [Double] {
        guard let layers else { throw SwiftMLError.modelNotFitted }
        return try mlpPreprocessedPrediction(features: features, preprocessing: preprocessing,
            layers: layers, activate: { applyActivation($0, activation: activation) }, finish: { $0[0] })
    }

    /// Predicts selected occurrences without gathering unchanged columns.
    package func predict(features: IndexedNumericBatch,
                         preprocessing: StandardPreprocessingPlan) async throws -> [Double] {
        guard let layers else { throw SwiftMLError.modelNotFitted }
        return try mlpPreprocessedPrediction(features: features, preprocessing: preprocessing,
            layers: layers, activate: { applyActivation($0, activation: activation) }, finish: { $0[0] })
    }

    /// Captures committed weights for explicit CPU Double prediction.
    /// Later fits do not change the returned owner. Idle snapshots hold no reservation.
    package func makeCPUPredictor() throws -> MLPCPUPredictor {
        guard let layers else { throw SwiftMLError.modelNotFitted }
        return try MLPCPUPredictor(layers: layers, activation: activation)
    }

    /// Admits captured CPU state, then computes on this model actor.
    /// The captured weights remain stable if a fit runs while admission waits.
    package func predict(features: PreparedMLPInput, budget: MemoryBudget) async throws -> [Double] {
        let fitted = try makeCPUPredictor()
        let reservation = try await budget.acquire(fitted.predictionEstimate(for: features))
        do {
            let result = try fitted.predictExternallyAdmitted(features: features)
            await reservation.finish()
            try Task.checkCancellation()
            return result
        } catch {
            await reservation.finish()
            throw error
        }
    }

    /// Predicts from a prepared matrix in either storage order using CPU Double precision.
    /// Columns must follow the positional feature order used during fitting; names do not reorder them.
    /// Empty input returns an empty result after the fitted-model check.
    /// Throws if the model is not fitted or a nonempty matrix has the wrong width.
    public func predict(features: PreparedNumericMatrix) async throws -> [Double] {
        guard let layers = layers else { throw SwiftMLError.modelNotFitted }
        guard features.rowCount > 0 else { return [] }
        try mlpValidateMatrix(features, layers: layers)
        if let batched = mlpBatchedPrediction(matrix: features, layers: layers,
            activate: { applyActivation($0, activation: activation) }, finish: { $0[0] }) { return batched }
        var results: [Double] = []
        results.reserveCapacity(features.rowCount)
        for row in 0..<features.rowCount {
            let prediction = mlpPredictRow(mlpMatrixRow(features, row: row), layers: layers,
                activate: { applyActivation($0, activation: activation) })
            results.append(prediction[0])
        }
        return results
    }

    /// Predict.
    /// - Parameters:
    ///   - features: The features.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[Double]` result.
    public func predict(features: [[Double]]) async throws -> [Double] {
        guard let layers = layers else {
            throw SwiftMLError.modelNotFitted
        }
        guard !features.isEmpty else { return [] }

        if let batched = try mlpBatchedPrediction(features: features, layers: layers,
            activate: { applyActivation($0, activation: activation) },
            finish: { $0[0] }) { return batched }

        var results = [Double]()
        results.reserveCapacity(features.count)

        for x in features {
            let curr = try mlpPredictRowValidated(x, layers: layers,
                activate: { applyActivation($0, activation: activation) })
            results.append(curr[0])
        }

        return results
    }
}

/// Immutable weights for CPU Double prediction, independent of prepared input ownership.
/// Direct calls are not serialized by the model actor. Callers control concurrency and idle retention.
package final class MLPCPUPredictor: Sendable {
    private let layers: [LayerWeights]
    private let activation: ActivationFunction
    /// Conservative weight capacity charge. Shared arrays may be charged again.
    package let retainedBytes: Int

    fileprivate init(layers: [LayerWeights], activation: ActivationFunction) throws {
        guard !layers.isEmpty else { throw SwiftMLError.modelNotFitted }
        self.layers = layers
        self.activation = activation
        retainedBytes = try preparedMLPModelBytes(layers)
    }

    /// Includes one input, captured weights, workspace, and replacing output buffers.
    /// Concurrent predictions, accumulated outputs, and unrelated owners need additional allowance.
    package func predictionEstimate(for features: PreparedMLPInput) throws -> MemoryEstimate {
        try preparedMLPPredictionEstimate(features, layers: layers, modelBytes: retainedBytes)
    }

    /// Admits the complete estimate and releases it after synchronous CPU work finishes.
    package func predict(features: PreparedMLPInput, budget: MemoryBudget) async throws -> [Double] {
        let reservation = try await budget.acquire(predictionEstimate(for: features))
        do {
            let result = try predictExternallyAdmitted(features: features)
            await reservation.finish()
            try Task.checkCancellation()
            return result
        } catch {
            await reservation.finish()
            throw error
        }
    }

    /// Caller-managed admission for this immutable model and input.
    /// Runs synchronously on the calling context. Individual BLAS calls are not interruptible.
    package func predictExternallyAdmitted(features: PreparedMLPInput) throws -> [Double] {
        try Task.checkCancellation()
        let result = try mlpPreparedInputPrediction(features: features, layers: layers, activation: activation)
        try Task.checkCancellation()
        return result
    }
}

private func mlpPreparedInputPrediction(features: PreparedMLPInput, layers: [LayerWeights],
                                        activation: ActivationFunction) throws -> [Double] {
    let prediction = try mlpPreprocessedPrediction(features: features.batch,
        preprocessing: features.preprocessing, layers: layers,
        activate: { applyActivation($0, activation: activation) }, finish: { $0[0] })
    return features.restoring(prediction)
}

private func applyActivation(_ x: Double, activation: ActivationFunction) -> Double {
    switch activation {
    case .relu: return max(0.0, x)
    case .sigmoid: return sigmoid(x)
    case .tanh: return tanh(x)
    }
}

private func applyActivationDeriv(_ a: Double, activation: ActivationFunction) -> Double {
    switch activation {
    case .relu: return a > 0.0 ? 1.0 : 0.0
    case .sigmoid: return a * (1.0 - a)
    case .tanh: return 1.0 - a * a
    }
}

private func mlpProbabilities<Values: Collection>(_ logits: Values, classCount: Int) -> [Double]
where Values.Element == Double {
    if classCount <= 2 {
        let p1 = sigmoid(logits.first!)
        return [1.0 - p1, p1]
    }
    let maxLogit = logits.max() ?? 0.0
    let exps = logits.map { exp($0 - maxLogit) }
    let sumExp = exps.reduce(0.0, +)
    return exps.map { $0 / sumExp }
}

// MARK: - Mini-Batch BLAS Training Engine

enum MLPLossKind {
    case binaryCrossEntropy(positiveClass: Double)
    case softmaxCrossEntropy(classes: [Double])
    case mse
}

final class MLPFlatBuffer {
    var data: [Double]
    init(count: Int) {
        self.data = [Double](repeating: 0.0, count: count)
    }
}

final class MLPMiniBatchWorkspace {
    var batchInput: [Double]
    var batchTargets: [Double]
    var layerOutputs: [MLPFlatBuffer]
    var deltaA: [Double]
    var deltaB: [Double]
    var gradW: [MLPFlatBuffer]
    var gradB: [MLPFlatBuffer]
    var sgdScratch: [Double]

    init(layers: [LayerWeights], maxBatchSize: Int) {
        let numFeatures = layers[0].inDim
        let maxDim = layers.reduce(numFeatures) { max($0, max($1.inDim, $1.outDim)) }
        let maxWeightCount = layers.reduce(1) { max($0, max($1.W.count, $1.outDim)) }

        self.batchInput = [Double](repeating: 0.0, count: maxBatchSize * numFeatures)
        self.batchTargets = [Double](repeating: 0.0, count: maxBatchSize)
        self.layerOutputs = layers.map { MLPFlatBuffer(count: maxBatchSize * $0.outDim) }
        self.deltaA = [Double](repeating: 0.0, count: maxBatchSize * maxDim)
        self.deltaB = [Double](repeating: 0.0, count: maxBatchSize * maxDim)
        self.gradW = layers.map { MLPFlatBuffer(count: $0.inDim * $0.outDim) }
        self.gradB = layers.map { MLPFlatBuffer(count: $0.outDim) }
        self.sgdScratch = [Double](repeating: 0.0, count: maxWeightCount)
    }
}

private func validateMLPTrainingInputs(
    features: [[Double]],
    targets: [Double],
    validationFeatures: [[Double]]?,
    validationTargets: [Double]?
) throws -> Int {
    guard features.count == targets.count else {
        throw SwiftMLError.dimensionMismatch(expected: features.count, got: targets.count)
    }
    let numFeatures = features[0].count
    for row in features {
        guard row.count == numFeatures else {
            throw SwiftMLError.dimensionMismatch(expected: numFeatures, got: row.count)
        }
    }
    if (validationFeatures != nil) != (validationTargets != nil) {
        throw SwiftMLError.invalidInput("Validation features and targets must both be provided")
    }
    if let valX = validationFeatures, let valY = validationTargets {
        guard valX.count == valY.count else {
            throw SwiftMLError.dimensionMismatch(expected: valX.count, got: valY.count)
        }
        for row in valX {
            guard row.count == numFeatures else {
                throw SwiftMLError.dimensionMismatch(expected: numFeatures, got: row.count)
            }
        }
    }
    return numFeatures
}

@discardableResult
func mlpComputeMiniBatchLossAndGradients(
    layers: [LayerWeights],
    features: [[Double]],
    targets: [Double],
    indices: ArraySlice<Int>,
    activation: ActivationFunction,
    lossKind: MLPLossKind,
    workspace: MLPMiniBatchWorkspace,
    computeGradients: Bool
) -> Double {
    let actualB = indices.count
    guard actualB > 0 else { return 0.0 }
    let numFeatures = layers[0].inDim

    workspace.batchInput.withUnsafeMutableBufferPointer { inputBuf in
        let inputPtr = inputBuf.baseAddress!
        var r = 0
        for sampleIdx in indices {
            let rowOff = r * numFeatures
            features[sampleIdx].withUnsafeBufferPointer { src in
                (inputPtr + rowOff).update(from: src.baseAddress!, count: numFeatures)
            }
            workspace.batchTargets[r] = targets[sampleIdx]
            r += 1
        }
    }

    // Forward pass across all layers
    for l in 0..<layers.count {
        let inD = layers[l].inDim
        let outD = layers[l].outDim
        let isLast = l == layers.count - 1

        layers[l].b.withUnsafeBufferPointer { bBuf in
            let bPtr = bBuf.baseAddress!
            workspace.layerOutputs[l].data.withUnsafeMutableBufferPointer { outBuf in
                let outPtr = outBuf.baseAddress!
                for r in 0..<actualB {
                    (outPtr + r * outD).update(from: bPtr, count: outD)
                }
                layers[l].W.withUnsafeBufferPointer { wBuf in
                    let wPtr = wBuf.baseAddress!
                    let runGemm: (UnsafePointer<Double>) -> Void = { prevPtr in
                        cblas_dgemm(
                            CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            Int32(actualB), Int32(outD), Int32(inD),
                            1.0,
                            prevPtr, Int32(inD),
                            wPtr, Int32(outD),
                            1.0,
                            outPtr, Int32(outD)
                        )
                    }
                    if l == 0 {
                        workspace.batchInput.withUnsafeBufferPointer { runGemm($0.baseAddress!) }
                    } else {
                        workspace.layerOutputs[l - 1].data.withUnsafeBufferPointer { runGemm($0.baseAddress!) }
                    }
                }
                if !isLast {
                    let count = actualB * outD
                    for idx in 0..<count {
                        outPtr[idx] = applyActivation(outPtr[idx], activation: activation)
                    }
                }
            }
        }
    }

    // Compute loss and initial output logit delta
    var totalLoss = 0.0
    workspace.layerOutputs[layers.count - 1].data.withUnsafeBufferPointer { lastBuf in
        let lastPtr = lastBuf.baseAddress!
        workspace.deltaA.withUnsafeMutableBufferPointer { deltaBuf in
            let deltaPtr = deltaBuf.baseAddress!
            switch lossKind {
            case .binaryCrossEntropy(let positiveClass):
                for r in 0..<actualB {
                    let z = lastPtr[r]
                    let y = workspace.batchTargets[r] == positiveClass ? 1.0 : 0.0
                    totalLoss += log1p(exp(-abs(z))) + max(z, 0.0) - z * y
                    if computeGradients {
                        deltaPtr[r] = sigmoid(z) - y
                    }
                }
            case .softmaxCrossEntropy(let uniqueClasses):
                let numClasses = uniqueClasses.count
                for r in 0..<actualB {
                    let rowOff = r * numClasses
                    var m = lastPtr[rowOff]
                    for c in 1..<numClasses {
                        if lastPtr[rowOff + c] > m {
                            m = lastPtr[rowOff + c]
                        }
                    }
                    var sumExp = 0.0
                    for c in 0..<numClasses {
                        let e = exp(lastPtr[rowOff + c] - m)
                        deltaPtr[rowOff + c] = e
                        sumExp += e
                    }
                    let logZ = m + log(sumExp)
                    let yVal = workspace.batchTargets[r]
                    if let targetIdx = uniqueClasses.firstIndex(of: yVal) {
                        totalLoss += logZ - lastPtr[rowOff + targetIdx]
                    } else {
                        totalLoss += logZ
                    }
                    if computeGradients {
                        let invSum = 1.0 / sumExp
                        for c in 0..<numClasses {
                            let p = deltaPtr[rowOff + c] * invSum
                            deltaPtr[rowOff + c] = p - (uniqueClasses[c] == yVal ? 1.0 : 0.0)
                        }
                    }
                }
            case .mse:
                for r in 0..<actualB {
                    let diff = lastPtr[r] - workspace.batchTargets[r]
                    totalLoss += 0.5 * diff * diff
                    if computeGradients {
                        deltaPtr[r] = diff
                    }
                }
            }
        }
    }

    let meanLoss = totalLoss / Double(actualB)
    guard computeGradients else {
        return meanLoss
    }

    // Backward pass across layers using BLAS dgemm
    let scale = 1.0 / Double(actualB)
    var deltaInA = true

    for l in stride(from: layers.count - 1, through: 0, by: -1) {
        let inD = layers[l].inDim
        let outD = layers[l].outDim

        mlpWithPrevAndDelta(workspace: workspace, layerIndex: l, deltaInA: deltaInA) { prevPtr, deltaPtr in
            workspace.gradW[l].data.withUnsafeMutableBufferPointer { gradWBuf in
                cblas_dgemm(
                    CblasRowMajor, CblasTrans, CblasNoTrans,
                    Int32(inD), Int32(outD), Int32(actualB),
                    scale,
                    prevPtr, Int32(inD),
                    deltaPtr, Int32(outD),
                    0.0,
                    gradWBuf.baseAddress!, Int32(outD)
                )
            }

            workspace.gradB[l].data.withUnsafeMutableBufferPointer { gradBBuf in
                let gradBPtr = gradBBuf.baseAddress!
                for j in 0..<outD {
                    gradBPtr[j] = 0.0
                }
                for r in 0..<actualB {
                    let rowOff = r * outD
                    for j in 0..<outD {
                        gradBPtr[j] += deltaPtr[rowOff + j]
                    }
                }
                for j in 0..<outD {
                    gradBPtr[j] *= scale
                }
            }
        }

        if l > 0 {
            let actCount = actualB * inD
            let computePrevDelta: (UnsafePointer<Double>, UnsafeMutablePointer<Double>) -> Void = { deltaPtr, nextDeltaPtr in
                layers[l].W.withUnsafeBufferPointer { wBuf in
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,
                        Int32(actualB), Int32(inD), Int32(outD),
                        1.0,
                        deltaPtr, Int32(outD),
                        wBuf.baseAddress!, Int32(outD),
                        0.0,
                        nextDeltaPtr, Int32(inD)
                    )
                }
                workspace.layerOutputs[l - 1].data.withUnsafeBufferPointer { prevActBuf in
                    let prevActPtr = prevActBuf.baseAddress!
                    for idx in 0..<actCount {
                        nextDeltaPtr[idx] *= applyActivationDeriv(prevActPtr[idx], activation: activation)
                    }
                }
            }

            if deltaInA {
                workspace.deltaA.withUnsafeBufferPointer { dBuf in
                    workspace.deltaB.withUnsafeMutableBufferPointer { nBuf in
                        computePrevDelta(dBuf.baseAddress!, nBuf.baseAddress!)
                    }
                }
            } else {
                workspace.deltaB.withUnsafeBufferPointer { dBuf in
                    workspace.deltaA.withUnsafeMutableBufferPointer { nBuf in
                        computePrevDelta(dBuf.baseAddress!, nBuf.baseAddress!)
                    }
                }
            }
            deltaInA.toggle()
        }
    }

    return meanLoss
}

private func mlpWithPrevAndDelta(
    workspace: MLPMiniBatchWorkspace,
    layerIndex l: Int,
    deltaInA: Bool,
    body: (UnsafePointer<Double>, UnsafePointer<Double>) -> Void
) {
    if l == 0 {
        workspace.batchInput.withUnsafeBufferPointer { prevBuf in
            if deltaInA {
                workspace.deltaA.withUnsafeBufferPointer { deltaBuf in
                    body(prevBuf.baseAddress!, deltaBuf.baseAddress!)
                }
            } else {
                workspace.deltaB.withUnsafeBufferPointer { deltaBuf in
                    body(prevBuf.baseAddress!, deltaBuf.baseAddress!)
                }
            }
        }
    } else {
        workspace.layerOutputs[l - 1].data.withUnsafeBufferPointer { prevBuf in
            if deltaInA {
                workspace.deltaA.withUnsafeBufferPointer { deltaBuf in
                    body(prevBuf.baseAddress!, deltaBuf.baseAddress!)
                }
            } else {
                workspace.deltaB.withUnsafeBufferPointer { deltaBuf in
                    body(prevBuf.baseAddress!, deltaBuf.baseAddress!)
                }
            }
        }
    }
}

private func mlpApplyOptimizerStep(
    layers: inout [LayerWeights],
    adamStates: inout [LayerAdamState],
    workspace: MLPMiniBatchWorkspace,
    solver: MLPSolver,
    learningRate: Double,
    beta1: Double,
    beta2: Double,
    epsilon: Double
) {
    if solver == .adam {
        let oneMinusBeta1 = 1.0 - beta1
        let oneMinusBeta2 = 1.0 - beta2
        for l in 0..<layers.count {
            adamStates[l].t += 1
            let t = Double(adamStates[l].t)
            let invB1Corr = 1.0 / (1.0 - pow(beta1, t))
            let invB2Corr = 1.0 / (1.0 - pow(beta2, t))
            let wCount = layers[l].W.count
            let outD = layers[l].outDim

            layers[l].W.withUnsafeMutableBufferPointer { wBuf in
                let wPtr = wBuf.baseAddress!
                workspace.gradW[l].data.withUnsafeBufferPointer { gBuf in
                    let gPtr = gBuf.baseAddress!
                    adamStates[l].mW.withUnsafeMutableBufferPointer { mBuf in
                        let mPtr = mBuf.baseAddress!
                        adamStates[l].vW.withUnsafeMutableBufferPointer { vBuf in
                            let vPtr = vBuf.baseAddress!
                            for p in 0..<wCount {
                                let g = gPtr[p]
                                let mNew = beta1 * mPtr[p] + oneMinusBeta1 * g
                                let vNew = beta2 * vPtr[p] + oneMinusBeta2 * g * g
                                mPtr[p] = mNew
                                vPtr[p] = vNew
                                let mHat = mNew * invB1Corr
                                let vHat = vNew * invB2Corr
                                wPtr[p] -= learningRate * mHat / (sqrt(vHat) + epsilon)
                            }
                        }
                    }
                }
            }

            layers[l].b.withUnsafeMutableBufferPointer { bBuf in
                let bPtr = bBuf.baseAddress!
                workspace.gradB[l].data.withUnsafeBufferPointer { gBuf in
                    let gPtr = gBuf.baseAddress!
                    adamStates[l].mB.withUnsafeMutableBufferPointer { mBuf in
                        let mPtr = mBuf.baseAddress!
                        adamStates[l].vB.withUnsafeMutableBufferPointer { vBuf in
                            let vPtr = vBuf.baseAddress!
                            for j in 0..<outD {
                                let g = gPtr[j]
                                let mNew = beta1 * mPtr[j] + oneMinusBeta1 * g
                                let vNew = beta2 * vPtr[j] + oneMinusBeta2 * g * g
                                mPtr[j] = mNew
                                vPtr[j] = vNew
                                let mHat = mNew * invB1Corr
                                let vHat = vNew * invB2Corr
                                bPtr[j] -= learningRate * mHat / (sqrt(vHat) + epsilon)
                            }
                        }
                    }
                }
            }
        }
    } else {
        var negLr = -learningRate
        for l in 0..<layers.count {
            let wCount = layers[l].W.count
            let outD = layers[l].outDim
            workspace.sgdScratch.withUnsafeMutableBufferPointer { scratchBuf in
                let scratchPtr = scratchBuf.baseAddress!
                workspace.gradW[l].data.withUnsafeBufferPointer { gWBuf in
                    vDSP_vsmulD(gWBuf.baseAddress!, 1, &negLr, scratchPtr, 1, vDSP_Length(wCount))
                    layers[l].W.withUnsafeMutableBufferPointer { wBuf in
                        vDSP_vaddD(wBuf.baseAddress!, 1, scratchPtr, 1, wBuf.baseAddress!, 1, vDSP_Length(wCount))
                    }
                }
                workspace.gradB[l].data.withUnsafeBufferPointer { gBBuf in
                    vDSP_vsmulD(gBBuf.baseAddress!, 1, &negLr, scratchPtr, 1, vDSP_Length(outD))
                    layers[l].b.withUnsafeMutableBufferPointer { bBuf in
                        vDSP_vaddD(bBuf.baseAddress!, 1, scratchPtr, 1, bBuf.baseAddress!, 1, vDSP_Length(outD))
                    }
                }
            }
        }
    }
}

private func mlpRunMiniBatchTraining(
    initialLayers: [LayerWeights],
    adamStates: inout [LayerAdamState],
    features: [[Double]],
    targets: [Double],
    validationFeatures: [[Double]]?,
    validationTargets: [Double]?,
    earlyStopping: EarlyStopping?,
    activation: ActivationFunction,
    solver: MLPSolver,
    maxIter: Int,
    learningRate: Double,
    beta1: Double,
    beta2: Double,
    epsilon: Double,
    batchSize: Int,
    lossKind: MLPLossKind,
    rng: inout SeededRandom
) -> [LayerWeights] {
    var layers = initialLayers
    let numSamples = features.count
    let effectiveBatchSize = max(1, min(batchSize, numSamples))
    let valCount = validationFeatures?.count ?? 0
    let maxWorkspaceRows = max(effectiveBatchSize, min(effectiveBatchSize, valCount))
    let workspace = MLPMiniBatchWorkspace(layers: layers, maxBatchSize: maxWorkspaceRows)

    var trainIndices = Array(0..<numSamples)
    let valIndices = Array(0..<valCount)
    var es = earlyStopping
    var bestLayers: [LayerWeights]? = nil

    for epoch in 0..<maxIter {
        trainIndices.shuffle(using: &rng)

        var start = 0
        while start < numSamples {
            let end = min(start + effectiveBatchSize, numSamples)
            mlpComputeMiniBatchLossAndGradients(
                layers: layers,
                features: features,
                targets: targets,
                indices: trainIndices[start..<end],
                activation: activation,
                lossKind: lossKind,
                workspace: workspace,
                computeGradients: true
            )
            mlpApplyOptimizerStep(
                layers: &layers,
                adamStates: &adamStates,
                workspace: workspace,
                solver: solver,
                learningRate: learningRate,
                beta1: beta1,
                beta2: beta2,
                epsilon: epsilon
            )
            start = end
        }

        if var tracker = es, let valX = validationFeatures, let valY = validationTargets, !valX.isEmpty {
            var valLossSum = 0.0
            var vStart = 0
            while vStart < valCount {
                let vEnd = min(vStart + effectiveBatchSize, valCount)
                let batchCount = vEnd - vStart
                let batchMean = mlpComputeMiniBatchLossAndGradients(
                    layers: layers,
                    features: valX,
                    targets: valY,
                    indices: valIndices[vStart..<vEnd],
                    activation: activation,
                    lossKind: lossKind,
                    workspace: workspace,
                    computeGradients: false
                )
                let batchValScore = switch lossKind {
                case .mse: 2.0 * batchMean
                case .binaryCrossEntropy, .softmaxCrossEntropy: batchMean
                }
                valLossSum += batchValScore * Double(batchCount)
                vStart = vEnd
            }
            let valScore = valLossSum / Double(valCount)
            let (shouldStop, isBest) = tracker.step(currentScore: valScore, epoch: epoch)
            if isBest {
                bestLayers = layers
            }
            es = tracker
            if shouldStop {
                if tracker.restoreBestWeights, let best = bestLayers {
                    layers = best
                }
                break
            }
        }
    }

    return layers
}
