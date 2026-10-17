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

private struct LayerAdamState: Sendable {
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
        let numFeatures = features[0].count

        self.resolvedDevice = await HardwareRouter.shared.resolveDevice(
            for: "MLPClassifier",
            sampleCount: numSamples,
            featureCount: numFeatures,
            requestedDevice: requestedDevice
        )

        let uniqueClasses = Array(Set(targets)).sorted()
        self.classes = uniqueClasses
        let numClasses = max(1, uniqueClasses.count)

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

        var es = earlyStopping
        var bestLayers: [LayerWeights]? = nil

        for epoch in 0..<maxIter {
            for i in 0..<numSamples {
                let x = features[i]
                let yVal = targets[i]

                // Forward Pass
                var activations = [x]
                for l in 0..<layers.count {
                    let prev = activations[l]
                    let isLast = l == layers.count - 1
                    var out = layers[l].b
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        1, Int32(layers[l].outDim), Int32(layers[l].inDim),
                        1.0,
                        prev, Int32(layers[l].inDim),
                        layers[l].W, Int32(layers[l].outDim),
                        1.0,
                        &out, Int32(layers[l].outDim)
                    )

                    if isLast {
                        if numClasses == 2 {
                            out[0] = sigmoid(out[0])
                        }
                    } else {
                        for j in 0..<layers[l].outDim {
                            out[j] = applyActivation(out[j], activation: activation)
                        }
                    }
                    activations.append(out)
                }

                // Output Error Delta
                var delta = [Double]()
                let lastOut = activations.last!

                if numClasses <= 2 {
                    let err = lastOut[0] - (yVal == (uniqueClasses.last ?? 1.0) ? 1.0 : 0.0)
                    delta = [err]
                } else {
                    delta = lastOut.enumerated().map { (cIdx, v) in
                        v - (uniqueClasses[cIdx] == yVal ? 1.0 : 0.0)
                    }
                }

                // Backward Pass & Gradient Updates
                for l in stride(from: layers.count - 1, through: 0, by: -1) {
                    let prevAct = activations[l]
                    let inD = layers[l].inDim
                    let outD = layers[l].outDim

                    var gradW = [Double](repeating: 0.0, count: inD * outD)
                    var nextDelta = [Double](repeating: 0.0, count: inD)

                    for k in 0..<inD {
                        for j in 0..<outD {
                            let idx = k * outD + j
                            gradW[idx] = delta[j] * prevAct[k]
                            nextDelta[k] += delta[j] * layers[l].W[idx]
                        }
                        if l > 0 {
                            nextDelta[k] *= applyActivationDeriv(prevAct[k], activation: activation)
                        }
                    }
                    let gradB = delta

                    // Optimizer Step
                    if solver == .adam {
                        adamStates[l].t += 1
                        let t = Double(adamStates[l].t)
                        let b1_corr = 1.0 - pow(beta1, t)
                        let b2_corr = 1.0 - pow(beta2, t)
                        let oneMinusBeta1 = 1.0 - beta1
                        let oneMinusBeta2 = 1.0 - beta2
                        let lr = learningRate
                        let eps = epsilon
                        let wCount = layers[l].W.count

                        for p in 0..<wCount {
                            let g = gradW[p]
                            adamStates[l].mW[p] = beta1 * adamStates[l].mW[p] + oneMinusBeta1 * g
                            adamStates[l].vW[p] = beta2 * adamStates[l].vW[p] + oneMinusBeta2 * g * g
                            let mHat = adamStates[l].mW[p] / b1_corr
                            let vHat = adamStates[l].vW[p] / b2_corr
                            layers[l].W[p] -= lr * mHat / (sqrt(vHat) + eps)
                        }

                        for j in 0..<outD {
                            let g = gradB[j]
                            adamStates[l].mB[j] = beta1 * adamStates[l].mB[j] + oneMinusBeta1 * g
                            adamStates[l].vB[j] = beta2 * adamStates[l].vB[j] + oneMinusBeta2 * g * g
                            let mHat = adamStates[l].mB[j] / b1_corr
                            let vHat = adamStates[l].vB[j] / b2_corr
                            layers[l].b[j] -= lr * mHat / (sqrt(vHat) + eps)
                        }
                    } else {
                        // SGD via vDSP
                        let wCount = layers[l].W.count
                        var negLr = -learningRate
                        var stepW = [Double](repeating: 0.0, count: wCount)
                        vDSP_vsmulD(gradW, 1, &negLr, &stepW, 1, vDSP_Length(wCount))
                        vDSP_vaddD(layers[l].W, 1, stepW, 1, &layers[l].W, 1, vDSP_Length(wCount))

                        var stepB = [Double](repeating: 0.0, count: outD)
                        vDSP_vsmulD(gradB, 1, &negLr, &stepB, 1, vDSP_Length(outD))
                        vDSP_vaddD(layers[l].b, 1, stepB, 1, &layers[l].b, 1, vDSP_Length(outD))
                    }

                    delta = nextDelta
                }
            }

            // Early stopping validation check
            if var tracker = es, let valX = validationFeatures, let valY = validationTargets, !valX.isEmpty {
                var totalLoss = 0.0
                for vi in 0..<valX.count {
                    let vx = valX[vi]
                    let vy = valY[vi]
                    var out = vx
                    for l in 0..<layers.count {
                        var nextOut = layers[l].b
                        cblas_dgemm(
                            CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            1, Int32(layers[l].outDim), Int32(layers[l].inDim),
                            1.0,
                            out, Int32(layers[l].inDim),
                            layers[l].W, Int32(layers[l].outDim),
                            1.0,
                            &nextOut, Int32(layers[l].outDim)
                        )
                        if l == layers.count - 1 {
                            if numClasses == 2 {
                                nextOut[0] = sigmoid(nextOut[0])
                            }
                        } else {
                            for j in 0..<layers[l].outDim {
                                nextOut[j] = applyActivation(nextOut[j], activation: activation)
                            }
                        }
                        out = nextOut
                    }

                    if numClasses <= 2 {
                        let targetVal = vy == (uniqueClasses.last ?? 1.0) ? 1.0 : 0.0
                        let p = max(1e-12, min(1.0 - 1e-12, out[0]))
                        totalLoss -= (targetVal * log(p) + (1.0 - targetVal) * log(1.0 - p))
                    } else {
                        let targetIdx = uniqueClasses.firstIndex(of: vy) ?? 0
                        let p = max(1e-12, min(1.0 - 1e-12, out[targetIdx]))
                        totalLoss -= log(p)
                    }
                }
                totalLoss /= Double(valX.count)
                let (shouldStop, isBest) = tracker.step(currentScore: totalLoss, epoch: epoch)
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

        if features.count >= mlpPredictionBatchRows,
           let batched = try mlpBatchedPrediction(features: features, layers: layers,
            activate: { applyActivation($0, activation: activation) },
            finish: { mlpProbabilities($0, classCount: classes.count) }) { return batched }

        var results = [[Double]]()
        results.reserveCapacity(features.count)

        for x in features {
            let curr = mlpPredictRow(x, layers: layers,
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
        let numFeatures = features[0].count

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

        var es = earlyStopping
        var bestLayers: [LayerWeights]? = nil

        for epoch in 0..<maxIter {
            for i in 0..<numSamples {
                let x = features[i]
                let yVal = targets[i]

                // Forward Pass
                var activations = [x]
                for l in 0..<layers.count {
                    let prev = activations[l]
                    let isLast = l == layers.count - 1
                    var out = layers[l].b
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        1, Int32(layers[l].outDim), Int32(layers[l].inDim),
                        1.0,
                        prev, Int32(layers[l].inDim),
                        layers[l].W, Int32(layers[l].outDim),
                        1.0,
                        &out, Int32(layers[l].outDim)
                    )

                    if !isLast {
                        for j in 0..<layers[l].outDim {
                            out[j] = applyActivation(out[j], activation: activation)
                        }
                    }
                    activations.append(out)
                }

                // Output Error Delta
                var delta = [activations.last![0] - yVal]

                // Backward Pass & Gradient Updates
                for l in stride(from: layers.count - 1, through: 0, by: -1) {
                    let prevAct = activations[l]
                    let inD = layers[l].inDim
                    let outD = layers[l].outDim

                    var gradW = [Double](repeating: 0.0, count: inD * outD)
                    var nextDelta = [Double](repeating: 0.0, count: inD)

                    for k in 0..<inD {
                        for j in 0..<outD {
                            let idx = k * outD + j
                            gradW[idx] = delta[j] * prevAct[k]
                            nextDelta[k] += delta[j] * layers[l].W[idx]
                        }
                        if l > 0 {
                            nextDelta[k] *= applyActivationDeriv(prevAct[k], activation: activation)
                        }
                    }
                    let gradB = delta

                    // Optimizer Step
                    if solver == .adam {
                        adamStates[l].t += 1
                        let t = Double(adamStates[l].t)
                        let b1_corr = 1.0 - pow(beta1, t)
                        let b2_corr = 1.0 - pow(beta2, t)
                        let oneMinusBeta1 = 1.0 - beta1
                        let oneMinusBeta2 = 1.0 - beta2
                        let lr = learningRate
                        let eps = epsilon

                        for j in 0..<outD {
                            for k in 0..<inD {
                                let idx = k * outD + j
                                let g = gradW[idx]
                                adamStates[l].mW[idx] = beta1 * adamStates[l].mW[idx] + oneMinusBeta1 * g
                                adamStates[l].vW[idx] = beta2 * adamStates[l].vW[idx] + oneMinusBeta2 * g * g
                                let mHat = adamStates[l].mW[idx] / b1_corr
                                let vHat = adamStates[l].vW[idx] / b2_corr
                                layers[l].W[idx] -= lr * mHat / (sqrt(vHat) + eps)
                            }
                            let gB = gradB[j]
                            adamStates[l].mB[j] = beta1 * adamStates[l].mB[j] + oneMinusBeta1 * gB
                            adamStates[l].vB[j] = beta2 * adamStates[l].vB[j] + oneMinusBeta2 * gB * gB
                            let mHat = adamStates[l].mB[j] / b1_corr
                            let vHat = adamStates[l].vB[j] / b2_corr
                            layers[l].b[j] -= lr * mHat / (sqrt(vHat) + eps)
                        }
                    } else {
                        // SGD via vDSP
                        let wCount = layers[l].W.count
                        var negLr = -learningRate
                        var stepW = [Double](repeating: 0.0, count: wCount)
                        vDSP_vsmulD(gradW, 1, &negLr, &stepW, 1, vDSP_Length(wCount))
                        vDSP_vaddD(layers[l].W, 1, stepW, 1, &layers[l].W, 1, vDSP_Length(wCount))

                        var stepB = [Double](repeating: 0.0, count: outD)
                        vDSP_vsmulD(gradB, 1, &negLr, &stepB, 1, vDSP_Length(outD))
                        vDSP_vaddD(layers[l].b, 1, stepB, 1, &layers[l].b, 1, vDSP_Length(outD))
                    }

                    delta = nextDelta
                }
            }

            // Early stopping validation check
            if var tracker = es, let valX = validationFeatures, let valY = validationTargets, !valX.isEmpty {
                var mse = 0.0
                for vi in 0..<valX.count {
                    let vx = valX[vi]
                    var out = vx
                    for l in 0..<layers.count {
                        let isLast = l == layers.count - 1
                        var nextOut = layers[l].b
                        cblas_dgemm(
                            CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            1, Int32(layers[l].outDim), Int32(layers[l].inDim),
                            1.0,
                            out, Int32(layers[l].inDim),
                            layers[l].W, Int32(layers[l].outDim),
                            1.0,
                            &nextOut, Int32(layers[l].outDim)
                        )
                        if !isLast {
                            for j in 0..<layers[l].outDim {
                                nextOut[j] = applyActivation(nextOut[j], activation: activation)
                            }
                        }
                        out = nextOut
                    }
                    let diff = out[0] - valY[vi]
                    mse += diff * diff
                }
                mse /= Double(valX.count)
                let (shouldStop, isBest) = tracker.step(currentScore: mse, epoch: epoch)
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

        if features.count >= mlpPredictionBatchRows,
           let batched = try mlpBatchedPrediction(features: features, layers: layers,
            activate: { applyActivation($0, activation: activation) },
            finish: { $0[0] }) { return batched }

        var results = [Double]()
        results.reserveCapacity(features.count)

        for x in features {
            let curr = mlpPredictRow(x, layers: layers,
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

private func applyActivationDeriv(_ x: Double, activation: ActivationFunction) -> Double {
    switch activation {
    case .relu: return x > 0 ? 1.0 : 0.0
    case .sigmoid:
        let s = sigmoid(x)
        return s * (1.0 - s)
    case .tanh:
        let t = tanh(x)
        return 1.0 - t * t
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
