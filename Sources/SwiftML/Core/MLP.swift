import Foundation
import Accelerate
import SwiftPreprocessing

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
    public var W: [Double]   // row-major flat buffer: inDim x outDim
    public var b: [Double]   // outDim
    public let inDim: Int
    public let outDim: Int

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
    public let hiddenLayerSizes: [Int]
    public let activation: ActivationFunction
    public let solver: MLPSolver
    public let maxIter: Int
    public let learningRate: Double
    public let beta1: Double
    public let beta2: Double
    public let epsilon: Double
    public let batchSize: Int
    public let seed: Int
    public let requestedDevice: ExecutionDevice
    public private(set) var resolvedDevice: ExecutionDevice = .cpu

    private var layers: [LayerWeights]?
    private var classes: [Double]?

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

    public func fit(features: [[Double]], targets: [Double]) async throws {
        guard !features.isEmpty, !targets.isEmpty else {
            throw MLError.emptyInput
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

        for _ in 0..<maxIter {
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
                            out[0] = 1.0 / (1.0 + exp(-out[0]))
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

                        for p in 0..<layers[l].W.count {
                            let g = gradW[p]
                            adamStates[l].mW[p] = beta1 * adamStates[l].mW[p] + (1.0 - beta1) * g
                            adamStates[l].vW[p] = beta2 * adamStates[l].vW[p] + (1.0 - beta2) * g * g
                            let mHat = adamStates[l].mW[p] / b1_corr
                            let vHat = adamStates[l].vW[p] / b2_corr
                            layers[l].W[p] -= learningRate * mHat / (sqrt(vHat) + epsilon)
                        }

                        for j in 0..<outD {
                            let g = gradB[j]
                            adamStates[l].mB[j] = beta1 * adamStates[l].mB[j] + (1.0 - beta1) * g
                            adamStates[l].vB[j] = beta2 * adamStates[l].vB[j] + (1.0 - beta2) * g * g
                            let mHat = adamStates[l].mB[j] / b1_corr
                            let vHat = adamStates[l].vB[j] / b2_corr
                            layers[l].b[j] -= learningRate * mHat / (sqrt(vHat) + epsilon)
                        }
                    } else {
                        // SGD
                        for p in 0..<layers[l].W.count {
                            layers[l].W[p] -= learningRate * gradW[p]
                        }
                        for j in 0..<outD {
                            layers[l].b[j] -= learningRate * gradB[j]
                        }
                    }

                    delta = nextDelta
                }
            }
        }

        self.layers = layers
    }

    public func predict(features: [[Double]]) async throws -> [Int] {
        let probs = try await predictProbability(features: features)
        guard let classes = self.classes else { return [] }
        if classes.count <= 2 {
            return probs.map { $0[1] >= 0.5 ? Int(classes.last ?? 1.0) : Int(classes.first ?? 0.0) }
        } else {
            return probs.map { p in
                let maxIdx = p.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
                return Int(classes[maxIdx])
            }
        }
    }

    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        guard let layers = layers, let classes = classes else {
            throw MLError.modelNotFitted
        }
        guard !features.isEmpty else { return [] }

        var results = [[Double]]()
        results.reserveCapacity(features.count)

        for x in features {
            var curr = x
            for l in 0..<layers.count {
                var next = layers[l].b
                cblas_dgemm(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    1, Int32(layers[l].outDim), Int32(layers[l].inDim),
                    1.0,
                    curr, Int32(layers[l].inDim),
                    layers[l].W, Int32(layers[l].outDim),
                    1.0,
                    &next, Int32(layers[l].outDim)
                )

                if l != layers.count - 1 {
                    for j in 0..<layers[l].outDim {
                        next[j] = applyActivation(next[j], activation: activation)
                    }
                }
                curr = next
            }

            if classes.count <= 2 {
                let p1 = 1.0 / (1.0 + exp(-curr[0]))
                results.append([1.0 - p1, p1])
            } else {
                let maxLogit = curr.max() ?? 0.0
                let exps = curr.map { exp($0 - maxLogit) }
                let sumExp = exps.reduce(0.0, +)
                results.append(exps.map { $0 / sumExp })
            }
        }

        return results
    }
}

/// Multi-Layer Perceptron Regressor.
public actor MLPRegressor: RegressorEstimator {
    public let hiddenLayerSizes: [Int]
    public let activation: ActivationFunction
    public let solver: MLPSolver
    public let maxIter: Int
    public let learningRate: Double
    public let beta1: Double
    public let beta2: Double
    public let epsilon: Double
    public let batchSize: Int
    public let seed: Int
    public let requestedDevice: ExecutionDevice
    public private(set) var resolvedDevice: ExecutionDevice = .cpu

    private var layers: [LayerWeights]?

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

    public func fit(features: [[Double]], targets: [Double]) async throws {
        guard !features.isEmpty, !targets.isEmpty else {
            throw MLError.emptyInput
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

        for _ in 0..<maxIter {
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

                        for p in 0..<layers[l].W.count {
                            let g = gradW[p]
                            adamStates[l].mW[p] = beta1 * adamStates[l].mW[p] + (1.0 - beta1) * g
                            adamStates[l].vW[p] = beta2 * adamStates[l].vW[p] + (1.0 - beta2) * g * g
                            let mHat = adamStates[l].mW[p] / b1_corr
                            let vHat = adamStates[l].vW[p] / b2_corr
                            layers[l].W[p] -= learningRate * mHat / (sqrt(vHat) + epsilon)
                        }

                        for j in 0..<outD {
                            let g = gradB[j]
                            adamStates[l].mB[j] = beta1 * adamStates[l].mB[j] + (1.0 - beta1) * g
                            adamStates[l].vB[j] = beta2 * adamStates[l].vB[j] + (1.0 - beta2) * g * g
                            let mHat = adamStates[l].mB[j] / b1_corr
                            let vHat = adamStates[l].vB[j] / b2_corr
                            layers[l].b[j] -= learningRate * mHat / (sqrt(vHat) + epsilon)
                        }
                    } else {
                        // SGD
                        for p in 0..<layers[l].W.count {
                            layers[l].W[p] -= learningRate * gradW[p]
                        }
                        for j in 0..<outD {
                            layers[l].b[j] -= learningRate * gradB[j]
                        }
                    }

                    delta = nextDelta
                }
            }
        }

        self.layers = layers
    }

    public func predict(features: [[Double]]) async throws -> [Double] {
        guard let layers = layers else {
            throw MLError.modelNotFitted
        }
        guard !features.isEmpty else { return [] }

        var results = [Double]()
        results.reserveCapacity(features.count)

        for x in features {
            var curr = x
            for l in 0..<layers.count {
                var next = layers[l].b
                cblas_dgemm(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    1, Int32(layers[l].outDim), Int32(layers[l].inDim),
                    1.0,
                    curr, Int32(layers[l].inDim),
                    layers[l].W, Int32(layers[l].outDim),
                    1.0,
                    &next, Int32(layers[l].outDim)
                )

                if l != layers.count - 1 {
                    for j in 0..<layers[l].outDim {
                        next[j] = applyActivation(next[j], activation: activation)
                    }
                }
                curr = next
            }
            results.append(curr[0])
        }

        return results
    }
}

private func applyActivation(_ x: Double, activation: ActivationFunction) -> Double {
    switch activation {
    case .relu: return max(0.0, x)
    case .sigmoid: return 1.0 / (1.0 + exp(-x))
    case .tanh: return tanh(x)
    }
}

private func applyActivationDeriv(_ x: Double, activation: ActivationFunction) -> Double {
    switch activation {
    case .relu: return x > 0 ? 1.0 : 0.0
    case .sigmoid:
        let s = 1.0 / (1.0 + exp(-x))
        return s * (1.0 - s)
    case .tanh:
        let t = tanh(x)
        return 1.0 - t * t
    }
}
