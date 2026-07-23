import Foundation
import Accelerate

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

/// Multi-Layer Perceptron Classifier.
public actor MLPClassifier: ClassifierEstimator {
    public let hiddenLayerSizes: [Int]
    public let activation: ActivationFunction
    public let solver: MLPSolver
    public let maxIter: Int
    public let learningRate: Double
    public let seed: Int

    private var weights: [[[Double]]]?
    private var biases: [[Double]]?
    private var classes: [Double]?

    public init(
        hiddenLayerSizes: [Int] = [100],
        activation: ActivationFunction = .relu,
        solver: MLPSolver = .adam,
        maxIter: Int = 200,
        learningRate: Double = 1e-3,
        seed: Int = 42
    ) {
        self.hiddenLayerSizes = hiddenLayerSizes
        self.activation = activation
        self.solver = solver
        self.maxIter = maxIter
        self.learningRate = learningRate
        self.seed = seed
    }

    public func fit(features: [[Double]], targets: [Double]) async throws {
        guard !features.isEmpty, !targets.isEmpty else {
            throw MLError.emptyInput
        }
        let numSamples = features.count
        let numFeatures = features[0].count

        let uniqueClasses = Array(Set(targets)).sorted()
        self.classes = uniqueClasses
        let numClasses = max(1, uniqueClasses.count)

        let layerSizes = [numFeatures] + hiddenLayerSizes + [numClasses > 2 ? numClasses : 1]
        var rng = SeededRandom(seed: seed)

        var w = [[[Double]]]()
        var b = [[Double]]()

        for l in 0..<(layerSizes.count - 1) {
            let inDim = layerSizes[l]
            let outDim = layerSizes[l + 1]
            let limit = sqrt(6.0 / Double(inDim + outDim))
            var wMat = [[Double]]()
            wMat.reserveCapacity(inDim)
            for _ in 0..<inDim {
                var row = [Double]()
                row.reserveCapacity(outDim)
                for _ in 0..<outDim {
                    row.append(rng.nextDouble() * 2.0 * limit - limit)
                }
                wMat.append(row)
            }
            w.append(wMat)
            b.append([Double](repeating: 0.0, count: outDim))
        }

        for _ in 0..<maxIter {
            for i in 0..<numSamples {
                let x = features[i]
                let yVal = targets[i]

                var activations = [x]
                for l in 0..<w.count {
                    let prev = activations[l]
                    let inD = w[l].count
                    let outD = w[l][0].count
                    var out = b[l]

                    for j in 0..<outD {
                        var sum = out[j]
                        for k in 0..<inD {
                            sum += prev[k] * w[l][k][j]
                        }
                        out[j] = l == w.count - 1 ? (numClasses == 2 ? 1.0 / (1.0 + exp(-sum)) : sum) : applyActivation(sum, activation: activation)
                    }
                    activations.append(out)
                }

                var delta = [Double]()
                let lastIdx = activations.count - 1
                let lastOut = activations[lastIdx]

                if numClasses <= 2 {
                    let err = lastOut[0] - (yVal == (uniqueClasses.last ?? 1.0) ? 1.0 : 0.0)
                    delta = [err]
                } else {
                    delta = lastOut.enumerated().map { (cIdx, v) in
                        v - (uniqueClasses[cIdx] == yVal ? 1.0 : 0.0)
                    }
                }

                for l in stride(from: w.count - 1, through: 0, by: -1) {
                    let prevAct = activations[l]
                    let inD = prevAct.count
                    let outD = delta.count

                    var nextDelta = [Double](repeating: 0.0, count: inD)
                    for k in 0..<inD {
                        var sum = 0.0
                        for j in 0..<outD {
                            w[l][k][j] -= learningRate * delta[j] * prevAct[k]
                            sum += delta[j] * w[l][k][j]
                        }
                        nextDelta[k] = sum * applyActivationDeriv(prevAct[k], activation: activation)
                    }
                    for j in 0..<outD {
                        b[l][j] -= learningRate * delta[j]
                    }
                    delta = nextDelta
                }
            }
        }

        self.weights = w
        self.biases = b
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
        guard let w = weights, let b = biases, let classes = classes else {
            throw MLError.modelNotFitted
        }
        guard !features.isEmpty else { return [] }

        var results = [[Double]]()
        results.reserveCapacity(features.count)

        for x in features {
            var curr = x
            for l in 0..<w.count {
                let inD = w[l].count
                let outD = w[l][0].count
                var next = b[l]
                for j in 0..<outD {
                    var sum = next[j]
                    for k in 0..<inD {
                        sum += curr[k] * w[l][k][j]
                    }
                    next[j] = l == w.count - 1 ? sum : applyActivation(sum, activation: activation)
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
    public let seed: Int

    private var weights: [[[Double]]]?
    private var biases: [[Double]]?

    public init(
        hiddenLayerSizes: [Int] = [100],
        activation: ActivationFunction = .relu,
        solver: MLPSolver = .adam,
        maxIter: Int = 200,
        learningRate: Double = 1e-3,
        seed: Int = 42
    ) {
        self.hiddenLayerSizes = hiddenLayerSizes
        self.activation = activation
        self.solver = solver
        self.maxIter = maxIter
        self.learningRate = learningRate
        self.seed = seed
    }

    public func fit(features: [[Double]], targets: [Double]) async throws {
        guard !features.isEmpty, !targets.isEmpty else {
            throw MLError.emptyInput
        }
        let numSamples = features.count
        let numFeatures = features[0].count

        let layerSizes = [numFeatures] + hiddenLayerSizes + [1]
        var rng = SeededRandom(seed: seed)

        var w = [[[Double]]]()
        var b = [[Double]]()

        for l in 0..<(layerSizes.count - 1) {
            let inDim = layerSizes[l]
            let outDim = layerSizes[l + 1]
            let limit = sqrt(6.0 / Double(inDim + outDim))
            var wMat = [[Double]]()
            wMat.reserveCapacity(inDim)
            for _ in 0..<inDim {
                var row = [Double]()
                row.reserveCapacity(outDim)
                for _ in 0..<outDim {
                    row.append(rng.nextDouble() * 2.0 * limit - limit)
                }
                wMat.append(row)
            }
            w.append(wMat)
            b.append([Double](repeating: 0.0, count: outDim))
        }

        for _ in 0..<maxIter {
            for i in 0..<numSamples {
                let x = features[i]
                let yVal = targets[i]

                var activations = [x]
                for l in 0..<w.count {
                    let prev = activations[l]
                    let inD = w[l].count
                    let outD = w[l][0].count
                    var out = b[l]

                    for j in 0..<outD {
                        var sum = out[j]
                        for k in 0..<inD {
                            sum += prev[k] * w[l][k][j]
                        }
                        out[j] = l == w.count - 1 ? sum : applyActivation(sum, activation: activation)
                    }
                    activations.append(out)
                }

                var delta = [activations.last![0] - yVal]
                for l in stride(from: w.count - 1, through: 0, by: -1) {
                    let prevAct = activations[l]
                    let inD = prevAct.count
                    let outD = delta.count

                    var nextDelta = [Double](repeating: 0.0, count: inD)
                    for k in 0..<inD {
                        var sum = 0.0
                        for j in 0..<outD {
                            w[l][k][j] -= learningRate * delta[j] * prevAct[k]
                            sum += delta[j] * w[l][k][j]
                        }
                        nextDelta[k] = sum * applyActivationDeriv(prevAct[k], activation: activation)
                    }
                    for j in 0..<outD {
                        b[l][j] -= learningRate * delta[j]
                    }
                    delta = nextDelta
                }
            }
        }

        self.weights = w
        self.biases = b
    }

    public func predict(features: [[Double]]) async throws -> [Double] {
        guard let w = weights, let b = biases else {
            throw MLError.modelNotFitted
        }
        guard !features.isEmpty else { return [] }

        var results = [Double]()
        results.reserveCapacity(features.count)

        for x in features {
            var curr = x
            for l in 0..<w.count {
                let inD = w[l].count
                let outD = w[l][0].count
                var next = b[l]
                for j in 0..<outD {
                    var sum = next[j]
                    for k in 0..<inD {
                        sum += curr[k] * w[l][k][j]
                    }
                    next[j] = l == w.count - 1 ? sum : applyActivation(sum, activation: activation)
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
