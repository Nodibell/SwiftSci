import Foundation
import MLX
import SwiftPreprocessing

/// Binary Logistic Regression model using sigmoid activation.
public actor LogisticRegression: ClassifierEstimator {
    public private(set) var weights: MLXArray?
    public private(set) var bias: MLXArray?
    
    public let requestedDevice: ExecutionDevice
    public private(set) var resolvedDevice: ExecutionDevice?
    
    private var cpuWeights: [Double]?
    private var cpuBias: Double?
    
    public init(device: ExecutionDevice = .auto) {
        self.requestedDevice = device
    }
    
    /// Fits the classifier model on the provided features and targets (ClassifierEstimator protocol).
    public func fit(features: [[Double]], targets: [Double]) async throws {
        try await fit(features: features, targets: targets, learningRate: 0.1, epochs: 1000)
    }
    
    /// Fits the logistic regression model to binary classification data (Sendable interface).
    public func fit(
        features: [[Double]],
        targets: [Double],
        learningRate lr: Float = 0.1,
        epochs: Int = 1000
    ) async throws {
        guard !features.isEmpty, !targets.isEmpty else {
            throw MLError.emptyInput
        }
        
        let numSamples = features.count
        let numFeatures = features[0].count
        
        let device = await HardwareRouter.shared.resolveDevice(
            for: "LogisticRegression",
            sampleCount: numSamples,
            featureCount: numFeatures,
            requestedDevice: requestedDevice
        )
        self.resolvedDevice = device
        
        switch device {
        case .cpu:
            try fitCPU(features: features, targets: targets, learningRate: Double(lr), epochs: epochs)
        case .gpu, .ane, .auto:
            try Device.withDefaultDevice(.gpu) {
                let X = MLXArray(features.flatMap { $0.map { Float($0) } }).reshaped([numSamples, numFeatures])
                let y = MLXArray(targets.map { Float($0) })
                try fitGPU(X: X, y: y, learningRate: lr, epochs: epochs)
            }
        }
    }
    
    // MARK: - CPU Backend
    
    private func fitCPU(features: [[Double]], targets: [Double], learningRate lr: Double, epochs: Int) throws {
        let numSamples = features.count
        let numFeatures = features[0].count
        
        var w = [Double](repeating: 0.0, count: numFeatures)
        var b = 0.0
        
        for _ in 0..<epochs {
            var predictions = [Double](repeating: 0.0, count: numSamples)
            for i in 0..<numSamples {
                var z = 0.0
                for j in 0..<numFeatures {
                    z += features[i][j] * w[j]
                }
                predictions[i] = stableSigmoid(z + b)
            }
            
            var gradW = [Double](repeating: 0.0, count: numFeatures)
            var gradB = 0.0
            
            for i in 0..<numSamples {
                let diff = predictions[i] - targets[i]
                gradB += diff
                for j in 0..<numFeatures {
                    gradW[j] += diff * features[i][j]
                }
            }
            
            let factor = 1.0 / Double(numSamples)
            gradB *= factor
            for j in 0..<numFeatures {
                gradW[j] *= factor
            }
            
            if gradB.isNaN || gradB.isInfinite || gradW.contains(where: { $0.isNaN || $0.isInfinite }) {
                throw MLError.trainingFailed("Gradient descent diverged: weights or bias contains NaN or Infinity. Try a lower learning rate.")
            }
            
            for j in 0..<numFeatures {
                w[j] -= lr * gradW[j]
            }
            b -= lr * gradB
        }
        
        self.cpuWeights = w
        self.cpuBias = b
        
        self.weights = MLXArray(w.map { Float($0) }).reshaped([numFeatures, 1])
        self.bias = MLXArray([Float(b)])
    }
    
    private func stableSigmoid(_ z: Double) -> Double {
        if z >= 0.0 {
            return 1.0 / (1.0 + exp(-z))
        } else {
            let zExp = exp(z)
            return zExp / (1.0 + zExp)
        }
    }
    
    // MARK: - GPU Backend (MLX)
    
    private func fitGPU(
        X: MLXArray,
        y: MLXArray,
        learningRate lr: Float,
        epochs: Int
    ) throws {
        let shape = X.shape
        let numSamples = shape[0]
        let numFeatures = shape[1]
        
        let yShape = y.shape
        let yReshaped: MLXArray
        if yShape.count == 1 {
            yReshaped = y.reshaped([numSamples, 1])
        } else {
            yReshaped = y
        }
        
        var w = MLXArray.zeros([numFeatures, 1])
        var b = MLXArray.zeros([1])
        
        func lossFn(params: [MLXArray]) -> [MLXArray] {
            let w = params[0]
            let b = params[1]
            let logits = matmul(X, w) + b
            return [Losses.binaryCrossEntropy(logits: logits, targets: yReshaped)]
        }
        
        let gradFn = valueAndGrad(lossFn, argumentNumbers: [0, 1])
        
        for _ in 0..<epochs {
            let (_, grads) = gradFn([w, b])
            w = w - lr * grads[0]
            b = b - lr * grads[1]
            
            eval(w, b)
            
            let wArray = w.asArray(Float.self)
            let bArray = b.asArray(Float.self)
            if wArray.contains(where: { $0.isNaN || $0.isInfinite }) ||
               bArray.contains(where: { $0.isNaN || $0.isInfinite }) {
                throw MLError.trainingFailed("Gradient descent diverged: weights or bias contains NaN or Infinity. Try a lower learning rate.")
            }
        }
        
        self.weights = w
        self.bias = b
        
        self.cpuWeights = w.asArray(Float.self).map { Double($0) }
        self.cpuBias = Double(b.item(Float.self))
    }
    
    /// Predicts target probabilities of class 1 for the given features matrix (Sendable interface).
    public func predictProbability1D(features: [[Double]]) throws -> [Double] {
        guard !features.isEmpty else {
            return []
        }
        
        if resolvedDevice == .cpu, let w = cpuWeights, let b = cpuBias {
            let numFeatures = w.count
            return features.map { row in
                var z = 0.0
                for j in 0..<numFeatures {
                    z += row[j] * w[j]
                }
                return stableSigmoid(z + b)
            }
        }
        
        let numSamples = features.count
        let numFeatures = features[0].count
        
        let X = MLXArray(features.flatMap { $0.map { Float($0) } }).reshaped([numSamples, numFeatures])
        let probs = try predictProbability(X: X)
        
        return probs.asArray(Float.self).map { Double($0) }
    }

    /// Predicts class probabilities [[prob_class_0, prob_class_1]] for the given features matrix (ClassifierEstimator protocol).
    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        let p1 = try predictProbability1D(features: features)
        return p1.map { [1.0 - $0, $0] }
    }
    
    /// Predicts target probabilities of class 1 for the given features X.
    public func predictProbability(X: MLXArray) throws -> MLXArray {
        guard let weights = self.weights, let bias = self.bias else {
            throw MLError.modelNotFitted
        }
        guard X.size > 0 else { throw MLError.emptyInput }
        
        let shape = X.shape
        guard shape.count == 2 else {
            throw MLError.dimensionMismatch(expected: 2, got: shape.count)
        }
        
        let numFeatures = weights.shape[0]
        guard shape[1] == numFeatures else {
            throw MLError.dimensionMismatch(expected: numFeatures, got: shape[1])
        }
        
        let logits = matmul(X, weights) + bias
        return sigmoid(logits)
    }
    
    /// Predicts class labels (ClassifierEstimator protocol).
    public func predict(features: [[Double]]) async throws -> [Int] {
        try predict(features: features, threshold: 0.5)
    }
    
    /// Predicts class labels (0 or 1) for the given features matrix (Sendable interface).
    public func predict(features: [[Double]], threshold: Float = 0.5) throws -> [Int] {
        guard !features.isEmpty else {
            return []
        }
        
        let probs = try predictProbability1D(features: features)
        return probs.map { $0 > Double(threshold) ? 1 : 0 }
    }
    
    /// Predicts class labels (0 or 1) for the given features X.
    public func predict(X: MLXArray, threshold: Float = 0.5) throws -> MLXArray {
        let probs = try predictProbability(X: X)
        return greater(probs, threshold).asType(.int32)
    }
    
    /// Returns the learned weights as a standard Sendable Double array.
    public func getWeights() -> [Double]? {
        if let cpuWeights { return cpuWeights }
        return weights?.asArray(Float.self).map { Double($0) }
    }
    
    /// Returns the learned bias as a standard Sendable Double array.
    public func getBias() -> Double? {
        if let cpuBias { return cpuBias }
        if let biasValue = bias {
            return Double(biasValue.item(Float.self))
        }
        return nil
    }
}
