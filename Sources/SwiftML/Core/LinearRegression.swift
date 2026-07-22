import Foundation
import MLX
import SwiftPreprocessing

/// Ordinary Least Squares linear regression using gradient descent.
public actor LinearRegression: RegressorEstimator {
    public private(set) var weights: MLXArray?
    public private(set) var bias: MLXArray?
    
    public let requestedDevice: ExecutionDevice
    public private(set) var resolvedDevice: ExecutionDevice?
    
    private var cpuWeights: [Double]?
    private var cpuBias: Double?
    
    public init(device: ExecutionDevice = .auto) {
        self.requestedDevice = device
    }

    public init(weights: [Double], bias: Double, device: ExecutionDevice = .auto) {
        self.requestedDevice = device
        self.cpuWeights = weights
        self.cpuBias = bias
    }

    public func getWeightsAndBias() -> (weights: [Double]?, bias: Double?) {
        return (cpuWeights, cpuBias)
    }
    
    /// Fits the regressor model on the provided features and targets (RegressorEstimator protocol).
    public func fit(features: [[Double]], targets: [Double]) async throws {
        try await fit(features: features, targets: targets, learningRate: 0.01, epochs: 1000)
    }
    
    /// Fits the linear regression model to the features X and target values y (Sendable interface).
    public func fit(
        features: [[Double]],
        targets: [Double],
        learningRate lr: Float = 0.01,
        epochs: Int = 1000
    ) async throws {
        guard !features.isEmpty, !targets.isEmpty else {
            throw MLError.emptyInput
        }
        
        let numSamples = features.count
        let numFeatures = features[0].count
        
        let device = await HardwareRouter.shared.resolveDevice(
            for: "LinearRegression",
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
                var sum = 0.0
                for j in 0..<numFeatures {
                    sum += features[i][j] * w[j]
                }
                predictions[i] = sum + b
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
            
            let factor = 2.0 / Double(numSamples)
            gradB *= factor
            for j in 0..<numFeatures {
                gradW[j] *= factor
            }
            
            // Check for NaN or Inf to early terminate
            if gradB.isNaN || gradB.isInfinite || gradW.contains(where: { $0.isNaN || $0.isInfinite }) {
                throw MLError.trainingFailed("Gradient descent diverged: weights or bias contains NaN or Infinity. Try a lower learning rate.")
            }
            
            // Weights updates
            for j in 0..<numFeatures {
                w[j] -= lr * gradW[j]
            }
            b -= lr * gradB
        }
        
        self.cpuWeights = w
        self.cpuBias = b
        
        // Sync to MLXArray variables for public API compatibility
        self.weights = MLXArray(w.map { Float($0) }).reshaped([numFeatures, 1])
        self.bias = MLXArray([Float(b)])
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
            let predictions = matmul(X, w) + b
            return [Losses.meanSquaredError(predictions: predictions, targets: yReshaped)]
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
        
        // Sync to CPU variables
        self.cpuWeights = w.asArray(Float.self).map { Double($0) }
        self.cpuBias = Double(b.item(Float.self))
    }
    
    /// Predicts target values for the given features matrix (Sendable interface).
    public func predict(features: [[Double]]) throws -> [Double] {
        guard !features.isEmpty else {
            return []
        }
        
        if resolvedDevice == .cpu, let w = cpuWeights, let b = cpuBias {
            let numFeatures = w.count
            return features.map { row in
                var sum = 0.0
                for j in 0..<numFeatures {
                    sum += row[j] * w[j]
                }
                return sum + b
            }
        }
        
        let numSamples = features.count
        let numFeatures = features[0].count
        
        let X = MLXArray(features.flatMap { $0.map { Float($0) } }).reshaped([numSamples, numFeatures])
        let preds = try predict(X: X)
        
        return preds.asArray(Float.self).map { Double($0) }
    }
    
    /// Predicts targets for the given feature matrix X (MLX interface).
    public func predict(X: MLXArray) throws -> MLXArray {
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
        
        return matmul(X, weights) + bias
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
