#if os(macOS)
import Foundation
import MLX
import SwiftPreprocessing

/// Linear Support Vector Classifier (LinearSVC) with L2 regularization and Hinge / Squared Hinge Loss.
/// Supports both CPU execution and MLX Metal GPU acceleration on Apple Silicon.
public actor LinearSVC: ClassifierEstimator {
    /// The weights array.
    public private(set) var weights: MLXArray?
    /// The bias array.
    public private(set) var bias: MLXArray?
    
    /// Regularization parameter C (inverse regularization strength).
    public let C: Double
    /// The requested execution device.
    public let requestedDevice: ExecutionDevice
    /// The resolved execution device.
    public private(set) var resolvedDevice: ExecutionDevice?
    
    private var cpuWeights: [Double]?
    private var cpuBias: Double?
    
    /// Creates a new LinearSVC classifier instance.
    /// - Parameters:
    ///   - C: Regularization parameter (default: 1.0).
    ///   - device: The execution device (.auto, .gpu, or .cpu).
    public init(C: Double = 1.0, device: ExecutionDevice = .auto) {
        self.C = max(1e-5, C)
        self.requestedDevice = device
    }
    
    /// Creates a restored LinearSVC instance with weights and bias.
    public init(C: Double = 1.0, weights: [Double], bias: Double, device: ExecutionDevice = .auto) {
        self.C = max(1e-5, C)
        self.requestedDevice = device
        self.cpuWeights = weights
        self.cpuBias = bias
    }
    
    /// Returns trained weights and bias.
    public func getWeightsAndBias() -> (weights: [Double]?, bias: Double?) {
        return (cpuWeights, cpuBias)
    }
    
    /// Fits the LinearSVC classifier (ClassifierEstimator protocol).
    public func fit(features: [[Double]], targets: [Double]) async throws {
        try await fit(features: features, targets: targets, learningRate: 0.1, epochs: 800)
    }
    
    /// Fits the LinearSVC model with custom learning rate and epochs.
    public func fit(
        features: [[Double]],
        targets: [Double],
        learningRate lr: Float = 0.1,
        epochs: Int = 800
    ) async throws {
        guard !features.isEmpty else {
            throw MLError.emptyInput
        }
        guard features.count == targets.count else {
            throw MLError.invalidInput("Features count (\(features.count)) != targets count (\(targets.count))")
        }
        
        let numFeatures = features[0].count
        guard numFeatures > 0 else {
            throw MLError.invalidInput("Feature dimension must be > 0")
        }
        
        let router = HardwareRouter.shared
        let deviceToUse = await router.resolveDevice(
            for: "LinearSVC",
            sampleCount: features.count,
            featureCount: numFeatures,
            requestedDevice: requestedDevice
        )
        self.resolvedDevice = deviceToUse
        
        if deviceToUse == .gpu {
            let flatX = features.flatMap { $0.map { Float($0) } }
            let X = MLXArray(flatX).reshaped([features.count, numFeatures])
            
            // Map 0/1 targets to -1/+1 for SVM Hinge loss
            let svmTargets = targets.map { $0 > 0.5 ? Float(1.0) : Float(-1.0) }
            let y = MLXArray(svmTargets).reshaped([targets.count, 1])
            
            try fitGPU(X: X, y: y, learningRate: lr, epochs: epochs)
        } else {
            try fitCPU(features: features, targets: targets, learningRate: Double(lr), epochs: epochs)
        }
    }
    
    // MARK: - GPU Backend (MLX Metal)
    
    private func fitGPU(
        X: MLXArray,
        y: MLXArray,
        learningRate lr: Float,
        epochs: Int
    ) throws {
        let shape = X.shape
        let numSamples = shape[0]
        let numFeatures = shape[1]
        
        var w = MLXArray.zeros([numFeatures, 1])
        var b = MLXArray.zeros([1])
        
        let regWeight = Float(1.0 / (C * Double(numSamples)))
        
        func lossFn(params: [MLXArray]) -> [MLXArray] {
            let w = params[0]
            let b = params[1]
            let margin = 1.0 - y * (matmul(X, w) + b)
            let hingeLoss = mean(maximum(margin, 0.0) * maximum(margin, 0.0))
            let l2Loss = 0.5 * regWeight * sum(w * w)
            return [hingeLoss + l2Loss]
        }
        
        let gradFn = valueAndGrad(lossFn, argumentNumbers: [0, 1])
        
        for _ in 0..<epochs {
            let (_, grads) = gradFn([w, b])
            w = w - lr * grads[0]
            b = b - lr * grads[1]
        }
        
        eval(w, b)
        let wArray = w.asArray(Float.self)
        let bArray = b.asArray(Float.self)
        
        if wArray.contains(where: { $0.isNaN || $0.isInfinite }) ||
           bArray.contains(where: { $0.isNaN || $0.isInfinite }) {
            throw MLError.trainingFailed("Gradient descent diverged in LinearSVC. Try a lower learning rate.")
        }
        
        self.weights = w
        self.bias = b
        self.cpuWeights = wArray.map { Double($0) }
        self.cpuBias = Double(b.item(Float.self))
    }
    
    // MARK: - CPU Backend
    
    private func fitCPU(
        features: [[Double]],
        targets: [Double],
        learningRate lr: Double,
        epochs: Int
    ) throws {
        let numSamples = features.count
        let numFeatures = features[0].count
        
        var w = [Double](repeating: 0.0, count: numFeatures)
        var b = 0.0
        let reg = 1.0 / (C * Double(numSamples))
        
        for epoch in 1...epochs {
            let currentLr = lr / (1.0 + 0.001 * Double(epoch))
            var gradW = [Double](repeating: 0.0, count: numFeatures)
            var gradB = 0.0
            
            for i in 0..<numSamples {
                let y_i = targets[i] > 0.5 ? 1.0 : -1.0
                var margin = b
                for j in 0..<numFeatures {
                    margin += w[j] * features[i][j]
                }
                
                let lossVal = 1.0 - y_i * margin
                if lossVal > 0.0 {
                    let factor = -2.0 * lossVal * y_i / Double(numSamples)
                    for j in 0..<numFeatures {
                        gradW[j] += factor * features[i][j]
                    }
                    gradB += factor
                }
            }
            
            for j in 0..<numFeatures {
                gradW[j] += reg * w[j]
                w[j] -= currentLr * gradW[j]
            }
            b -= currentLr * gradB
        }
        
        self.cpuWeights = w
        self.cpuBias = b
        self.weights = MLXArray(w.map { Float($0) }).reshaped([numFeatures, 1])
        self.bias = MLXArray([Float(b)])
    }
    
    /// Predicts binary class labels (0 or 1) for feature matrix.
    public func predict(features: [[Double]]) async throws -> [Int] {
        let decisionValues = try decisionFunction(features: features)
        return decisionValues.map { $0 >= 0.0 ? 1 : 0 }
    }
    
    /// Computes raw SVM decision function values (w^T x + b).
    public func decisionFunction(features: [[Double]]) throws -> [Double] {
        guard !features.isEmpty else { return [] }
        
        if resolvedDevice == .cpu || cpuWeights != nil, let w = cpuWeights, let b = cpuBias {
            let numFeatures = w.count
            return features.map { row in
                var score = b
                let count = min(row.count, numFeatures)
                for j in 0..<count {
                    score += w[j] * row[j]
                }
                return score
            }
        }
        
        guard let w = weights, let b = bias else {
            throw SwiftMLError.modelNotFitted
        }
        
        let numFeatures = w.shape[0]
        let flatX = features.flatMap { $0.map { Float($0) } }
        let X = MLXArray(flatX).reshaped([features.count, numFeatures])
        let scores = matmul(X, w) + b
        eval(scores)
        return scores.asArray(Float.self).map { Double($0) }
    }
    
    /// Predicts probabilities using sigmoid calibration over decision values.
    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        let scores = try decisionFunction(features: features)
        return scores.map { score in
            let prob1 = sigmoid(score)
            return [1.0 - prob1, prob1]
        }

    }
}
#endif
