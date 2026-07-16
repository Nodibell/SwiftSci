import Foundation
import MLX
import SwiftPreprocessing

/// Binary Logistic Regression model using sigmoid activation.
public actor LogisticRegression: ClassifierEstimator {
    public private(set) var weights: MLXArray?
    public private(set) var bias: MLXArray?
    
    public init() {}
    
    /// Fits the classifier model on the provided features and targets (ClassifierEstimator protocol).
    public func fit(features: [[Double]], targets: [Double]) async throws {
        try fit(features: features, targets: targets, learningRate: 0.1, epochs: 1000)
    }
    
    /// Fits the logistic regression model to binary classification data (Sendable interface).
    public func fit(
        features: [[Double]],
        targets: [Double],
        learningRate lr: Float = 0.1,
        epochs: Int = 1000
    ) throws {
        guard !features.isEmpty, !targets.isEmpty else {
            throw MLError.emptyInput
        }
        
        let numSamples = features.count
        let numFeatures = features[0].count
        
        let X = MLXArray(features.flatMap { $0.map { Float($0) } }).reshaped([numSamples, numFeatures])
        let y = MLXArray(targets.map { Float($0) })
        
        try fit(X: X, y: y, learningRate: lr, epochs: epochs)
    }
    
    /// Fits the logistic regression model to binary classification data.
    /// - Parameters:
    ///   - X: A 2D MLXArray of shape [samples, features].
    ///   - y: An MLXArray of labels containing 0 or 1, shape [samples] or [samples, 1].
    ///   - lr: Learning rate.
    ///   - epochs: Number of iterations.
    public func fit(
        X: MLXArray,
        y: MLXArray,
        learningRate lr: Float = 0.1,
        epochs: Int = 1000
    ) throws {
        guard X.size > 0, y.size > 0 else {
            throw MLError.emptyInput
        }
        
        let shape = X.shape
        guard shape.count == 2 else {
            throw MLError.trainingFailed("X must be a 2D matrix of shape [samples, features]")
        }
        
        let numSamples = shape[0]
        let numFeatures = shape[1]
        
        let yShape = y.shape
        guard yShape[0] == numSamples else {
            throw MLError.dimensionMismatch(expected: numSamples, got: yShape[0])
        }
        
        let yReshaped: MLXArray
        if yShape.count == 1 {
            yReshaped = y.reshaped([numSamples, 1])
        } else {
            yReshaped = y
        }
        
        // Initialize weights to zeros (shape [M, 1]) and bias to zero (shape [1])
        var w = MLXArray.zeros([numFeatures, 1])
        var b = MLXArray.zeros([1])
        
        // Define the loss function using binary cross entropy
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
            
            // Force eager evaluation to clear intermediate lazy nodes
            eval(w, b)
            
            // Early divergence diagnostics (NaN/Inf check)
            let wArray = w.asArray(Float.self)
            let bArray = b.asArray(Float.self)
            if wArray.contains(where: { $0.isNaN || $0.isInfinite }) ||
               bArray.contains(where: { $0.isNaN || $0.isInfinite }) {
                throw MLError.trainingFailed("Gradient descent diverged: weights or bias contains NaN or Infinity. Try a lower learning rate.")
            }
        }
        
        self.weights = w
        self.bias = b
    }
    
    /// Predicts target probabilities of class 1 for the given features matrix (Sendable interface).
    public func predictProbability(features: [[Double]]) throws -> [Double] {
        guard !features.isEmpty else {
            return []
        }
        
        let numSamples = features.count
        let numFeatures = features[0].count
        
        let X = MLXArray(features.flatMap { $0.map { Float($0) } }).reshaped([numSamples, numFeatures])
        let probs = try predictProbability(X: X)
        
        return probs.asArray(Float.self).map { Double($0) }
    }
    
    /// Predicts target probabilities of class 1 for the given features X.
    /// - Parameter X: A 2D MLXArray of shape [samples, features].
    /// - Returns: An MLXArray of probabilities between [0, 1].
    public func predictProbability(X: MLXArray) throws -> MLXArray {
        guard let weights = self.weights, let bias = self.bias else {
            throw MLError.modelNotFitted
        }
        
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
        
        let numSamples = features.count
        let numFeatures = features[0].count
        
        let X = MLXArray(features.flatMap { $0.map { Float($0) } }).reshaped([numSamples, numFeatures])
        let preds = try predict(X: X, threshold: threshold)
        
        return preds.asArray(Int32.self).map { Int($0) }
    }
    
    /// Predicts class labels (0 or 1) for the given features X.
    /// - Parameters:
    ///   - X: A 2D MLXArray of shape [samples, features].
    ///   - threshold: Decision threshold (default 0.5).
    /// - Returns: An MLXArray of class labels.
    public func predict(X: MLXArray, threshold: Float = 0.5) throws -> MLXArray {
        let probs = try predictProbability(X: X)
        return greater(probs, threshold).asType(.int32)
    }
    
    /// Returns the learned weights as a standard Sendable Double array.
    public func getWeights() -> [Double]? {
        return weights?.asArray(Float.self).map { Double($0) }
    }
    
    /// Returns the learned bias as a standard Sendable Double array.
    public func getBias() -> Double? {
        if let biasValue = bias {
            return Double(biasValue.item(Float.self))
        }
        return nil
    }
}
