import Foundation
import MLX

/// Helper loss functions for training linear and logistic regression models.
internal enum Losses {
    
    /// Computes the Mean Squared Error (MSE) loss: mean((predictions - targets)^2)
    static func meanSquaredError(predictions: MLXArray, targets: MLXArray) -> MLXArray {
        mean(square(predictions - targets))
    }
    
    /// Computes the numerically stable Binary Cross Entropy (BCE) loss from logits:
    /// loss = mean(max(logits, 0) - logits * targets + log(1 + exp(-abs(logits))))
    static func binaryCrossEntropy(logits: MLXArray, targets: MLXArray) -> MLXArray {
        let maxLogits = maximum(logits, 0.0)
        let absLogits = abs(logits)
        return mean(maxLogits - logits * targets + log(1.0 + exp(-absLogits)))
    }
}
