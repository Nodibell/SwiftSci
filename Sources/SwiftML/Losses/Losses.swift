#if os(macOS)
import Foundation
import MLX

/// Helper loss functions for training linear and logistic regression models.
internal enum Losses {
    
    /// Computes the Mean Squared Error (MSE) loss: mean((predictions - targets)^2)
    static func meanSquaredError(predictions: MLXArray, targets: MLXArray) -> MLXArray {
        mean(square(predictions - targets))
    }
    
    /// Computes the numerically stable Binary Cross Entropy (BCE) loss from logits:
    /// Uses log-add-exp so the derivative is sigmoid(logits) - targets, including at zero.
    static func binaryCrossEntropy(logits: MLXArray, targets: MLXArray) -> MLXArray {
        mean(logAddExp(0, logits) - logits * targets)
    }
}
#endif // os(macOS)
