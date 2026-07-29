#if os(macOS)
import Foundation
import MLX

/// Generates next token predictions from logits using probabilistic sampling.
public struct Sampler: Sendable {
    /// Samples a token ID from logits.
    /// - Parameters:
    ///   - logits: Logits vector for the next token, shape [vocab_size].
    ///   - options: Configuration options (temperature, topK).
    /// - Returns: A sampled token ID.
    public static func sample(logits: MLXArray, options: LLMOptions) -> Int {
        if options.temperature == 0.0 {
            // Greedy search: return argmax
            return MLX.argMax(logits, keepDims: false).item(Int.self)
        }
        
        let scaledLogits = logits / options.temperature
        var filteredLogits = scaledLogits
        
        // Apply Top-K filtering
        if options.topK > 0 && options.topK < logits.shape[0] {
            let sortedLogits = MLX.sorted(filteredLogits)
            let thresholdIndex = sortedLogits.shape[0] - options.topK
            let threshold = sortedLogits[thresholdIndex]
            
            // Mask out values below the threshold
            let mask = MLX.greaterEqual(filteredLogits, threshold)
            filteredLogits = MLX.where(mask, filteredLogits, MLX.full(filteredLogits.shape, values: -1e9))
        }
        
        // Sample from the categorical distribution
        let sampled = MLXRandom.categorical(filteredLogits)
        return sampled.item(Int.self)
    }
}
#endif // os(macOS)
