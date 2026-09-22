#if os(macOS)
import Foundation
import MLX

/// Generates next token predictions from logits using probabilistic sampling.
public struct Sampler: Sendable {
    /// Samples a token ID from logits using a decoupled SamplingConfiguration.
    ///
    /// Pipeline:
    /// ```
    /// Logits ──► Repetition Penalty ──► Temperature ──► Top-K Mask (-inf) ──► Top-P Mask (-inf) ──► Softmax ──► Sample
    /// ```
    ///
    /// - Parameters:
    ///   - logits: Logits vector for the next token, shape `[vocab_size]` or `[1, vocab_size]`.
    ///   - config: Sampling parameters (temperature, topK, topP, repetitionPenalty).
    ///   - pastTokens: Array of previously generated token IDs to penalize against repetition.
    /// - Returns: A sampled token ID.
    public static func sample(
        logits: MLXArray,
        config: SamplingConfiguration,
        pastTokens: [Int] = []
    ) -> Int {
        var filteredLogits = logits.ndim > 1 ? logits.squeezed() : logits
        if filteredLogits.dtype == .float64 {
            filteredLogits = filteredLogits.asType(.float32)
        }

        // 1. Repetition Penalty:
        // if logit > 0: logit / penalty; if logit <= 0: logit * penalty
        if config.repetitionPenalty != 1.0 && !pastTokens.isEmpty {
            let vocabSize = filteredLogits.shape[0]
            let uniqueTokens = Array(Set(pastTokens)).filter { $0 >= 0 && $0 < vocabSize }
            if !uniqueTokens.isEmpty {
                let idx = MLXArray(uniqueTokens.map { Int32($0) })
                let pastLogits = filteredLogits[idx]
                let penalty = MLXArray(config.repetitionPenalty)
                let penalized = MLX.where(pastLogits .> 0, pastLogits / penalty, pastLogits * penalty)
                filteredLogits[idx] = penalized
            }
        }

        // 2. Temperature:
        if config.temperature == 0.0 {
            // Greedy search: return argmax
            return MLX.argMax(filteredLogits, keepDims: false).item(Int.self)
        }

        if config.temperature != 1.0 {
            filteredLogits = filteredLogits / config.temperature
        }

        let vocabSize = filteredLogits.shape[0]

        // 3. Top-K Masking (on logits before softmax):
        if config.topK > 0 && config.topK < vocabSize {
            let sortedLogits = MLX.sorted(filteredLogits)
            let thresholdIndex = vocabSize - config.topK
            let threshold = sortedLogits[thresholdIndex]
            let mask = MLX.greaterEqual(filteredLogits, threshold)
            filteredLogits = MLX.where(mask, filteredLogits, MLX.full(filteredLogits.shape, values: -Float.infinity))
        }

        // 4. Top-P (Nucleus) Masking (on logits before softmax):
        if config.topP < 1.0 && config.topP > 0.0 && vocabSize > 1 {
            let probs = softmax(filteredLogits, axis: -1)
            let sortedIndices = MLX.argSort(-probs, axis: -1)
            let sortedProbs = probs[sortedIndices]
            let cumProbs = MLX.cumsum(sortedProbs, axis: -1)

            // A token is masked if cumulative sum of all tokens strictly before it already exceeds topP
            let maskSorted = cumProbs .> config.topP
            let shiftedMask = MLX.concatenated([MLXArray([false]), maskSorted[0..<(vocabSize - 1)]], axis: 0)

            let maskOriginal = MLX.zeros(filteredLogits.shape, dtype: .bool)
            maskOriginal[sortedIndices] = shiftedMask
            filteredLogits = MLX.where(maskOriginal, MLX.full(filteredLogits.shape, values: -Float.infinity), filteredLogits)
        }

        // 5. Sample from the categorical distribution
        let sampled = MLXRandom.categorical(filteredLogits)
        return sampled.item(Int.self)
    }

    /// Samples a token ID from logits using LLMOptions.
    ///
    /// - Parameters:
    ///   - logits: Logits vector for the next token, shape [vocab_size].
    ///   - options: Configuration options containing sampling settings.
    ///   - pastTokens: Array of previously generated token IDs to penalize against repetition.
    /// - Returns: A sampled token ID.
    public static func sample(
        logits: MLXArray,
        options: LLMOptions,
        pastTokens: [Int] = []
    ) -> Int {
        sample(logits: logits, config: options.sampling, pastTokens: pastTokens)
    }
}
#endif // os(macOS)
