import Foundation
import Accelerate

// MARK: - LocalEmbeddingEngine

/// A lightweight, on-device dense text embedding generator optimized for Apple Silicon.
///
/// ``LocalEmbeddingEngine`` projects arbitrary natural language strings into fixed-dimensional
/// dense vectors (\(\mathbb{R}^d\)) with \(L_2\) unit normalization (\(\|v\|_2 = 1.0\)).
/// The produced vectors can be stored and queried directly using `SwiftCluster.VectorStore`.
///
/// ## Features
/// - **Zero API Dependencies**: Operates 100% locally and offline without external network calls.
/// - **Subword Character N-Gram Hashing**: Robust against out-of-vocabulary (OOV) tokens and misspellings.
/// - **SIMD Accelerated**: Leverages Apple Accelerate for instant batch vector normalizations.
///
/// ## Example
/// ```swift
/// import SwiftNLP
///
/// let engine = LocalEmbeddingEngine(dimension: 128)
/// let embedding = engine.embed("Apple Silicon Machine Learning")
/// print(embedding.count) // 128
/// ```
public final class LocalEmbeddingEngine: Sendable {

    /// Dimensionality of the generated dense embedding vectors.
    public let dimension: Int

    /// Minimum character n-gram length.
    public let minNGram: Int

    /// Maximum character n-gram length.
    public let maxNGram: Int

    /// Creates a new local text embedding generator.
    ///
    /// - Parameters:
    ///   - dimension: Output vector length (default `128`).
    ///   - minNGram: Minimum character n-gram length (default `3`).
    ///   - maxNGram: Maximum character n-gram length (default `5`).
    public init(dimension: Int = 128, minNGram: Int = 3, maxNGram: Int = 5) {
        self.dimension = max(16, dimension)
        self.minNGram = max(1, minNGram)
        self.maxNGram = max(minNGram, maxNGram)
    }

    /// Computes a dense \(L_2\)-normalized embedding vector for the provided text.
    ///
    /// - Parameter text: Input string to embed.
    /// - Returns: Dense numerical vector of length ``dimension`` with unit \(L_2\) norm.
    public func embed(_ text: String) -> [Double] {
        let cleaned = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else {
            return [Double](repeating: 0.0, count: dimension)
        }

        var vec = [Double](repeating: 0.0, count: dimension)
        let words = cleaned.split { !$0.isLetter && !$0.isNumber }

        for wordSub in words {
            let word = String(wordSub)
            let wordChars = Array(word)

            // Word-level hash contribution
            let wordHash = abs(word.hashValue) % dimension
            let wordSign: Double = ((word.hashValue & 1) == 0) ? 1.0 : -1.0
            vec[wordHash] += wordSign * 2.0

            // Subword character n-gram contributions
            if wordChars.count >= minNGram {
                for n in minNGram...min(maxNGram, wordChars.count) {
                    for i in 0...(wordChars.count - n) {
                        let ngram = String(wordChars[i..<(i + n)])
                        let h = abs(ngram.hashValue) % dimension
                        let sign: Double = ((ngram.hashValue & 1) == 0) ? 1.0 : -1.0
                        vec[h] += sign * 1.0
                    }
                }
            }
        }

        // L2 Normalization via Accelerate
        var sumSq: Double = 0.0
        vDSP_svesqD(vec, 1, &sumSq, vDSP_Length(dimension))
        let norm = sqrt(sumSq)

        if norm > 1e-12 {
            var scale = 1.0 / norm
            var normalized = [Double](repeating: 0.0, count: dimension)
            vDSP_vsmulD(vec, 1, &scale, &normalized, 1, vDSP_Length(dimension))
            return normalized
        }

        return vec
    }

    /// Computes dense \(L_2\)-normalized embedding vectors for a batch of text strings.
    ///
    /// - Parameter texts: Array of input strings.
    /// - Returns: Array of dense embedding vectors corresponding to the input order.
    public func embedBatch(_ texts: [String]) -> [[Double]] {
        texts.map { embed($0) }
    }
}
