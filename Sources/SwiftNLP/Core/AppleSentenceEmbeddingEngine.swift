import Foundation
import Accelerate
#if canImport(NaturalLanguage)
import NaturalLanguage
#endif

/// A sentence embedding generator powered by Apple's native pre-trained `NLEmbedding` models.
///
/// `AppleSentenceEmbeddingEngine` provides offline, high-quality sentence embeddings on Apple Silicon
/// with zero external downloads or network latency. Output vectors are $L_2$-normalized
/// to unit length (\(\|v\|_2 = 1.0\)) using SIMD Accelerate kernels.
public final class AppleSentenceEmbeddingEngine: SentenceEmbeddingProvider, @unchecked Sendable {

    /// Dimensionality of the sentence embedding space (typically 512 for English).
    public let dimension: Int

    #if canImport(NaturalLanguage)
    private let embedding: NLEmbedding
    #endif

    /// Initializes a sentence embedding engine using the specified language.
    ///
    /// - Parameter language: The language vector space. Defaults to `.english`.
    /// - Throws: `NLPError.modelNotFound` if Apple's system embedding for the language is unavailable,
    ///   or `NLPError.unavailableOnPlatform` if `NaturalLanguage` is not supported on the target OS.
    public init(language: AppleNLEmbedding.Language = .english) throws {
        #if canImport(NaturalLanguage)
        let nlLang = NLLanguage(rawValue: language.rawValue)
        guard let emb = NLEmbedding.sentenceEmbedding(for: nlLang) else {
            throw NLPError.modelNotFound("NLEmbedding.sentenceEmbedding for language '\(language.rawValue)'")
        }
        self.embedding = emb
        self.dimension = emb.dimension
        #else
        throw NLPError.unavailableOnPlatform(feature: "AppleSentenceEmbeddingEngine")
        #endif
    }

    #if canImport(NaturalLanguage)
    /// Initializes a sentence embedding engine using an explicit `NLLanguage`.
    ///
    /// - Parameter nlLanguage: Apple NaturalLanguage language identifier.
    /// - Throws: `NLPError.modelNotFound` if the system embedding is not available.
    public init(nlLanguage: NLLanguage) throws {
        guard let emb = NLEmbedding.sentenceEmbedding(for: nlLanguage) else {
            throw NLPError.modelNotFound("NLEmbedding.sentenceEmbedding for language '\(nlLanguage.rawValue)'")
        }
        self.embedding = emb
        self.dimension = emb.dimension
    }
    #endif

    /// Computes an $L_2$-normalized dense embedding vector for the given text.
    ///
    /// - Parameter text: Input string to embed.
    /// - Throws: `NLPError` if embedding lookup fails.
    /// - Returns: Dense numerical vector of length ``dimension`` with unit $L_2$ norm.
    public func embed(_ text: String) throws -> [Double] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return [Double](repeating: 0.0, count: dimension)
        }

        #if canImport(NaturalLanguage)
        guard let rawVec = embedding.vector(for: trimmed) else {
            return [Double](repeating: 0.0, count: dimension)
        }

        // L2 normalize using Accelerate
        var sumSq = 0.0
        vDSP_svesqD(rawVec, 1, &sumSq, vDSP_Length(dimension))
        let norm = sqrt(sumSq)

        if norm > 1e-12 {
            var scale = 1.0 / norm
            var unitVec = [Double](repeating: 0.0, count: dimension)
            vDSP_vsmulD(rawVec, 1, &scale, &unitVec, 1, vDSP_Length(dimension))
            return unitVec
        }

        return rawVec
        #else
        throw NLPError.unavailableOnPlatform(feature: "AppleSentenceEmbeddingEngine")
        #endif
    }

    /// Computes dense $L_2$-normalized embedding vectors for a batch of text strings.
    ///
    /// If the batch contains more than 16 items, processing is automatically parallelized
    /// across available CPU cores via structured Swift concurrency.
    ///
    /// - Parameter texts: Array of input strings.
    /// - Throws: `NLPError` if embedding computation fails.
    /// - Returns: Array of dense embedding vectors in the same order as `texts`.
    public func embedBatch(_ texts: [String]) async throws -> [[Double]] {
        guard !texts.isEmpty else { return [] }

        if texts.count <= 16 {
            return try texts.map { try embed($0) }
        }

        return try await withThrowingTaskGroup(of: (Int, [Double]).self) { group in
            for (index, text) in texts.enumerated() {
                group.addTask {
                    let vec = try self.embed(text)
                    return (index, vec)
                }
            }

            var results = [[Double]](repeating: [], count: texts.count)
            for try await (index, vec) in group {
                results[index] = vec
            }
            return results
        }
    }
}
