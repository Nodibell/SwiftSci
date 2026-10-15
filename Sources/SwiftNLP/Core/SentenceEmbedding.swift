import Foundation

/// A protocol representing a local or model-based sentence embedding generator.
///
/// Implementations project natural language sentences and documents into dense
/// numerical vector representations suitable for semantic search, clustering,
/// and zero-shot classification.
public protocol SentenceEmbeddingProvider: Sendable {
    /// Dimensionality of the generated dense embedding vectors.
    var dimension: Int { get }

    /// Computes a dense embedding vector for the provided text.
    ///
    /// - Parameter text: Input string to embed.
    /// - Throws: `NLPError` if embedding computation fails or the model is unavailable.
    /// - Returns: Dense numerical vector of length ``dimension``.
    func embed(_ text: String) throws -> [Double]

    /// Computes dense embedding vectors for a batch of text strings.
    ///
    /// - Parameter texts: Array of input strings.
    /// - Throws: `NLPError` if embedding computation fails or the model is unavailable.
    /// - Returns: Array of dense embedding vectors corresponding to the input order.
    func embedBatch(_ texts: [String]) async throws -> [[Double]]
}

extension SentenceEmbeddingProvider {
    /// Default sequential implementation for batch embedding.
    public func embedBatch(_ texts: [String]) async throws -> [[Double]] {
        try texts.map { try embed($0) }
    }
}
