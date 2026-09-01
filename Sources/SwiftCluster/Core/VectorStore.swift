import Foundation
import Accelerate

// MARK: - VectorMetric

/// Distance or similarity metrics supported by ``VectorStore``.
public enum VectorMetric: String, Sendable, Codable {
    /// Cosine similarity: \(\frac{u \cdot v}{\|u\| \|v\|}\), ranging in \([-1, 1]\). Higher is closer.
    case cosineSimilarity
    /// Inner dot product: \(u \cdot v\). Higher is closer.
    case dotProduct
    /// Euclidean distance (L2): \(\|u - v\|_2\). Lower is closer.
    case euclideanDistance
}

// MARK: - VectorEntry

/// Represents an item stored inside a ``VectorStore``.
public struct VectorEntry: Sendable, Codable, Equatable {
    /// Unique identifier for this vector entry.
    public let id: String
    /// High-dimensional dense numerical vector.
    public let vector: [Double]
    /// Optional key-value metadata associated with this vector.
    public let metadata: [String: String]

    /// Creates a new vector entry.
    ///
    /// - Parameters:
    ///   - id: Unique identifier.
    ///   - vector: Dense numerical embedding array.
    ///   - metadata: Optional key-value string metadata.
    public init(id: String, vector: [Double], metadata: [String: String] = [:]) {
        self.id = id
        self.vector = vector
        self.metadata = metadata
    }
}

// MARK: - VectorSearchResult

/// Represents a single nearest-neighbor match result from a vector search query.
public struct VectorSearchResult: Sendable, Codable, Equatable {
    /// Unique identifier of the matched vector.
    public let id: String
    /// Similarity or distance score (higher is better for cosine/dot, lower is better for euclidean).
    public let score: Double
    /// Metadata associated with the matched vector.
    public let metadata: [String: String]

    /// Creates a new search result item.
    ///
    /// - Parameters:
    ///   - id: Matched entry identifier.
    ///   - score: Evaluated similarity or distance metric.
    ///   - metadata: Associated key-value dictionary.
    public init(id: String, score: Double, metadata: [String: String] = [:]) {
        self.id = id
        self.score = score
        self.metadata = metadata
    }
}

// MARK: - VectorStore

/// An in-memory vector database and similarity search index optimized for Apple Silicon.
///
/// ``VectorStore`` stores high-dimensional dense embeddings and provides high-speed Top-K
/// nearest-neighbor retrieval using Apple Accelerate SIMD operations.
///
/// ## Supported Metrics
/// - ``VectorMetric/cosineSimilarity``: Cosine angle between normalized vectors (default).
/// - ``VectorMetric/dotProduct``: Raw inner product.
/// - ``VectorMetric/euclideanDistance``: L2 Euclidean distance.
///
/// ## Example
/// ```swift
/// import SwiftCluster
///
/// let store = VectorStore(metric: .cosineSimilarity)
/// store.add(id: "doc_1", vector: [1.0, 0.0, 0.0], metadata: ["title": "Science"])
/// store.add(id: "doc_2", vector: [0.0, 1.0, 0.0], metadata: ["title": "Art"])
///
/// let results = store.search(query: [0.9, 0.1, 0.0], topK: 1)
/// print(results.first?.id) // "doc_1"
/// ```
public final class VectorStore: @unchecked Sendable {

    /// Distance/similarity metric used for nearest neighbor evaluation.
    public let metric: VectorMetric

    private var entries: [VectorEntry] = []
    private var idToIndex: [String: Int] = [:]
    private let lock = NSLock()

    /// Number of vectors stored in the index.
    public var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return entries.count
    }

    /// Whether the vector store is currently empty.
    public var isEmpty: Bool {
        lock.lock()
        defer { lock.unlock() }
        return entries.isEmpty
    }

    /// Creates a new in-memory vector store.
    ///
    /// - Parameter metric: Similarity or distance metric (default ``VectorMetric/cosineSimilarity``).
    public init(metric: VectorMetric = .cosineSimilarity) {
        self.metric = metric
    }

    /// Inserts or replaces a vector in the index.
    ///
    /// - Parameters:
    ///   - id: Unique identifier.
    ///   - vector: Dense numerical embedding array.
    ///   - metadata: Optional key-value metadata dictionary.
    public func add(id: String, vector: [Double], metadata: [String: String] = [:]) {
        let entry = VectorEntry(id: id, vector: vector, metadata: metadata)
        lock.lock()
        defer { lock.unlock() }

        if let existingIdx = idToIndex[id] {
            entries[existingIdx] = entry
        } else {
            idToIndex[id] = entries.count
            entries.append(entry)
        }
    }

    /// Inserts a batch of vector entries into the index.
    ///
    /// - Parameter batch: Array of ``VectorEntry`` items.
    public func addBatch(entries batch: [VectorEntry]) {
        lock.lock()
        defer { lock.unlock() }

        for entry in batch {
            if let existingIdx = idToIndex[entry.id] {
                entries[existingIdx] = entry
            } else {
                idToIndex[entry.id] = entries.count
                entries.append(entry)
            }
        }
    }

    /// Removes a vector by its unique identifier.
    ///
    /// - Parameter id: Unique identifier of the entry to remove.
    /// - Returns: `true` if the vector was found and removed, `false` otherwise.
    @discardableResult
    public func remove(id: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        guard let idx = idToIndex[id] else { return false }
        entries.remove(at: idx)
        idToIndex.removeAll(keepingCapacity: true)
        for (i, entry) in entries.enumerated() {
            idToIndex[entry.id] = i
        }
        return true
    }

    /// Clears all entries from the vector store.
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        entries.removeAll()
        idToIndex.removeAll()
    }

    /// Checks if a vector identifier exists in the store.
    ///
    /// - Parameter id: Identifier to look up.
    /// - Returns: `true` if found, `false` otherwise.
    public func contains(id: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return idToIndex[id] != nil
    }

    /// Retrieves an entry by its identifier.
    ///
    /// - Parameter id: Identifier to look up.
    /// - Returns: The ``VectorEntry`` if found, `nil` otherwise.
    public func get(id: String) -> VectorEntry? {
        lock.lock()
        defer { lock.unlock() }
        guard let idx = idToIndex[id] else { return nil }
        return entries[idx]
    }

    /// Searches the index for the Top-K nearest neighbors to the query vector.
    ///
    /// - Parameters:
    ///   - query: Query embedding vector.
    ///   - topK: Maximum number of nearest neighbors to return (default `5`).
    /// - Returns: Ordered list of ``VectorSearchResult`` sorted by closest match.
    public func search(query: [Double], topK: Int = 5) -> [VectorSearchResult] {
        guard topK > 0 else { return [] }
        
        lock.lock()
        defer { lock.unlock() }
        
        guard !entries.isEmpty else { return [] }

        let queryNorm: Double
        if metric == .cosineSimilarity {
            var sumSq: Double = 0
            vDSP_svesqD(query, 1, &sumSq, vDSP_Length(query.count))
            queryNorm = sqrt(sumSq)
        } else {
            queryNorm = 1.0
        }

        // Bounded topK structure avoiding full array allocation and sorting
        var topResults = [(id: String, score: Double, metadata: [String: String])]()
        topResults.reserveCapacity(topK + 1)

        let isAscending = metric == .euclideanDistance

        for entry in entries {
            let dim = min(query.count, entry.vector.count)
            guard dim > 0 else { continue }

            let score: Double
            switch metric {
            case .cosineSimilarity:
                var dot: Double = 0
                vDSP_dotprD(query, 1, entry.vector, 1, &dot, vDSP_Length(dim))
                var entrySumSq: Double = 0
                vDSP_svesqD(entry.vector, 1, &entrySumSq, vDSP_Length(dim))
                let entryNorm = sqrt(entrySumSq)
                let denom = queryNorm * entryNorm
                score = denom > 1e-12 ? (dot / denom) : 0.0

            case .dotProduct:
                var dot: Double = 0
                vDSP_dotprD(query, 1, entry.vector, 1, &dot, vDSP_Length(dim))
                score = dot

            case .euclideanDistance:
                var distSq: Double = 0
                vDSP_distancesqD(query, 1, entry.vector, 1, &distSq, vDSP_Length(dim))
                score = sqrt(distSq)
            }

            let item = (id: entry.id, score: score, metadata: entry.metadata)
            
            if topResults.count < topK {
                topResults.append(item)
                if isAscending {
                    topResults.sort { $0.score < $1.score }
                } else {
                    topResults.sort { $0.score > $1.score }
                }
            } else {
                let worstScore = topResults.last!.score
                let isBetter = isAscending ? (score < worstScore) : (score > worstScore)
                if isBetter {
                    topResults[topResults.count - 1] = item
                    if isAscending {
                        topResults.sort { $0.score < $1.score }
                    } else {
                        topResults.sort { $0.score > $1.score }
                    }
                }
            }
        }

        return topResults.map { VectorSearchResult(id: $0.id, score: $0.score, metadata: $0.metadata) }
    }
}
