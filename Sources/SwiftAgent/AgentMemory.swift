import Foundation
import SwiftCluster

/// An immutable record stored within an autonomous agent's episodic or semantic memory.
public struct AgentMemoryRecord: Sendable, Identifiable, Equatable {
    /// Unique memory record identifier.
    public let id: String
    /// The message, observation, or factual content.
    public let content: String
    /// The originating entity or role (e.g. `"user"`, `"agent"`, `"tool"`, `"system"`).
    public let role: String
    /// Timestamp when the memory was registered.
    public let timestamp: Date
    /// Contextual key-value metadata attached to this memory.
    public let metadata: [String: String]

    /// Initializes a memory record.
    ///
    /// - Parameters:
    ///   - id: Unique record ID. Default is a new UUID string.
    ///   - content: Text payload.
    ///   - role: Author or generator role.
    ///   - timestamp: Creation timestamp.
    ///   - metadata: Key-value attributes.
    public init(
        id: String = UUID().uuidString,
        content: String,
        role: String,
        timestamp: Date = Date(),
        metadata: [String: String] = [:]
    ) {
        self.id = id
        self.content = content
        self.role = role
        self.timestamp = timestamp
        self.metadata = metadata
    }
}

/// Abstract protocol governing conversational, episodic, and semantic agent memory stores.
public protocol AgentMemory: Sendable {
    /// Appends a new interaction, observation, or fact to the memory store.
    ///
    /// - Parameters:
    ///   - content: Textual content.
    ///   - role: Source role (e.g. "user", "agent", "tool").
    ///   - metadata: Additional key-value attributes.
    func record(content: String, role: String, metadata: [String: String]) async throws

    /// Retrieves relevant memory context matching the query or recent conversation history.
    ///
    /// - Parameters:
    ///   - query: Target query or goal string.
    ///   - limit: Maximum number of records to return.
    /// - Returns: An ordered array of retrieved memory records.
    func retrieveContext(query: String, limit: Int) async throws -> [AgentMemoryRecord]

    /// Purges all memories from the store.
    func clear() async throws
}

/// A short-term memory store retaining a sliding window of the most recent agent interactions.
public actor SlidingWindowMemory: AgentMemory {
    /// Maximum number of records preserved before oldest are evicted.
    public let maxRecords: Int
    private var records: [AgentMemoryRecord] = []

    /// Initializes a sliding window memory buffer.
    ///
    /// - Parameter maxRecords: Capacity limit (default: 20).
    public init(maxRecords: Int = 20) {
        precondition(maxRecords > 0, "maxRecords must be positive")
        self.maxRecords = maxRecords
    }

    /// Records a new memory item, evicting the oldest record if capacity is exceeded.
    public func record(content: String, role: String, metadata: [String: String] = [:]) async throws {
        let rec = AgentMemoryRecord(content: content, role: role, metadata: metadata)
        records.append(rec)
        if records.count > maxRecords {
            records.removeFirst(records.count - maxRecords)
        }
    }

    /// Retrieves the most recent memory records up to `limit`.
    public func retrieveContext(query: String, limit: Int = 10) async throws -> [AgentMemoryRecord] {
        let count = min(limit, records.count)
        guard count > 0 else { return [] }
        return Array(records.suffix(count))
    }

    /// Clears all stored records.
    public func clear() async throws {
        records.removeAll()
    }
}

/// A long-term semantic memory store backed by a Hierarchical Navigable Small World (`HNSWIndex`) graph.
public actor SemanticVectorMemory: AgentMemory {
    private let index: HNSWIndex
    private var records: [String: AgentMemoryRecord] = [:]
    private let embedder: @Sendable (String) async throws -> [Double]

    /// Initializes a semantic vector memory store.
    ///
    /// - Parameters:
    ///   - metric: Vector similarity metric (default: `.cosineSimilarity`).
    ///   - embedder: Asynchronous closure converting text into dense numerical embedding vectors.
    public init(
        metric: VectorMetric = .cosineSimilarity,
        embedder: @escaping @Sendable (String) async throws -> [Double]
    ) {
        self.index = HNSWIndex(metric: metric, M: 16, efConstruction: 200, efSearch: 50)
        self.embedder = embedder
    }

    /// Embeds the content and indexes it into the HNSW graph for sub-millisecond similarity recall.
    public func record(content: String, role: String, metadata: [String: String] = [:]) async throws {
        let id = UUID().uuidString
        let rec = AgentMemoryRecord(id: id, content: content, role: role, metadata: metadata)
        records[id] = rec

        let vector = try await embedder(content)
        await index.add(id: id, vector: vector, metadata: metadata)
    }

    /// Retrieves the top-K semantically most relevant historical memories using HNSW vector search.
    public func retrieveContext(query: String, limit: Int = 5) async throws -> [AgentMemoryRecord] {
        guard !records.isEmpty else { return [] }
        let queryVector = try await embedder(query)
        let results = await index.search(query: queryVector, topK: limit)

        var matched: [AgentMemoryRecord] = []
        for res in results {
            if let rec = records[res.id] {
                matched.append(rec)
            }
        }
        return matched
    }

    /// Clears all indexed vectors and records.
    public func clear() async throws {
        records.removeAll()
    }
}
