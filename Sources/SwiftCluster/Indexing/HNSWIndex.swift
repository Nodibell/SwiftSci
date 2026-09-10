import Foundation
import Accelerate

/// A hierarchical navigable small world graph index for sub-millisecond approximate nearest neighbor search.
///
/// ## Scalability
/// Scales logarithmically \(O(\log N)\) over high-dimensional vector embeddings, replacing brute-force \(O(N)\) cosine search.
///
/// ## Architecture
/// HNSW constructs a multi-layer graph where upper layers contain sparse, long-distance links for fast geometric
/// skip-list routing, while lower layers contain dense, short-distance links for high-precision local refinement.
///
/// ## Thread Safety
/// Implemented as a Swift actor ensuring data-race free concurrent insertions and searches under Swift 6 strict concurrency.
public actor HNSWIndex {

    // MARK: - Internal Node Representation

    private final class Node {
        let id: String
        let vector: [Double]
        let metadata: [String: String]
        let level: Int
        var neighbors: [[Int]]

        init(id: String, vector: [Double], metadata: [String: String], level: Int) {
            self.id = id
            self.vector = vector
            self.metadata = metadata
            self.level = level
            self.neighbors = Array(repeating: [Int](), count: level + 1)
        }
    }

    // MARK: - Properties

    /// Metric used for evaluating vector similarity and distance.
    public let metric: VectorMetric

    /// Maximum number of bidirectional outgoing connections per node for layers > 0.
    public let M: Int

    /// Maximum number of bidirectional outgoing connections per node at the base layer (layer 0).
    public let M0: Int

    /// Size of the dynamic candidate list evaluated during index construction.
    public let efConstruction: Int

    /// Default size of the dynamic candidate list evaluated during nearest-neighbor search.
    public let efSearch: Int

    /// Normalization factor for level generation: \(m_L = \frac{1}{\ln(M)}\).
    private let mL: Double

    /// All indexed nodes.
    private var nodes: [Node] = []

    /// Mapping from external string ID to internal node index.
    private var idToIndex: [String: Int] = [:]

    /// Current entry point node index at the top layer.
    private var enterPoint: Int?

    /// Current maximum level present in the hierarchical graph.
    private var maxLevel: Int = -1

    /// Internal PRNG state for deterministic level assignments.
    private var rngState: UInt64

    // MARK: - Initialization

    /// Creates a new Hierarchical Navigable Small World index.
    ///
    /// - Parameters:
    ///   - metric: The distance/similarity metric to use. Default is `.euclideanDistance`.
    ///   - M: Max number of outgoing connections per node (layers > 0). Default is `16`.
    ///   - efConstruction: Candidate evaluation pool size during graph construction. Default is `200`.
    ///   - efSearch: Candidate evaluation pool size during query search. Default is `50`.
    ///   - seed: Random seed for reproducible graph level construction. Default is `42`.
    ///
    /// ## Thread Safety
    /// Actor-isolated initialization ensuring strict thread safety.
    ///
    /// ## Complexity
    /// \(O(1)\) initialization cost.
    public init(
        metric: VectorMetric = .euclideanDistance,
        M: Int = 16,
        efConstruction: Int = 200,
        efSearch: Int = 50,
        seed: Int = 42
    ) {
        precondition(M > 1, "M must be greater than 1")
        precondition(efConstruction >= M, "efConstruction must be at least M")
        precondition(efSearch >= 1, "efSearch must be at least 1")

        self.metric = metric
        self.M = M
        self.M0 = 2 * M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.mL = 1.0 / log(Double(M))
        self.rngState = UInt64(bitPattern: Int64(seed)) ^ 0x5DEECE66D
    }

    // MARK: - Public API

    /// The total number of vectors stored in the HNSW index.
    ///
    /// ## Thread Safety
    /// Thread-safe via actor isolation.
    ///
    /// ## Complexity
    /// \(O(1)\).
    public var count: Int {
        return nodes.count
    }

    /// Checks if a vector with the specified identifier is present in the index.
    ///
    /// - Parameter id: The unique vector identifier.
    /// - Returns: `true` if the vector exists, `false` otherwise.
    ///
    /// ## Thread Safety
    /// Thread-safe via actor isolation.
    ///
    /// ## Complexity
    /// \(O(1)\) average hash lookup.
    public func contains(id: String) -> Bool {
        return idToIndex[id] != nil
    }

    /// Inserts a new vector embedding into the hierarchical graph index.
    ///
    /// - Parameters:
    ///   - id: Unique identifier for the vector.
    ///   - vector: Dense numerical embedding array.
    ///   - metadata: Optional key-value metadata associated with the vector.
    ///
    /// ## Thread Safety
    /// Thread-safe via actor isolation. Concurrent callers are serialized safely.
    ///
    /// ## Complexity
    /// \(O(M \cdot \log N)\) average time per insertion.
    public func add(id: String, vector: [Double], metadata: [String: String] = [:]) {
        guard !vector.isEmpty else { return }

        if let existingIdx = idToIndex[id] {
            // Replace existing node vector and metadata
            let existingNode = nodes[existingIdx]
            let updatedNode = Node(id: id, vector: vector, metadata: metadata, level: existingNode.level)
            updatedNode.neighbors = existingNode.neighbors
            nodes[existingIdx] = updatedNode
            return
        }

        let nodeLevel = randomLevel()
        let newNodeIdx = nodes.count
        let newNode = Node(id: id, vector: vector, metadata: metadata, level: nodeLevel)
        nodes.append(newNode)
        idToIndex[id] = newNodeIdx

        guard let ep = enterPoint else {
            enterPoint = newNodeIdx
            maxLevel = nodeLevel
            return
        }

        var currObj = ep
        let topLevel = maxLevel

        // Phase 1: Greedily navigate down from topLevel to nodeLevel + 1
        if topLevel > nodeLevel {
            for lc in stride(from: topLevel, through: nodeLevel + 1, by: -1) {
                var changed = true
                while changed {
                    changed = false
                    let currDist = distance(vector, nodes[currObj].vector)
                    for neighborIdx in nodes[currObj].neighbors[lc] {
                        let d = distance(vector, nodes[neighborIdx].vector)
                        if d < currDist {
                            currObj = neighborIdx
                            changed = true
                            break
                        }
                    }
                }
            }
        }

        // Phase 2: Insert into levels from min(topLevel, nodeLevel) down to 0
        let insertionStartLevel = min(topLevel, nodeLevel)
        var enterPoints = [currObj]

        for lc in stride(from: insertionStartLevel, through: 0, by: -1) {
            let candidates = searchLayer(query: vector, enterPoints: enterPoints, numCandidates: efConstruction, level: lc)
            let maxM = (lc == 0) ? M0 : M
            let selectedNeighbors = selectNeighbors(candidates: candidates, maxNeighbors: maxM)

            newNode.neighbors[lc] = selectedNeighbors

            // Add bidirectional connections
            for neighborIdx in selectedNeighbors {
                let neighbor = nodes[neighborIdx]
                neighbor.neighbors[lc].append(newNodeIdx)
                if neighbor.neighbors[lc].count > maxM {
                    shrinkNeighbors(nodeIdx: neighborIdx, level: lc, maxNeighbors: maxM)
                }
            }

            enterPoints = candidates.map { $0.nodeIdx }
            if let first = enterPoints.first {
                currObj = first
            }
        }

        if nodeLevel > maxLevel {
            maxLevel = nodeLevel
            enterPoint = newNodeIdx
        }
    }

    /// Queries the hierarchical index for the Top-K approximate nearest neighbors.
    ///
    /// - Parameters:
    ///   - query: Dense numerical query embedding vector.
    ///   - topK: Number of nearest neighbors to retrieve. Default is `10`.
    ///   - efSearch: Dynamic candidate queue depth during traversal. Overrides the default when specified.
    /// - Returns: An ordered array of `VectorSearchResult` instances, sorted closest to farthest.
    ///
    /// ## Thread Safety
    /// Thread-safe via actor isolation.
    ///
    /// ## Complexity
    /// \(O(\log N)\) average time over high-dimensional datasets.
    public func search(query: [Double], topK: Int = 10, efSearch: Int? = nil) -> [VectorSearchResult] {
        guard !nodes.isEmpty, let ep = enterPoint else { return [] }
        precondition(topK > 0, "topK must be positive")

        let effectiveEf = max(topK, efSearch ?? self.efSearch)
        var currObj = ep

        // 1. Zoom in greedily from maxLevel down to level 1
        if maxLevel > 0 {
            for lc in stride(from: maxLevel, through: 1, by: -1) {
                var changed = true
                while changed {
                    changed = false
                    let currDist = distance(query, nodes[currObj].vector)
                    for neighborIdx in nodes[currObj].neighbors[lc] {
                        let d = distance(query, nodes[neighborIdx].vector)
                        if d < currDist {
                            currObj = neighborIdx
                            changed = true
                            break
                        }
                    }
                }
            }
        }

        // 2. Search bottom layer 0 with full dynamic priority queue
        let candidates = searchLayer(query: query, enterPoints: [currObj], numCandidates: effectiveEf, level: 0)
        let sorted = candidates.sorted { $0.distance < $1.distance }
        let k = min(topK, sorted.count)

        return (0..<k).map { i in
            let item = sorted[i]
            let node = nodes[item.nodeIdx]
            let evaluatedScore = score(fromDistance: item.distance)
            return VectorSearchResult(id: node.id, score: evaluatedScore, metadata: node.metadata)
        }
    }

    // MARK: - Graph Traversal & Construction Details

    private struct DistEntry {
        let nodeIdx: Int
        let distance: Double
    }

    private func searchLayer(query: [Double], enterPoints: [Int], numCandidates: Int, level: Int) -> [DistEntry] {
        var visited = Set<Int>()
        var candidates: [DistEntry] = []
        var nearest: [DistEntry] = []

        for ep in enterPoints {
            visited.insert(ep)
            let dist = distance(query, nodes[ep].vector)
            let entry = DistEntry(nodeIdx: ep, distance: dist)
            candidates.append(entry)
            nearest.append(entry)
        }

        while !candidates.isEmpty {
            // Find closest candidate
            candidates.sort { $0.distance < $1.distance }
            let closestCandidate = candidates.removeFirst()

            // Furthest node in current results pool
            nearest.sort { $0.distance < $1.distance }
            guard let furthestNearest = nearest.last else { break }

            if closestCandidate.distance > furthestNearest.distance && nearest.count >= numCandidates {
                break
            }

            let candidateNode = nodes[closestCandidate.nodeIdx]
            let currentNeighbors = candidateNode.neighbors[level]

            for neighborIdx in currentNeighbors {
                if !visited.contains(neighborIdx) {
                    visited.insert(neighborIdx)

                    let neighborDist = distance(query, nodes[neighborIdx].vector)
                    furthestNearestCheck: if nearest.count < numCandidates || neighborDist < nearest.last!.distance {
                        let newEntry = DistEntry(nodeIdx: neighborIdx, distance: neighborDist)
                        candidates.append(newEntry)
                        nearest.append(newEntry)

                        if nearest.count > numCandidates {
                            nearest.sort { $0.distance < $1.distance }
                            nearest.removeLast()
                        }
                    }
                }
            }
        }

        return nearest
    }

    private func selectNeighbors(candidates: [DistEntry], maxNeighbors: Int) -> [Int] {
        let sorted = candidates.sorted { $0.distance < $1.distance }
        return Array(sorted.prefix(maxNeighbors).map { $0.nodeIdx })
    }

    private func shrinkNeighbors(nodeIdx: Int, level: Int, maxNeighbors: Int) {
        let node = nodes[nodeIdx]
        let currentNeighbors = node.neighbors[level]
        let nodeVec = node.vector

        let neighborEntries = currentNeighbors.map { idx in
            DistEntry(nodeIdx: idx, distance: distance(nodeVec, nodes[idx].vector))
        }.sorted { $0.distance < $1.distance }

        node.neighbors[level] = Array(neighborEntries.prefix(maxNeighbors).map { $0.nodeIdx })
    }

    // MARK: - Distance Calculations via Accelerate vDSP

    private func distance(_ a: [Double], _ b: [Double]) -> Double {
        switch metric {
        case .euclideanDistance:
            return euclideanDistance(a, b)
        case .cosineSimilarity:
            return cosineDistance(a, b)
        case .dotProduct:
            return dotProductDistance(a, b)
        }
    }

    private func euclideanDistance(_ a: [Double], _ b: [Double]) -> Double {
        let count = min(a.count, b.count)
        guard count > 0 else { return 0.0 }
        let n = vDSP_Length(count)
        var diff = [Double](repeating: 0.0, count: count)
        vDSP_vsubD(b, 1, a, 1, &diff, 1, n)
        var sumSq = 0.0
        vDSP_svesqD(diff, 1, &sumSq, n)
        return sqrt(sumSq)
    }

    private func cosineDistance(_ a: [Double], _ b: [Double]) -> Double {
        let count = min(a.count, b.count)
        guard count > 0 else { return 1.0 }
        let n = vDSP_Length(count)
        var dot = 0.0
        vDSP_dotprD(a, 1, b, 1, &dot, n)
        var normA = 0.0
        var normB = 0.0
        vDSP_svesqD(a, 1, &normA, n)
        vDSP_svesqD(b, 1, &normB, n)
        let denom = sqrt(normA * normB)
        if denom < 1e-12 { return 1.0 }
        let sim = max(-1.0, min(1.0, dot / denom))
        return 1.0 - sim
    }

    private func dotProductDistance(_ a: [Double], _ b: [Double]) -> Double {
        let count = min(a.count, b.count)
        guard count > 0 else { return 0.0 }
        let n = vDSP_Length(count)
        var dot = 0.0
        vDSP_dotprD(a, 1, b, 1, &dot, n)
        return -dot
    }

    private func score(fromDistance dist: Double) -> Double {
        switch metric {
        case .euclideanDistance:
            return dist
        case .cosineSimilarity:
            return 1.0 - dist
        case .dotProduct:
            return -dist
        }
    }

    // MARK: - Level Generation

    private func nextUniform() -> Double {
        // Linear congruential step for reproducible level distribution
        rngState = rngState &* 6364136223846793005 &+ 1442695040888963407
        let x = (rngState >> 11)
        return Double(x) / Double(1 << 53)
    }

    private func randomLevel() -> Int {
        var u = nextUniform()
        while u == 0.0 {
            u = nextUniform()
        }
        let r = -log(u) * mL
        return max(0, Int(r))
    }
}
