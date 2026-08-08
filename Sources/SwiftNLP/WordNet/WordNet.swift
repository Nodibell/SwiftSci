import Foundation

/// Part of Speech categories in WordNet.
public enum POS: String, Sendable, Codable {
    case noun = "n"
    case verb = "v"
    case adjective = "a"
    case adverb = "r"
}

/// Represents a WordNet Synset (Syntactic/Semantic Set of Synonym Concepts).
public struct Synset: Sendable, Hashable, Identifiable {
    /// The unique identifier of the synset (e.g., "dog.n.01").
    public let id: String
    
    /// The canonical label or short concept name of the synset.
    public let name: String
    
    /// The syntactic Part of Speech classification of the synset.
    public let pos: POS
    
    /// The natural language gloss definition explaining the semantic concept.
    public let definition: String
    
    /// List of synonym word forms (lemmas) belonging to this synset concept.
    public let lemmas: [String]
    
    /// List of parent synset identifiers in the hypernym taxonomy graph.
    public let hypernymIDs: [String]
    
    /// List of child synset identifiers in the hyponym taxonomy graph.
    public let hyponymIDs: [String]
    
    /// Creates a new WordNet `Synset` concept instance.
    ///
    /// - Parameters:
    ///   - id: Unique synset identifier.
    ///   - name: Canonical concept name.
    ///   - pos: Part of speech category.
    ///   - definition: Gloss text definition.
    ///   - lemmas: Array of synonym words.
    ///   - hypernymIDs: Array of parent synset IDs.
    ///   - hyponymIDs: Array of child synset IDs.
    public init(
        id: String,
        name: String,
        pos: POS,
        definition: String,
        lemmas: [String],
        hypernymIDs: [String] = [],
        hyponymIDs: [String] = []
    ) {
        self.id = id
        self.name = name
        self.pos = pos
        self.definition = definition
        self.lemmas = lemmas
        self.hypernymIDs = hypernymIDs
        self.hyponymIDs = hyponymIDs
    }
}

/// Native WordNet semantic database query engine and concept similarity computer.
public struct WordNet: Sendable {
    private let synsetMap: [String: Synset]
    private let lemmaIndex: [String: [String]] // lemma -> [synsetID]
    
    /// Creates a WordNet database initialized with pre-built or custom synsets.
    public init(synsets: [Synset] = WordNet.defaultSynsets) {
        var map: [String: Synset] = [:]
        var idx: [String: [String]] = [:]
        
        for syn in synsets {
            map[syn.id] = syn
            for lemma in syn.lemmas {
                let key = lemma.lowercased()
                idx[key, default: []].append(syn.id)
            }
        }
        
        self.synsetMap = map
        self.lemmaIndex = idx
    }
    
    /// Returns all synsets matching the given lemma (word).
    public func synsets(for lemma: String, pos: POS? = nil) -> [Synset] {
        let key = lemma.lowercased()
        guard let ids = lemmaIndex[key] else { return [] }
        let matches = ids.compactMap { synsetMap[$0] }
        if let pos {
            return matches.filter { $0.pos == pos }
        }
        return matches
    }
    
    /// Returns direct hypernyms (parent concepts) for a synset.
    public func hypernyms(of synset: Synset) -> [Synset] {
        synset.hypernymIDs.compactMap { synsetMap[$0] }
    }
    
    /// Returns direct hyponyms (child concepts) for a synset.
    public func hyponyms(of synset: Synset) -> [Synset] {
        synset.hyponymIDs.compactMap { synsetMap[$0] }
    }
    
    /// Computes shortest path distance between two synsets in the hypernym hierarchy.
    public func pathDistance(_ s1: Synset, _ s2: Synset) -> Int? {
        if s1 == s2 { return 0 }
        
        var visited1: [String: Int] = [s1.id: 0]
        var queue1 = [s1.id]
        
        while !queue1.isEmpty {
            let curr = queue1.removeFirst()
            let dist = visited1[curr]!
            if let syn = synsetMap[curr] {
                for parentID in syn.hypernymIDs {
                    if visited1[parentID] == nil {
                        visited1[parentID] = dist + 1
                        queue1.append(parentID)
                    }
                }
            }
        }
        
        var visited2: [String: Int] = [s2.id: 0]
        var queue2 = [s2.id]
        var minTotalDist: Int? = nil
        
        while !queue2.isEmpty {
            let curr = queue2.removeFirst()
            let dist2 = visited2[curr]!
            if let dist1 = visited1[curr] {
                let total = dist1 + dist2
                minTotalDist = min(minTotalDist ?? Int.max, total)
            }
            if let syn = synsetMap[curr] {
                for parentID in syn.hypernymIDs {
                    if visited2[parentID] == nil {
                        visited2[parentID] = dist2 + 1
                        queue2.append(parentID)
                    }
                }
            }
        }
        
        return minTotalDist
    }
    
    /// Computes Path Similarity (1 / (path_distance + 1)) between two synsets in [0, 1].
    public func pathSimilarity(_ s1: Synset, _ s2: Synset) -> Double {
        guard let dist = pathDistance(s1, s2) else { return 0.0 }
        return 1.0 / Double(dist + 1)
    }
    
    /// Computes Wu-Palmer Similarity (2 * depth(LCS) / (depth(s1) + depth(s2))) between two synsets.
    public func wupSimilarity(_ s1: Synset, _ s2: Synset) -> Double {
        let ancestors1 = ancestorDepths(s1)
        let ancestors2 = ancestorDepths(s2)
        
        var maxLCSDepth = 0
        for (id, d1) in ancestors1 {
            if let d2 = ancestors2[id] {
                let lcsDepth = min(d1, d2)
                maxLCSDepth = max(maxLCSDepth, lcsDepth)
            }
        }
        
        let depth1 = (ancestors1[s1.id] ?? 1)
        let depth2 = (ancestors2[s2.id] ?? 1)
        
        let denom = Double(depth1 + depth2)
        guard denom > 0 else { return 0.0 }
        return (2.0 * Double(maxLCSDepth)) / denom
    }
    
    private func ancestorDepths(_ synset: Synset) -> [String: Int] {
        var result: [String: Int] = [synset.id: 1]
        var queue: [(id: String, depth: Int)] = [(synset.id, 1)]
        
        while !queue.isEmpty {
            let (currID, d) = queue.removeFirst()
            if let syn = synsetMap[currID] {
                for parentID in syn.hypernymIDs {
                    if result[parentID] == nil {
                        result[parentID] = d + 1
                        queue.append((parentID, d + 1))
                    }
                }
            }
        }
        return result
    }
    
    /// Default core synsets (curated English WordNet subset for fast offline execution).
    public static let defaultSynsets: [Synset] = [
        Synset(id: "entity.n.01", name: "entity", pos: .noun, definition: "That which is perceived or known as having its own distinct existence.", lemmas: ["entity"]),
        Synset(id: "organism.n.01", name: "organism", pos: .noun, definition: "A living thing that has an organized structure.", lemmas: ["organism", "living_thing"], hypernymIDs: ["entity.n.01"]),
        Synset(id: "animal.n.01", name: "animal", pos: .noun, definition: "A living organism that feeds on organic matter.", lemmas: ["animal", "fauna"], hypernymIDs: ["organism.n.01"], hyponymIDs: ["dog.n.01", "cat.n.01"]),
        Synset(id: "dog.n.01", name: "dog", pos: .noun, definition: "A domesticated carnivorous mammal (Canis familiaris).", lemmas: ["dog", "canine"], hypernymIDs: ["animal.n.01"]),
        Synset(id: "cat.n.01", name: "cat", pos: .noun, definition: "A small domesticated carnivorous mammal (Felis catus).", lemmas: ["cat", "feline"], hypernymIDs: ["animal.n.01"]),
        Synset(id: "computer.n.01", name: "computer", pos: .noun, definition: "An electronic device for storing and processing data.", lemmas: ["computer", "machine"], hypernymIDs: ["entity.n.01"])
    ]
}
