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
    /// - Parameters:
    ///   - lemma: <#description#>
    ///   - pos: <#description#>
    /// - Returns: <#description#>
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
    /// - Parameters:
    ///   - synset: <#description#>
    /// - Returns: <#description#>
    public func hypernyms(of synset: Synset) -> [Synset] {
        synset.hypernymIDs.compactMap { synsetMap[$0] }
    }
    
    /// Returns direct hyponyms (child concepts) for a synset.
    /// - Parameters:
    ///   - synset: <#description#>
    /// - Returns: <#description#>
    public func hyponyms(of synset: Synset) -> [Synset] {
        synset.hyponymIDs.compactMap { synsetMap[$0] }
    }
    
    /// Computes shortest path distance between two synsets in the hypernym hierarchy.
    /// - Parameters:
    ///   - s1: <#description#>
    ///   - s2: <#description#>
    /// - Returns: <#description#>
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
    /// - Parameters:
    ///   - s1: <#description#>
    ///   - s2: <#description#>
    /// - Returns: <#description#>
    public func pathSimilarity(_ s1: Synset, _ s2: Synset) -> Double {
        guard let dist = pathDistance(s1, s2) else { return 0.0 }
        return 1.0 / Double(dist + 1)
    }
    
    /// Computes Wu-Palmer Similarity (2 * depth(LCS) / (depth(s1) + depth(s2))) between two synsets.
    /// - Parameters:
    ///   - s1: The source synset.
    ///   - s2: The target synset.
    /// - Returns: The Wu-Palmer conceptual similarity score between 0.0 and 1.0.
    public func wupSimilarity(_ s1: Synset, _ s2: Synset) -> Double {
        if s1.id == s2.id { return 1.0 }
        
        let ancestors1 = allAncestors(of: s1)
        let ancestors2 = allAncestors(of: s2)
        let common = ancestors1.intersection(ancestors2)
        guard !common.isEmpty else { return 0.0 }
        
        var maxLCSDepth = 0
        for ancestorID in common {
            if let syn = synsetMap[ancestorID] {
                let d = depth(of: syn)
                if d > maxLCSDepth {
                    maxLCSDepth = d
                }
            }
        }
        
        let d1 = depth(of: s1)
        let d2 = depth(of: s2)
        let denom = Double(d1 + d2)
        guard denom > 0 else { return 0.0 }
        return (2.0 * Double(maxLCSDepth)) / denom
    }
    
    /// Computes the maximum depth of a synset from the taxonomy root(s).
    /// A root synset (having no hypernyms) has depth 1.
    /// - Parameters:
    ///   - synset: The synset whose taxonomic depth is measured.
    /// - Returns: An integer representing depth from the root concept (root = 1).
    public func depth(of synset: Synset) -> Int {
        var visited = Set<String>()
        return synsetDepth(synset.id, visited: &visited)
    }
    
    private func synsetDepth(_ id: String, visited: inout Set<String>) -> Int {
        guard let synset = synsetMap[id] else { return 1 }
        let validHypernyms = synset.hypernymIDs.filter { synsetMap[$0] != nil }
        if validHypernyms.isEmpty {
            return 1
        }
        visited.insert(id)
        var maxParentDepth = 0
        for parentID in validHypernyms {
            if !visited.contains(parentID) {
                let d = synsetDepth(parentID, visited: &visited)
                maxParentDepth = max(maxParentDepth, d)
            }
        }
        visited.remove(id)
        return maxParentDepth + 1
    }
    
    private func allAncestors(of synset: Synset) -> Set<String> {
        var ancestors: Set<String> = [synset.id]
        var queue = [synset.id]
        var visited: Set<String> = [synset.id]
        
        while !queue.isEmpty {
            let currID = queue.removeFirst()
            if let syn = synsetMap[currID] {
                for parentID in syn.hypernymIDs {
                    if !visited.contains(parentID) {
                        visited.insert(parentID)
                        ancestors.insert(parentID)
                        queue.append(parentID)
                    }
                }
            }
        }
        return ancestors
    }
    
    // MARK: - Princeton WordNet Data Loader
    
    /// Loads synsets from a standard Princeton WordNet database file (e.g. `dict/data.noun`, `dict/data.verb`).
    ///
    /// - Parameters:
    ///   - url: File URL to the Princeton WordNet data file.
    ///   - pos: Syntactic category of the file (defaults to `.noun`).
    /// - Returns: An array of parsed `Synset` instances.
    /// - Throws: `CocoaError` if the file cannot be read, or parsing errors.
    public static func load(fromDataFile url: URL, pos: POS = .noun) throws -> [Synset] {
        let content = try String(contentsOf: url, encoding: .utf8)
        return try parsePrincetonData(content, defaultPOS: pos)
    }
    
    /// Loads a complete WordNet database from a directory containing standard Princeton WordNet data files.
    /// Searches for `data.noun`, `data.verb`, `data.adj`, `data.adv` (or within a `dict/` subdirectory).
    ///
    /// - Parameter directoryURL: Directory URL containing WordNet data files.
    /// - Returns: An initialized `WordNet` instance.
    /// - Throws: `CocoaError` if no Princeton data files are found, or parsing errors.
    public static func load(fromDirectory directoryURL: URL) throws -> WordNet {
        let fileManager = FileManager.default
        var allSynsets: [Synset] = []
        
        let possibleDirs = [
            directoryURL,
            directoryURL.appendingPathComponent("dict")
        ]
        
        let targets: [(name: String, pos: POS)] = [
            ("data.noun", .noun),
            ("data.verb", .verb),
            ("data.adj", .adjective),
            ("data.adv", .adverb)
        ]
        
        var foundAny = false
        for dir in possibleDirs {
            for (filename, pos) in targets {
                let fileURL = dir.appendingPathComponent(filename)
                if fileManager.fileExists(atPath: fileURL.path) {
                    let synsets = try load(fromDataFile: fileURL, pos: pos)
                    allSynsets.append(contentsOf: synsets)
                    foundAny = true
                }
            }
            if foundAny { break }
        }
        
        guard foundAny else {
            throw CocoaError(.fileNoSuchFile)
        }
        
        return WordNet(synsets: allSynsets)
    }
    
    /// Parses the raw text contents of a Princeton WordNet database file.
    ///
    /// - Parameters:
    ///   - content: Raw text of a `data.{pos}` file.
    ///   - defaultPOS: Fallback Part of Speech category.
    /// - Returns: Array of parsed `Synset` instances.
    /// - Throws: `Error` if the string content cannot be parsed.
    public static func parsePrincetonData(_ content: String, defaultPOS: POS = .noun) throws -> [Synset] {
        var synsets: [Synset] = []
        synsets.reserveCapacity(2048)
        
        content.enumerateLines { line, _ in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || line.hasPrefix("  ") || line.hasPrefix("#") {
                return // skip header license / comments
            }
            
            guard let synset = parsePrincetonLine(line, defaultPOS: defaultPOS) else {
                return
            }
            synsets.append(synset)
        }
        
        return synsets
    }
    
    private static func parsePrincetonLine(_ line: String, defaultPOS: POS) -> Synset? {
        let parts = line.components(separatedBy: "|")
        let head = parts[0].trimmingCharacters(in: .whitespaces)
        let gloss = parts.count > 1 ? parts[1].trimmingCharacters(in: .whitespaces) : ""
        
        let tokens = head.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        guard tokens.count >= 5 else { return nil }
        
        let offset = tokens[0]
        let ssType = tokens[2]
        let pos: POS
        switch ssType {
        case "n": pos = .noun
        case "v": pos = .verb
        case "a", "s": pos = .adjective
        case "r": pos = .adverb
        default: pos = defaultPOS
        }
        
        guard let wCnt = Int(tokens[3], radix: 16), wCnt > 0 else { return nil }
        var cursor = 4
        var lemmas: [String] = []
        lemmas.reserveCapacity(wCnt)
        
        for _ in 0..<wCnt {
            guard cursor + 1 < tokens.count else { break }
            let word = tokens[cursor].replacingOccurrences(of: "_", with: " ")
            lemmas.append(word.lowercased())
            cursor += 2 // skip word and lex_id
        }
        guard !lemmas.isEmpty else { return nil }
        
        var hypernyms: [String] = []
        var hyponyms: [String] = []
        
        if cursor < tokens.count, let pCnt = Int(tokens[cursor]) {
            cursor += 1
            for _ in 0..<pCnt {
                guard cursor + 3 < tokens.count else { break }
                let ptrSymbol = tokens[cursor]
                let targetOffset = tokens[cursor + 1]
                let targetPOS = tokens[cursor + 2]
                let targetID = "\(targetOffset)-\(targetPOS)"
                
                if ptrSymbol == "@" || ptrSymbol == "@i" {
                    hypernyms.append(targetID)
                } else if ptrSymbol == "~" || ptrSymbol == "~i" {
                    hyponyms.append(targetID)
                }
                cursor += 4
            }
        }
        
        let id = "\(offset)-\(ssType)"
        let canonicalName = lemmas[0]
        
        return Synset(
            id: id,
            name: canonicalName,
            pos: pos,
            definition: gloss,
            lemmas: lemmas,
            hypernymIDs: hypernyms,
            hyponymIDs: hyponyms
        )
    }

    // MARK: - Default Core Synsets
    
    /// Default core synsets (curated English WordNet taxonomy for offline execution).
    public static let defaultSynsets: [Synset] = [
        // --- Root & Top-Level Ontological Concepts ---
        Synset(id: "entity.n.01", name: "entity", pos: .noun, definition: "That which is perceived or known as having its own distinct existence.", lemmas: ["entity"], hyponymIDs: ["physical_entity.n.01", "abstraction.n.01", "act.n.01", "organism.n.01", "computer.n.01"]),
        Synset(id: "physical_entity.n.01", name: "physical entity", pos: .noun, definition: "An entity that has physical existence.", lemmas: ["physical entity", "physical_entity"], hypernymIDs: ["entity.n.01"], hyponymIDs: ["object.n.01", "substance.n.01", "food.n.01"]),
        Synset(id: "object.n.01", name: "object", pos: .noun, definition: "A tangible and visible entity; an entity that can cast a shadow.", lemmas: ["object", "physical object"], hypernymIDs: ["physical_entity.n.01"], hyponymIDs: ["living_thing.n.01", "artifact.n.01", "natural_object.n.01"]),
        Synset(id: "natural_object.n.01", name: "natural object", pos: .noun, definition: "An object occurring naturally; not made by humans.", lemmas: ["natural object", "natural_object"], hypernymIDs: ["object.n.01"]),
        
        // --- Living Organisms Taxonomy ---
        Synset(id: "living_thing.n.01", name: "living thing", pos: .noun, definition: "A living (animate) entity.", lemmas: ["living thing", "living_thing"], hypernymIDs: ["object.n.01"], hyponymIDs: ["organism.n.01"]),
        Synset(id: "organism.n.01", name: "organism", pos: .noun, definition: "A living thing that has an organized structure.", lemmas: ["organism", "living_thing"], hypernymIDs: ["entity.n.01", "living_thing.n.01"], hyponymIDs: ["animal.n.01", "plant.n.01"]),
        Synset(id: "animal.n.01", name: "animal", pos: .noun, definition: "A living organism that feeds on organic matter.", lemmas: ["animal", "fauna", "creature"], hypernymIDs: ["organism.n.01"], hyponymIDs: ["dog.n.01", "cat.n.01", "mammal.n.01", "bird.n.01", "fish.n.01", "reptile.n.01", "insect.n.01"]),
        Synset(id: "mammal.n.01", name: "mammal", pos: .noun, definition: "Any warm-blooded vertebrate having the skin covered with hair and female secreting milk.", lemmas: ["mammal", "mammalian"], hypernymIDs: ["animal.n.01"], hyponymIDs: ["canine.n.01", "feline.n.01", "primate.n.01", "rodent.n.01", "horse.n.01", "elephant.n.01", "whale.n.01"]),
        Synset(id: "canine.n.01", name: "canine", pos: .noun, definition: "Any of various carnivorous mammals with nonretractile claws.", lemmas: ["canine", "canid"], hypernymIDs: ["mammal.n.01"], hyponymIDs: ["wolf.n.01", "fox.n.01"]),
        Synset(id: "dog.n.01", name: "dog", pos: .noun, definition: "A domesticated carnivorous mammal (Canis familiaris).", lemmas: ["dog", "canine", "hound"], hypernymIDs: ["animal.n.01"]),
        Synset(id: "wolf.n.01", name: "wolf", pos: .noun, definition: "A wild carnivorous mammal of the dog family (Canis lupus).", lemmas: ["wolf"], hypernymIDs: ["canine.n.01"]),
        Synset(id: "fox.n.01", name: "fox", pos: .noun, definition: "A carnivorous mammal of the dog family with a pointed muzzle and bushy tail.", lemmas: ["fox"], hypernymIDs: ["canine.n.01"]),
        Synset(id: "feline.n.01", name: "feline", pos: .noun, definition: "Any of various agile carnivorous mammals usually having retractile claws.", lemmas: ["feline", "felid"], hypernymIDs: ["mammal.n.01"], hyponymIDs: ["lion.n.01", "tiger.n.01"]),
        Synset(id: "cat.n.01", name: "cat", pos: .noun, definition: "A small domesticated carnivorous mammal (Felis catus).", lemmas: ["cat", "feline", "kitty"], hypernymIDs: ["animal.n.01"]),
        Synset(id: "lion.n.01", name: "lion", pos: .noun, definition: "A large tawny-colored cat that lives in prides (Panthera leo).", lemmas: ["lion"], hypernymIDs: ["feline.n.01"]),
        Synset(id: "tiger.n.01", name: "tiger", pos: .noun, definition: "A large predatory cat with a yellow-brown coat striped with black (Panthera tigris).", lemmas: ["tiger"], hypernymIDs: ["feline.n.01"]),
        Synset(id: "horse.n.01", name: "horse", pos: .noun, definition: "A solid-hoofed herbivorous quadruped domesticated since prehistoric times.", lemmas: ["horse", "equine"], hypernymIDs: ["mammal.n.01"]),
        Synset(id: "elephant.n.01", name: "elephant", pos: .noun, definition: "A very large herbivorous mammal with a long trunk and ivory tusks.", lemmas: ["elephant"], hypernymIDs: ["mammal.n.01"]),
        Synset(id: "whale.n.01", name: "whale", pos: .noun, definition: "Any of the larger cetacean mammals having a streamlined body and horizontal tail fin.", lemmas: ["whale", "cetacean"], hypernymIDs: ["mammal.n.01"]),
        Synset(id: "rodent.n.01", name: "rodent", pos: .noun, definition: "A gnawing mammal having continuously growing incisors.", lemmas: ["rodent"], hypernymIDs: ["mammal.n.01"], hyponymIDs: ["mouse.n.01"]),
        Synset(id: "mouse.n.01", name: "mouse", pos: .noun, definition: "Any of numerous small rodents typically having a pointed snout and long tail.", lemmas: ["mouse"], hypernymIDs: ["rodent.n.01"]),
        
        // --- Primates & Humans ---
        Synset(id: "primate.n.01", name: "primate", pos: .noun, definition: "Any mammal of the group including lemurs, monkeys, apes, and humans.", lemmas: ["primate"], hypernymIDs: ["mammal.n.01"], hyponymIDs: ["human.n.01"]),
        Synset(id: "human.n.01", name: "human", pos: .noun, definition: "Any living or extinct member of the family Hominidae characterized by superior intelligence.", lemmas: ["human", "person", "individual", "someone"], hypernymIDs: ["primate.n.01"], hyponymIDs: ["scientist.n.01", "engineer.n.01", "teacher.n.01", "student.n.01", "doctor.n.01", "artist.n.01"]),
        Synset(id: "scientist.n.01", name: "scientist", pos: .noun, definition: "A person with advanced knowledge of one or more sciences.", lemmas: ["scientist", "researcher"], hypernymIDs: ["human.n.01"]),
        Synset(id: "engineer.n.01", name: "engineer", pos: .noun, definition: "A person who designs, builds, or maintains engines, machines, or public works.", lemmas: ["engineer", "developer"], hypernymIDs: ["human.n.01"]),
        Synset(id: "teacher.n.01", name: "teacher", pos: .noun, definition: "A person who helps others acquire knowledge, competences, or values.", lemmas: ["teacher", "educator", "instructor"], hypernymIDs: ["human.n.01"]),
        Synset(id: "student.n.01", name: "student", pos: .noun, definition: "A learner who is enrolled in an educational institution.", lemmas: ["student", "pupil", "scholar"], hypernymIDs: ["human.n.01"]),
        Synset(id: "doctor.n.01", name: "doctor", pos: .noun, definition: "A physician who is qualified and licensed to practice medicine.", lemmas: ["doctor", "physician"], hypernymIDs: ["human.n.01"]),
        Synset(id: "artist.n.01", name: "artist", pos: .noun, definition: "A person whose creative work shows sensitivity and imagination.", lemmas: ["artist", "creator"], hypernymIDs: ["human.n.01"]),
        
        // --- Other Animals: Birds, Fish, Reptiles, Insects ---
        Synset(id: "bird.n.01", name: "bird", pos: .noun, definition: "Warm-blooded egg-laying vertebrates characterized by feathers and wings.", lemmas: ["bird", "avian"], hypernymIDs: ["animal.n.01"], hyponymIDs: ["eagle.n.01", "penguin.n.01", "owl.n.01"]),
        Synset(id: "eagle.n.01", name: "eagle", pos: .noun, definition: "Any of various large diurnal birds of prey noted for broad wings and powerful soaring.", lemmas: ["eagle"], hypernymIDs: ["bird.n.01"]),
        Synset(id: "penguin.n.01", name: "penguin", pos: .noun, definition: "Short-legged flightless birds of cold southern regions.", lemmas: ["penguin"], hypernymIDs: ["bird.n.01"]),
        Synset(id: "owl.n.01", name: "owl", pos: .noun, definition: "Nocturnal bird of prey with large forward-facing eyes and silent flight.", lemmas: ["owl"], hypernymIDs: ["bird.n.01"]),
        Synset(id: "fish.n.01", name: "fish", pos: .noun, definition: "Any of various mostly cold-blooded aquatic vertebrates usually having scales and breathing through gills.", lemmas: ["fish"], hypernymIDs: ["animal.n.01"], hyponymIDs: ["salmon.n.01", "shark.n.01"]),
        Synset(id: "salmon.n.01", name: "salmon", pos: .noun, definition: "Any of various large food and game fishes of northern waters.", lemmas: ["salmon"], hypernymIDs: ["fish.n.01"]),
        Synset(id: "shark.n.01", name: "shark", pos: .noun, definition: "Any of numerous elongate mostly marine carnivorous fishes with cartilaginous skeletons.", lemmas: ["shark"], hypernymIDs: ["fish.n.01"]),
        Synset(id: "reptile.n.01", name: "reptile", pos: .noun, definition: "Cold-blooded vertebrates including tortoises, turtles, snakes, and lizards.", lemmas: ["reptile"], hypernymIDs: ["animal.n.01"], hyponymIDs: ["snake.n.01"]),
        Synset(id: "snake.n.01", name: "snake", pos: .noun, definition: "Limbless scaly elongate reptile with jaw mechanism allowing ingestion of large prey.", lemmas: ["snake", "serpent"], hypernymIDs: ["reptile.n.01"]),
        Synset(id: "insect.n.01", name: "insect", pos: .noun, definition: "Small air-breathing arthropod having a body divided into three parts.", lemmas: ["insect", "bug"], hypernymIDs: ["animal.n.01"], hyponymIDs: ["bee.n.01", "ant.n.01"]),
        Synset(id: "bee.n.01", name: "bee", pos: .noun, definition: "Any of numerous hairy-bodied insects producing honey and beeswax.", lemmas: ["bee", "honeybee"], hypernymIDs: ["insect.n.01"]),
        Synset(id: "ant.n.01", name: "ant", pos: .noun, definition: "Social insect living in organized colonies.", lemmas: ["ant"], hypernymIDs: ["insect.n.01"]),
        
        // --- Plants ---
        Synset(id: "plant.n.01", name: "plant", pos: .noun, definition: "A living organism of the vegetable group lacking locomotive movement.", lemmas: ["plant", "flora"], hypernymIDs: ["organism.n.01"], hyponymIDs: ["tree.n.01", "flower.n.01", "grass.n.01"]),
        Synset(id: "tree.n.01", name: "tree", pos: .noun, definition: "A tall perennial woody plant having a main trunk and branches.", lemmas: ["tree"], hypernymIDs: ["plant.n.01"], hyponymIDs: ["oak.n.01", "pine.n.01"]),
        Synset(id: "oak.n.01", name: "oak", pos: .noun, definition: "A tree of the genus Quercus bearing acorns and having tough wood.", lemmas: ["oak"], hypernymIDs: ["tree.n.01"]),
        Synset(id: "pine.n.01", name: "pine", pos: .noun, definition: "A coniferous evergreen tree with needle-shaped leaves and cones.", lemmas: ["pine"], hypernymIDs: ["tree.n.01"]),
        Synset(id: "flower.n.01", name: "flower", pos: .noun, definition: "A plant cultivated for its blooms or blossoms.", lemmas: ["flower", "blossom"], hypernymIDs: ["plant.n.01"], hyponymIDs: ["rose.n.01"]),
        Synset(id: "rose.n.01", name: "rose", pos: .noun, definition: "Any of many shrubs of the genus Rosa that produce fragrant flowers.", lemmas: ["rose"], hypernymIDs: ["flower.n.01"]),
        Synset(id: "grass.n.01", name: "grass", pos: .noun, definition: "Narrow-leaved green herbage suitable for grazing.", lemmas: ["grass"], hypernymIDs: ["plant.n.01"]),

        // --- Artifacts & Technology ---
        Synset(id: "artifact.n.01", name: "artifact", pos: .noun, definition: "A man-made object taken as a whole.", lemmas: ["artifact", "artefact"], hypernymIDs: ["object.n.01"], hyponymIDs: ["instrumentality.n.01", "structure.n.01"]),
        Synset(id: "structure.n.01", name: "structure", pos: .noun, definition: "A thing constructed; a complex construction or edifice.", lemmas: ["structure", "construction"], hypernymIDs: ["artifact.n.01"], hyponymIDs: ["building.n.01", "bridge.n.01"]),
        Synset(id: "building.n.01", name: "building", pos: .noun, definition: "A structure that has a roof and walls and stands more or less permanently.", lemmas: ["building", "edifice"], hypernymIDs: ["structure.n.01"], hyponymIDs: ["house.n.01", "hospital.n.01", "school.n.01"]),
        Synset(id: "house.n.01", name: "house", pos: .noun, definition: "A dwelling that serves as living quarters for one or more families.", lemmas: ["house", "dwelling", "home"], hypernymIDs: ["building.n.01"]),
        Synset(id: "hospital.n.01", name: "hospital", pos: .noun, definition: "A health facility where patients receive medical treatment.", lemmas: ["hospital", "infirmary", "clinic"], hypernymIDs: ["building.n.01"]),
        Synset(id: "school.n.01", name: "school", pos: .noun, definition: "An educational institution where students learn.", lemmas: ["school", "academy"], hypernymIDs: ["building.n.01"]),
        Synset(id: "bridge.n.01", name: "bridge", pos: .noun, definition: "A structure that allows people or vehicles to cross an obstacle.", lemmas: ["bridge"], hypernymIDs: ["structure.n.01"]),
        Synset(id: "instrumentality.n.01", name: "instrumentality", pos: .noun, definition: "An artifact that is instrumental in accomplishing some end.", lemmas: ["instrumentality", "implement"], hypernymIDs: ["artifact.n.01"], hyponymIDs: ["device.n.01", "machine.n.01", "tool.n.01"]),
        Synset(id: "machine.n.01", name: "machine", pos: .noun, definition: "Any mechanical or electrical device that transmits or modifies energy.", lemmas: ["machine"], hypernymIDs: ["instrumentality.n.01"], hyponymIDs: ["computer.n.01", "vehicle.n.01"]),
        Synset(id: "computer.n.01", name: "computer", pos: .noun, definition: "An electronic device for storing and processing data.", lemmas: ["computer", "machine", "data processor"], hypernymIDs: ["entity.n.01", "machine.n.01"], hyponymIDs: ["server.n.01", "laptop.n.01", "supercomputer.n.01"]),
        Synset(id: "server.n.01", name: "server", pos: .noun, definition: "A computer that provides data and services to other computers on a network.", lemmas: ["server", "host"], hypernymIDs: ["computer.n.01"]),
        Synset(id: "laptop.n.01", name: "laptop", pos: .noun, definition: "A portable computer small enough to use on one's lap.", lemmas: ["laptop", "notebook"], hypernymIDs: ["computer.n.01"]),
        Synset(id: "supercomputer.n.01", name: "supercomputer", pos: .noun, definition: "A particularly powerful mainframe computer.", lemmas: ["supercomputer"], hypernymIDs: ["computer.n.01"]),
        Synset(id: "device.n.01", name: "device", pos: .noun, definition: "An instrumentality invented for a particular purpose.", lemmas: ["device", "gadget"], hypernymIDs: ["instrumentality.n.01"], hyponymIDs: ["phone.n.01", "camera.n.01", "sensor.n.01"]),
        Synset(id: "phone.n.01", name: "phone", pos: .noun, definition: "An electronic telecommunication device for transmitting sound or data.", lemmas: ["phone", "telephone", "smartphone"], hypernymIDs: ["device.n.01"]),
        Synset(id: "camera.n.01", name: "camera", pos: .noun, definition: "Equipment for taking photographs or recording video.", lemmas: ["camera"], hypernymIDs: ["device.n.01"]),
        Synset(id: "sensor.n.01", name: "sensor", pos: .noun, definition: "A device that detects or measures a physical property.", lemmas: ["sensor", "detector"], hypernymIDs: ["device.n.01"]),
        Synset(id: "tool.n.01", name: "tool", pos: .noun, definition: "An implement used to carry out a particular function.", lemmas: ["tool", "instrument"], hypernymIDs: ["instrumentality.n.01"], hyponymIDs: ["hammer.n.01"]),
        Synset(id: "hammer.n.01", name: "hammer", pos: .noun, definition: "A hand tool with a heavy rigid head used to deliver an impact.", lemmas: ["hammer"], hypernymIDs: ["tool.n.01"]),
        Synset(id: "vehicle.n.01", name: "vehicle", pos: .noun, definition: "A conveyance that transports people or objects.", lemmas: ["vehicle", "conveyance"], hypernymIDs: ["machine.n.01"], hyponymIDs: ["car.n.01", "truck.n.01", "airplane.n.01", "ship.n.01", "bicycle.n.01"]),
        Synset(id: "car.n.01", name: "car", pos: .noun, definition: "A 4-wheeled motor vehicle used for land transport.", lemmas: ["car", "automobile", "motorcar"], hypernymIDs: ["vehicle.n.01"]),
        Synset(id: "truck.n.01", name: "truck", pos: .noun, definition: "A large motor vehicle designed to carry heavy cargo.", lemmas: ["truck", "lorry"], hypernymIDs: ["vehicle.n.01"]),
        Synset(id: "airplane.n.01", name: "airplane", pos: .noun, definition: "An aircraft that has fixed wings and is powered by propellers or jets.", lemmas: ["airplane", "aeroplane", "plane"], hypernymIDs: ["vehicle.n.01"]),
        Synset(id: "ship.n.01", name: "ship", pos: .noun, definition: "A vessel that carries passengers or cargo over water.", lemmas: ["ship", "vessel"], hypernymIDs: ["vehicle.n.01"]),
        Synset(id: "bicycle.n.01", name: "bicycle", pos: .noun, definition: "A wheeled vehicle that has two wheels and is moved by foot pedals.", lemmas: ["bicycle", "bike", "cycle"], hypernymIDs: ["vehicle.n.01"]),

        // --- Substances & Matter ---
        Synset(id: "substance.n.01", name: "substance", pos: .noun, definition: "The real physical matter of which a person or thing consists.", lemmas: ["substance", "matter"], hypernymIDs: ["physical_entity.n.01"], hyponymIDs: ["water.n.01", "air.n.01", "metal.n.01"]),
        Synset(id: "water.n.01", name: "water", pos: .noun, definition: "Binary compound that occurs at room temperature as a clear liquid (H2O).", lemmas: ["water", "h2o"], hypernymIDs: ["substance.n.01"]),
        Synset(id: "air.n.01", name: "air", pos: .noun, definition: "A mixture of gases that encloses the Earth in an atmosphere.", lemmas: ["air", "atmosphere"], hypernymIDs: ["substance.n.01"]),
        Synset(id: "metal.n.01", name: "metal", pos: .noun, definition: "Any of several chemical elements that are good conductors of heat and electricity.", lemmas: ["metal", "metallic element"], hypernymIDs: ["substance.n.01"], hyponymIDs: ["gold.n.01", "iron.n.01"]),
        Synset(id: "gold.n.01", name: "gold", pos: .noun, definition: "A soft yellow malleable ductile precious metallic element (Au).", lemmas: ["gold", "au"], hypernymIDs: ["metal.n.01"]),
        Synset(id: "iron.n.01", name: "iron", pos: .noun, definition: "A heavy ductile magnetic metallic element (Fe).", lemmas: ["iron", "fe"], hypernymIDs: ["metal.n.01"]),
        
        // --- Food ---
        Synset(id: "food.n.01", name: "food", pos: .noun, definition: "Any substance that can be metabolized by an organism to give energy and build tissue.", lemmas: ["food", "nutrient"], hypernymIDs: ["physical_entity.n.01"], hyponymIDs: ["fruit.n.01", "bread.n.01"]),
        Synset(id: "fruit.n.01", name: "fruit", pos: .noun, definition: "The ripened reproductive body of a seed plant.", lemmas: ["fruit"], hypernymIDs: ["food.n.01"], hyponymIDs: ["apple.n.01", "banana.n.01"]),
        Synset(id: "apple.n.01", name: "apple", pos: .noun, definition: "Fruit with red or yellow or green skin and sweet to tart crisp whitish flesh.", lemmas: ["apple"], hypernymIDs: ["fruit.n.01"]),
        Synset(id: "banana.n.01", name: "banana", pos: .noun, definition: "Elongated crescent-shaped yellow fruit with soft sweet flesh.", lemmas: ["banana"], hypernymIDs: ["fruit.n.01"]),
        Synset(id: "bread.n.01", name: "bread", pos: .noun, definition: "Food made from dough of flour or meal and usually raised with yeast.", lemmas: ["bread", "loaf"], hypernymIDs: ["food.n.01"]),

        // --- Abstractions, Disciplines, Science & Mathematics ---
        Synset(id: "abstraction.n.01", name: "abstraction", pos: .noun, definition: "A general concept formed by extracting common features from examples.", lemmas: ["abstraction", "abstract entity"], hypernymIDs: ["entity.n.01"], hyponymIDs: ["concept.n.01", "measure.n.01", "discipline.n.01", "communication.n.01", "information.n.01", "group.n.01"]),
        Synset(id: "concept.n.01", name: "concept", pos: .noun, definition: "An abstract or general idea inferred from specific instances.", lemmas: ["concept", "conception"], hypernymIDs: ["abstraction.n.01"], hyponymIDs: ["idea.n.01", "theory.n.01"]),
        Synset(id: "idea.n.01", name: "idea", pos: .noun, definition: "The content of cognition; the main thing you are thinking about.", lemmas: ["idea", "thought"], hypernymIDs: ["concept.n.01"]),
        Synset(id: "theory.n.01", name: "theory", pos: .noun, definition: "A well-substantiated explanation of some aspect of the natural world.", lemmas: ["theory", "hypothesis"], hypernymIDs: ["concept.n.01"]),
        Synset(id: "measure.n.01", name: "measure", pos: .noun, definition: "How much there is or how many there are of something that you can quantify.", lemmas: ["measure", "quantity", "amount"], hypernymIDs: ["abstraction.n.01"], hyponymIDs: ["time.n.01", "distance.n.01", "speed.n.01"]),
        Synset(id: "time.n.01", name: "time", pos: .noun, definition: "The continuum of experience in which events pass from future through present to past.", lemmas: ["time", "period"], hypernymIDs: ["measure.n.01"]),
        Synset(id: "distance.n.01", name: "distance", pos: .noun, definition: "The property created by the space between two points.", lemmas: ["distance", "length"], hypernymIDs: ["measure.n.01"]),
        Synset(id: "speed.n.01", name: "speed", pos: .noun, definition: "Distance travelled per unit time.", lemmas: ["speed", "velocity"], hypernymIDs: ["measure.n.01"]),
        Synset(id: "discipline.n.01", name: "discipline", pos: .noun, definition: "A branch of knowledge; a field of study.", lemmas: ["discipline", "subject", "field"], hypernymIDs: ["abstraction.n.01"], hyponymIDs: ["science.n.01", "mathematics.n.01", "linguistics.n.01"]),
        Synset(id: "science.n.01", name: "science", pos: .noun, definition: "A systematically organized body of knowledge on any subject.", lemmas: ["science", "scientific discipline"], hypernymIDs: ["discipline.n.01"], hyponymIDs: ["computer_science.n.01", "physics.n.01", "biology.n.01", "chemistry.n.01"]),
        Synset(id: "computer_science.n.01", name: "computer science", pos: .noun, definition: "The branch of engineering science that studies computable processes and structures.", lemmas: ["computer science", "computer_science", "computing", "informatics"], hypernymIDs: ["science.n.01"]),
        Synset(id: "physics.n.01", name: "physics", pos: .noun, definition: "The science of matter and energy and their interactions.", lemmas: ["physics", "physical science"], hypernymIDs: ["science.n.01"]),
        Synset(id: "chemistry.n.01", name: "chemistry", pos: .noun, definition: "The science of matter; the branch of modern science that studies elemental composition.", lemmas: ["chemistry", "chemical science"], hypernymIDs: ["science.n.01"]),
        Synset(id: "biology.n.01", name: "biology", pos: .noun, definition: "The science that studies living organisms.", lemmas: ["biology", "biological science"], hypernymIDs: ["science.n.01"]),
        Synset(id: "mathematics.n.01", name: "mathematics", pos: .noun, definition: "A science dealing with logic of quantity, shape and arrangement.", lemmas: ["mathematics", "math", "maths"], hypernymIDs: ["discipline.n.01"]),
        Synset(id: "linguistics.n.01", name: "linguistics", pos: .noun, definition: "The scientific study of language.", lemmas: ["linguistics"], hypernymIDs: ["discipline.n.01"]),
        
        // --- Communication & Information ---
        Synset(id: "communication.n.01", name: "communication", pos: .noun, definition: "The activity of conveying information.", lemmas: ["communication"], hypernymIDs: ["abstraction.n.01"], hyponymIDs: ["language.n.01", "document.n.01"]),
        Synset(id: "language.n.01", name: "language", pos: .noun, definition: "A systematic means of communicating by the use of sounds or conventional symbols.", lemmas: ["language", "tongue", "linguistic system"], hypernymIDs: ["communication.n.01"], hyponymIDs: ["natural_language.n.01", "programming_language.n.01"]),
        Synset(id: "natural_language.n.01", name: "natural language", pos: .noun, definition: "A human written or spoken language as opposed to a computer language.", lemmas: ["natural language", "natural_language"], hypernymIDs: ["language.n.01"]),
        Synset(id: "programming_language.n.01", name: "programming language", pos: .noun, definition: "An artificial language designed to communicate instructions to a computer.", lemmas: ["programming language", "programming_language", "source code"], hypernymIDs: ["language.n.01"]),
        Synset(id: "document.n.01", name: "document", pos: .noun, definition: "Writing that provides information (especially information of an official nature).", lemmas: ["document", "written document", "text"], hypernymIDs: ["communication.n.01"], hyponymIDs: ["book.n.01"]),
        Synset(id: "book.n.01", name: "book", pos: .noun, definition: "A written work or composition that has been published.", lemmas: ["book", "volume"], hypernymIDs: ["document.n.01"]),
        Synset(id: "information.n.01", name: "information", pos: .noun, definition: "Knowledge acquired through study, experience, or instruction.", lemmas: ["information", "info"], hypernymIDs: ["abstraction.n.01"], hyponymIDs: ["data.n.01", "knowledge.n.01", "software.n.01", "algorithm.n.01", "database.n.01"]),
        Synset(id: "data.n.01", name: "data", pos: .noun, definition: "A collection of facts from which conclusions may be drawn.", lemmas: ["data", "information points"], hypernymIDs: ["information.n.01"]),
        Synset(id: "knowledge.n.01", name: "knowledge", pos: .noun, definition: "The psychological result of perception and learning and reasoning.", lemmas: ["knowledge", "understanding"], hypernymIDs: ["information.n.01"]),
        Synset(id: "software.n.01", name: "software", pos: .noun, definition: "Written programs or procedures or rules and associated documentation.", lemmas: ["software", "software system", "computer program"], hypernymIDs: ["information.n.01"]),
        Synset(id: "algorithm.n.01", name: "algorithm", pos: .noun, definition: "A precise rule or set of rules specifying how to solve some problem.", lemmas: ["algorithm", "algorithmic rule"], hypernymIDs: ["information.n.01"]),
        Synset(id: "database.n.01", name: "database", pos: .noun, definition: "An organized body of related information.", lemmas: ["database"], hypernymIDs: ["information.n.01"]),
        Synset(id: "group.n.01", name: "group", pos: .noun, definition: "Any number of entities (members) considered as a unit.", lemmas: ["group", "grouping"], hypernymIDs: ["abstraction.n.01"], hyponymIDs: ["organization.n.01"]),
        Synset(id: "organization.n.01", name: "organization", pos: .noun, definition: "A group of people who work together in an organized way for a shared purpose.", lemmas: ["organization", "organisation"], hypernymIDs: ["group.n.01"], hyponymIDs: ["company.n.01", "university.n.01"]),
        Synset(id: "company.n.01", name: "company", pos: .noun, definition: "An institution created to conduct business.", lemmas: ["company", "corporation", "firm"], hypernymIDs: ["organization.n.01"]),
        Synset(id: "university.n.01", name: "university", pos: .noun, definition: "An institution of higher education and research.", lemmas: ["university", "college"], hypernymIDs: ["organization.n.01"]),

        // --- Acts, Processes & Events ---
        Synset(id: "act.n.01", name: "act", pos: .noun, definition: "Something that people do or cause to happen.", lemmas: ["act", "action", "deed"], hypernymIDs: ["entity.n.01"], hyponymIDs: ["activity.n.01", "process.n.01", "event.n.01"]),
        Synset(id: "activity.n.01", name: "activity", pos: .noun, definition: "Any specific behavior or state of being active.", lemmas: ["activity"], hypernymIDs: ["act.n.01"], hyponymIDs: ["experiment.n.01"]),
        Synset(id: "process.n.01", name: "process", pos: .noun, definition: "A sustained phenomenon or one marked by gradual changes through a series of states.", lemmas: ["process", "procedure"], hypernymIDs: ["act.n.01"], hyponymIDs: ["computation.n.01"]),
        Synset(id: "event.n.01", name: "event", pos: .noun, definition: "Something that happens at a given place and time.", lemmas: ["event", "happening"], hypernymIDs: ["act.n.01"]),
        Synset(id: "experiment.n.01", name: "experiment", pos: .noun, definition: "The act of conducting a controlled test or investigation.", lemmas: ["experiment", "trial"], hypernymIDs: ["activity.n.01"]),
        Synset(id: "computation.n.01", name: "computation", pos: .noun, definition: "The procedure of calculating; determining something by mathematical or logical methods.", lemmas: ["computation", "calculation"], hypernymIDs: ["process.n.01"]),

        // --- Verbs (Action & Cognition) ---
        Synset(id: "move.v.01", name: "move", pos: .verb, definition: "Change position or place.", lemmas: ["move", "travel"], hyponymIDs: ["run.v.01", "walk.v.01", "fly.v.01"]),
        Synset(id: "run.v.01", name: "run", pos: .verb, definition: "Move fast by using one's feet.", lemmas: ["run"], hypernymIDs: ["move.v.01"]),
        Synset(id: "walk.v.01", name: "walk", pos: .verb, definition: "Advance by steps without running.", lemmas: ["walk"], hypernymIDs: ["move.v.01"]),
        Synset(id: "fly.v.01", name: "fly", pos: .verb, definition: "Travel through the air.", lemmas: ["fly"], hypernymIDs: ["move.v.01"]),
        Synset(id: "think.v.01", name: "think", pos: .verb, definition: "Use one's mind actively to form ideas or make decisions.", lemmas: ["think", "cogitate"], hyponymIDs: ["reason.v.01", "learn.v.01"]),
        Synset(id: "reason.v.01", name: "reason", pos: .verb, definition: "Think logically and evaluate arguments.", lemmas: ["reason"], hypernymIDs: ["think.v.01"]),
        Synset(id: "learn.v.01", name: "learn", pos: .verb, definition: "Acquire knowledge or skills.", lemmas: ["learn", "acquire"], hypernymIDs: ["think.v.01"]),
        Synset(id: "communicate.v.01", name: "communicate", pos: .verb, definition: "Share or exchange information, news, or ideas.", lemmas: ["communicate"], hyponymIDs: ["speak.v.01", "write.v.01", "read.v.01"]),
        Synset(id: "speak.v.01", name: "speak", pos: .verb, definition: "Express in speech.", lemmas: ["speak", "talk"], hypernymIDs: ["communicate.v.01"]),
        Synset(id: "write.v.01", name: "write", pos: .verb, definition: "Produce words, letters, or symbols on a surface.", lemmas: ["write"], hypernymIDs: ["communicate.v.01"]),
        Synset(id: "read.v.01", name: "read", pos: .verb, definition: "Look at and comprehend the meaning of written words.", lemmas: ["read"], hypernymIDs: ["communicate.v.01"]),
        Synset(id: "create.v.01", name: "create", pos: .verb, definition: "Bring something into existence.", lemmas: ["create", "make"], hyponymIDs: ["build.v.01", "program.v.01"]),
        Synset(id: "build.v.01", name: "build", pos: .verb, definition: "Construct by putting parts together.", lemmas: ["build", "construct"], hypernymIDs: ["create.v.01"]),
        Synset(id: "program.v.01", name: "program", pos: .verb, definition: "Write computer code or instruct a computing machine.", lemmas: ["program", "code"], hypernymIDs: ["create.v.01"]),
        Synset(id: "compute.v.01", name: "compute", pos: .verb, definition: "Determine mathematically or by computer.", lemmas: ["compute", "calculate"], hyponymIDs: ["analyze.v.01"]),
        Synset(id: "analyze.v.01", name: "analyze", pos: .verb, definition: "Examine methodically and in detail.", lemmas: ["analyze", "analyse"], hypernymIDs: ["compute.v.01"]),

        // --- Adjectives (Qualitative Attributes) ---
        Synset(id: "fast.a.01", name: "fast", pos: .adjective, definition: "Moving or operating at high speed.", lemmas: ["fast", "quick", "rapid"]),
        Synset(id: "slow.a.01", name: "slow", pos: .adjective, definition: "Moving or operating with low speed.", lemmas: ["slow"]),
        Synset(id: "large.a.01", name: "large", pos: .adjective, definition: "Of considerable or relatively great size.", lemmas: ["large", "big"]),
        Synset(id: "small.a.01", name: "small", pos: .adjective, definition: "Of limited size or extent.", lemmas: ["small", "little"]),
        Synset(id: "good.a.01", name: "good", pos: .adjective, definition: "Having the qualities desirable or appropriate.", lemmas: ["good"]),
        Synset(id: "bad.a.01", name: "bad", pos: .adjective, definition: "Of poor quality or low standard.", lemmas: ["bad"]),
        Synset(id: "accurate.a.01", name: "accurate", pos: .adjective, definition: "Correct in all details; exact.", lemmas: ["accurate", "precise"]),
        Synset(id: "efficient.a.01", name: "efficient", pos: .adjective, definition: "Achieving maximum productivity with minimum wasted effort.", lemmas: ["efficient"]),
        Synset(id: "intelligent.a.01", name: "intelligent", pos: .adjective, definition: "Having or showing intelligence or mental capacity.", lemmas: ["intelligent", "smart"])
    ]
}

