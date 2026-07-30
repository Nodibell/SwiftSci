import Foundation
#if canImport(Accelerate)
import Accelerate
#endif

/// A representation of pre-trained word embeddings (e.g. Word2Vec/GloVe).

public struct WordEmbeddings: Sendable {
    /// The embeddings.
    public let embeddings: [String: [Double]]
    /// The vector size.
    public let vectorSize: Int
    
    /// Initializes WordEmbeddings by parsing a pre-trained vector file.
    /// Supports standard space-separated format: `word float1 float2 ... floatN`
    public init(filePath: String) throws {
        let content = try String(contentsOfFile: filePath, encoding: .utf8)
        let lines = content.components(separatedBy: .newlines)
        
        var parsedEmbeddings = [String: [Double]]()
        var size = 0
        
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            
            let parts = trimmed.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
            guard parts.count > 1 else { continue }
            
            // Handle header lines (often found in Word2Vec files: "vocab_size vector_size")
            if parts.count == 2, let _ = Int(parts[0]), let _ = Int(parts[1]) {
                continue
            }
            
            let word = parts[0]
            let vector = parts[1...].compactMap { Double($0) }
            
            guard !vector.isEmpty else { continue }
            
            if size == 0 {
                size = vector.count
            }
            
            // Only add vectors of consistent size
            if vector.count == size {
                parsedEmbeddings[word] = vector
            }
        }
        
        self.embeddings = parsedEmbeddings
        self.vectorSize = size
    }
    
    /// Initializes WordEmbeddings from a pre-defined dictionary.
    public init(embeddings: [String: [Double]]) {
        self.embeddings = embeddings
        self.vectorSize = embeddings.first?.value.count ?? 0
    }
    
    /// Returns the embedding vector for the given word.
    public func vector(for word: String) -> [Double]? {
        if let vec = embeddings[word] {
            return vec
        }
        // Fallback to lowercase
        return embeddings[word.lowercased()]
    }

    /// Computes the cosine similarity between two words.
    public func cosineSimilarity(_ word1: String, _ word2: String) -> Double? {
        guard let v1 = vector(for: word1), let v2 = vector(for: word2) else {
            return nil
        }
        
        guard v1.count == v2.count, v1.count > 0 else {
            return nil
        }

        #if canImport(Accelerate)
        var dotProduct = 0.0
        var normA = 0.0
        var normB = 0.0
        let n = vDSP_Length(v1.count)

        vDSP_dotprD(v1, 1, v2, 1, &dotProduct, n)
        vDSP_svesqD(v1, 1, &normA, n)
        vDSP_svesqD(v2, 1, &normB, n)

        guard normA > 0.0, normB > 0.0 else {
            return nil
        }
        return dotProduct / (sqrt(normA) * sqrt(normB))
        #else
        var dotProduct = 0.0
        var normA = 0.0
        var normB = 0.0
        
        for i in 0..<v1.count {
            dotProduct += v1[i] * v2[i]
            normA += v1[i] * v1[i]
            normB += v2[i] * v2[i]
        }
        
        guard normA > 0.0, normB > 0.0 else {
            return nil
        }
        
        return dotProduct / (sqrt(normA) * sqrt(normB))
        #endif
    }

    
    /// Finds the top K most similar words to the given word.
    public func mostSimilar(to word: String, topK: Int = 10) -> [(word: String, similarity: Double)]? {
        guard let _ = vector(for: word) else { return nil }
        
        var similarities = [(word: String, similarity: Double)]()
        similarities.reserveCapacity(embeddings.count)
        
        for (w, _) in embeddings {
            if w == word { continue } // Skip self
            
            if let sim = cosineSimilarity(word, w) {
                similarities.append((w, sim))
            }
        }
        
        similarities.sort { $0.similarity > $1.similarity }
        
        return Array(similarities.prefix(topK))
    }
}
