import Foundation

/// CountVectorizer converts a collection of text documents to a matrix of token counts.
public final class CountVectorizer: @unchecked Sendable {
    /// The max features.
    public let maxFeatures: Int?
    /// The lowercase.
    public let lowercase: Bool
    
    private let lock = NSLock()
    private var _vocabulary: [String: Int] = [:]

    /// Mapping of token strings to column feature indices.
    public var vocabulary: [String: Int] {
        lock.withLock { _vocabulary }
    }
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - maxFeatures: The max features. Must be strictly positive if provided.
    ///   - lowercase: Whether to convert text to lowercase before tokenization.
    public init(maxFeatures: Int? = nil, lowercase: Bool = true) {
        if let maxF = maxFeatures {
            precondition(maxF > 0, "maxFeatures must be positive, got \(maxF)")
        }
        self.maxFeatures = maxFeatures
        self.lowercase = lowercase
    }
    
    /// Fits vocabulary from document corpus.
    /// - Parameters:
    ///   - documents: Collection of text documents to process.
    public func fit(documents: [String]) {
        var counts: [String: Int] = [:]
        for doc in documents {
            let tokens = tokenize(doc)
            for token in tokens {
                counts[token, default: 0] += 1
            }
        }
        
        var sortedTokens = counts.keys.sorted { t1, t2 in
            let c1 = counts[t1]!
            let c2 = counts[t2]!
            if c1 != c2 {
                return c1 > c2
            }
            return t1 < t2
        }
        if let maxF = maxFeatures {
            sortedTokens = Array(sortedTokens.prefix(max(0, maxF)))
        }
        
        var vocab: [String: Int] = [:]
        for (i, token) in sortedTokens.sorted().enumerated() {
            vocab[token] = i
        }
        lock.withLock {
            self._vocabulary = vocab
        }
    }
    
    /// Transforms document corpus to term count feature matrix.
    /// - Parameters:
    ///   - documents: Collection of text documents to process.
    /// - Returns: 2D numerical matrix of shape `[N, P]`.
    public func transform(documents: [String]) -> [[Double]] {
        let currentVocab = lock.withLock { _vocabulary }
        let vocabSize = currentVocab.count
        guard vocabSize > 0 else { return documents.map { _ in [] } }
        
        return documents.map { doc in
            var row = [Double](repeating: 0.0, count: vocabSize)
            let tokens = tokenize(doc)
            for token in tokens {
                if let idx = currentVocab[token] {
                    row[idx] += 1.0
                }
            }
            return row
        }
    }
    
    /// Fits vocabulary and transforms documents in a single pass.
    /// - Parameters:
    ///   - documents: Collection of text documents to process.
    /// - Returns: 2D numerical matrix of shape `[N, P]`.
    public func fitTransform(documents: [String]) -> [[Double]] {
        fit(documents: documents)
        return transform(documents: documents)
    }
    
    private func tokenize(_ text: String) -> [String] {
        let input = lowercase ? text.lowercased() : text
        return input.components(separatedBy: CharacterSet.alphanumerics.inverted).filter { !$0.isEmpty }
    }
}
