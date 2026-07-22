import Foundation

/// CountVectorizer converts a collection of text documents to a matrix of token counts.
public final class CountVectorizer: @unchecked Sendable {
    public let maxFeatures: Int?
    public let lowercase: Bool
    
    private var vocabulary: [String: Int] = [:]
    
    public init(maxFeatures: Int? = nil, lowercase: Bool = true) {
        self.maxFeatures = maxFeatures
        self.lowercase = lowercase
    }
    
    /// Fits vocabulary from document corpus.
    public func fit(documents: [String]) {
        var counts: [String: Int] = [:]
        for doc in documents {
            let tokens = tokenize(doc)
            for token in tokens {
                counts[token, default: 0] += 1
            }
        }
        
        var sortedTokens = counts.keys.sorted { counts[$0]! > counts[$1]! }
        if let maxF = maxFeatures {
            sortedTokens = Array(sortedTokens.prefix(maxF))
        }
        
        var vocab: [String: Int] = [:]
        for (i, token) in sortedTokens.sorted().enumerated() {
            vocab[token] = i
        }
        self.vocabulary = vocab
    }
    
    /// Transforms document corpus to term count feature matrix.
    public func transform(documents: [String]) -> [[Double]] {
        let vocabSize = vocabulary.count
        guard vocabSize > 0 else { return documents.map { _ in [] } }
        
        return documents.map { doc in
            var row = [Double](repeating: 0.0, count: vocabSize)
            let tokens = tokenize(doc)
            for token in tokens {
                if let idx = vocabulary[token] {
                    row[idx] += 1.0
                }
            }
            return row
        }
    }
    
    /// Fits vocabulary and transforms documents in a single pass.
    public func fitTransform(documents: [String]) -> [[Double]] {
        fit(documents: documents)
        return transform(documents: documents)
    }
    
    private func tokenize(_ text: String) -> [String] {
        let input = lowercase ? text.lowercased() : text
        return input.components(separatedBy: CharacterSet.alphanumerics.inverted).filter { !$0.isEmpty }
    }
}
