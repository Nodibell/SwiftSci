import Foundation

/// TF-IDF Vectorizer for text feature extraction.
public actor TFIDFVectorizer {
    /// Stop words list to filter out during tokenization.
    private static let stopWords: Set<String> = [
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to", "in", "on", 
        "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", 
        "before", "after", "above", "below", "from", "up", "down", "is", "are", "was", "were", 
        "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", 
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
        "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", 
        "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", 
        "who", "whom", "this", "that", "these", "those", "am", "as"
    ]
    
    /// Map of word to vocabulary index.
    public private(set) var vocabulary: [String: Int] = [:]
    
    /// Inverse document frequency (IDF) vector. Shape: [vocabSize]
    public private(set) var idfs: [Double] = []
    
    /// Initializes a new TFIDFVectorizer.
    public init() {}
    
    /// Tokenizes and cleans a document string.
    /// - Parameter doc: Input string document.
    /// - Returns: A list of clean tokens.
    private func tokenize(_ doc: String) -> [String] {
        let cleaned = doc.lowercased().map { char -> String in
            if char.isLetter || char.isNumber || char.isWhitespace {
                return String(char)
            } else {
                return " "
            }
        }.joined()
        
        let tokens = cleaned.components(separatedBy: .whitespacesAndNewlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            
        return tokens.filter { !Self.stopWords.contains($0) }
    }
    
    /// Fits the vectorizer on a corpus of documents, building the vocabulary and computing IDFs.
    /// - Parameter documents: List of string documents.
    public func fit(_ documents: [String]) throws {
        guard !documents.isEmpty else {
            throw NLPError.emptyInput
        }
        
        // 1. Tokenize all documents and build vocabulary
        var allTokens = Set<String>()
        var docTokenSets = [Set<String>]()
        
        for doc in documents {
            let tokens = tokenize(doc)
            let tokenSet = Set(tokens)
            docTokenSets.append(tokenSet)
            for tok in tokens {
                allTokens.insert(tok)
            }
        }
        
        guard !allTokens.isEmpty else {
            throw NLPError.invalidVocabulary
        }
        
        // Sort vocabulary for deterministic indexing
        let sortedVocab = allTokens.sorted()
        var vocabMap = [String: Int]()
        for (idx, word) in sortedVocab.enumerated() {
            vocabMap[word] = idx
        }
        
        // 2. Compute IDF for each vocabulary term using smooth IDF: log((1 + N) / (1 + df)) + 1
        let n = Double(documents.count)
        var idfValues = [Double](repeating: 0.0, count: sortedVocab.count)
        
        for (idx, word) in sortedVocab.enumerated() {
            let df = Double(docTokenSets.filter { $0.contains(word) }.count)
            idfValues[idx] = log((1.0 + n) / (1.0 + df)) + 1.0
        }
        
        self.vocabulary = vocabMap
        self.idfs = idfValues
    }
    
    /// Transforms the documents into a TF-IDF matrix.
    /// - Parameter documents: List of string documents.
    /// - Returns: A 2D array of shape [documents, vocabSize].
    public func transform(_ documents: [String]) throws -> [[Double]] {
        guard !vocabulary.isEmpty, !idfs.isEmpty else {
            throw NLPError.fittingRequired
        }
        guard !documents.isEmpty else {
            throw NLPError.emptyInput
        }
        
        let vocabSize = vocabulary.count
        var result = [[Double]]()
        
        for doc in documents {
            let docTokens = tokenize(doc)
            var termCounts = [String: Int]()
            for tok in docTokens {
                termCounts[tok, default: 0] += 1
            }
            
            var vector = [Double](repeating: 0.0, count: vocabSize)
            if !docTokens.isEmpty {
                for tok in docTokens {
                    if let idx = vocabulary[tok] {
                        let tf = Double(termCounts[tok]!) / Double(docTokens.count)
                        vector[idx] = tf * idfs[idx]
                    }
                }
            }
            result.append(vector)
        }
        
        return result
    }
    
    /// Fits the model and transforms the documents.
    /// - Parameter documents: List of string documents.
    /// - Returns: A 2D array of shape [documents, vocabSize].
    public func fitTransform(_ documents: [String]) throws -> [[Double]] {
        try fit(documents)
        return try transform(documents)
    }
}
