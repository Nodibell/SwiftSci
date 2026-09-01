import Foundation
import SwiftDataFrame

/// A sparse vector representation containing only non-zero entries.
public struct SparseVector: Sendable, Codable, Equatable {
    /// Non-zero feature column indices.
    public let indices: [Int]
    /// Non-zero feature values corresponding to `indices`.
    public let values: [Double]
    /// Total dimension of the feature space.
    public let dimension: Int

    /// Creates a new sparse vector representation.
    /// - Parameters:
    ///   - indices: Non-zero column indices.
    ///   - values: Corresponding non-zero values.
    ///   - dimension: Total vector dimension.
    public init(indices: [Int], values: [Double], dimension: Int) {
        self.indices = indices
        self.values = values
        self.dimension = dimension
    }

    /// Converts the sparse vector back to a dense `[Double]` array.
    public func toDense() -> [Double] {
        var dense = [Double](repeating: 0.0, count: dimension)
        for (idx, val) in zip(indices, values) {
            if idx >= 0 && idx < dimension {
                dense[idx] = val
            }
        }
        return dense
    }
}

/// TF-IDF Vectorizer for text feature extraction.
public actor TFIDFVectorizer {
    /// Default stop words list to filter out during tokenization.
    public static let defaultStopWords: Set<String> = [
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to", "in", "on", 
        "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", 
        "before", "after", "above", "below", "from", "up", "down", "is", "are", "was", "were", 
        "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", 
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
        "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", 
        "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", 
        "who", "whom", "this", "that", "these", "those", "am", "as"
    ]
    
    /// Active stop words set.
    public let stopWords: Set<String>

    /// Optional maximum number of vocabulary features to retain (ordered by document frequency).
    public let maxFeatures: Int?
    
    /// Minimum document frequency required for a term to be included in vocabulary.
    public let minDF: Int

    /// Map of word to vocabulary index.
    public private(set) var vocabulary: [String: Int] = [:]
    
    /// Inverse document frequency (IDF) vector. Shape: [vocabSize]
    public private(set) var idfs: [Double] = []
    
    /// Initializes a new TFIDFVectorizer.
    /// - Parameters:
    ///   - maxFeatures: Optional maximum number of features to retain by term frequency.
    ///   - minDF: Minimum document frequency threshold. Defaults to 1.
    ///   - stopWords: Optional custom set of stop words to filter. If nil, `defaultStopWords` is used.
    public init(maxFeatures: Int? = nil, minDF: Int = 1, stopWords: Set<String>? = nil) {
        self.maxFeatures = maxFeatures.map { max(1, $0) }
        self.minDF = max(1, minDF)
        self.stopWords = stopWords ?? Self.defaultStopWords
    }
    
    /// Tokenizes a preprocessed document string into tokens.
    /// - Parameter doc: Input string document.
    /// - Returns: A list of clean tokens.
    private func tokenize(_ doc: String) -> [String] {
        return doc.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && !self.stopWords.contains($0) }
    }
    
    /// Fits the vectorizer on a corpus of documents, building the vocabulary and computing IDFs.
    /// - Parameter documents: List of string documents.
    public func fit(_ documents: [String]) throws {
        guard !documents.isEmpty else {
            throw NLPError.emptyInput
        }
        
        // 1. Tokenize all documents and build document frequencies
        var docTokenSets = [Set<String>]()
        var dfMap = [String: Int]()
        
        for doc in documents {
            let tokens = tokenize(doc)
            let tokenSet = Set(tokens)
            docTokenSets.append(tokenSet)
            for tok in tokenSet {
                dfMap[tok, default: 0] += 1
            }
        }
        
        // Filter terms by minDF
        var validTerms = dfMap.filter { $0.value >= minDF }.keys.map { String($0) }
        
        guard !validTerms.isEmpty else {
            throw NLPError.invalidVocabulary
        }
        
        // If maxFeatures is specified, select top maxFeatures by document frequency
        if let maxF = maxFeatures, validTerms.count > maxF {
            validTerms.sort { t1, t2 in
                let df1 = dfMap[t1] ?? 0
                let df2 = dfMap[t2] ?? 0
                if df1 != df2 {
                    return df1 > df2
                }
                return t1 < t2
            }
            validTerms = Array(validTerms.prefix(maxF))
        }
        
        // Sort selected vocabulary alphabetically for deterministic indexing
        let sortedVocab = validTerms.sorted()
        var vocabMap = [String: Int]()
        for (idx, word) in sortedVocab.enumerated() {
            vocabMap[word] = idx
        }
        
        // 2. Compute IDF for each vocabulary term using smooth IDF: log((1 + N) / (1 + df)) + 1
        let n = Double(documents.count)
        var idfValues = [Double](repeating: 0.0, count: sortedVocab.count)
        
        for (idx, word) in sortedVocab.enumerated() {
            let df = Double(dfMap[word] ?? 0)
            idfValues[idx] = log((1.0 + n) / (1.0 + df)) + 1.0
        }
        
        self.vocabulary = vocabMap
        self.idfs = idfValues
    }
    
    /// Transforms the documents into a dense TF-IDF matrix.
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

    /// Transforms documents into memory-efficient `SparseVector` representations.
    /// - Parameter documents: List of string documents.
    /// - Returns: An array of `SparseVector` objects.
    public func transformSparse(_ documents: [String]) throws -> [SparseVector] {
        guard !vocabulary.isEmpty, !idfs.isEmpty else {
            throw NLPError.fittingRequired
        }
        guard !documents.isEmpty else {
            throw NLPError.emptyInput
        }

        let vocabSize = vocabulary.count
        var sparseVectors = [SparseVector]()
        sparseVectors.reserveCapacity(documents.count)

        for doc in documents {
            let docTokens = tokenize(doc)
            var termCounts = [String: Int]()
            for tok in docTokens {
                termCounts[tok, default: 0] += 1
            }

            var activeIndices = [Int]()
            var activeValues = [Double]()

            if !docTokens.isEmpty {
                // Collect unique terms present in doc
                let uniqueDocTerms = Set(docTokens).sorted()
                for tok in uniqueDocTerms {
                    if let idx = vocabulary[tok], let count = termCounts[tok] {
                        let tf = Double(count) / Double(docTokens.count)
                        let val = tf * idfs[idx]
                        if val > 0.0 {
                            activeIndices.append(idx)
                            activeValues.append(val)
                        }
                    }
                }
            }

            sparseVectors.append(SparseVector(indices: activeIndices, values: activeValues, dimension: vocabSize))
        }

        return sparseVectors
    }
    
    /// Fits the model and transforms the documents into a dense matrix.
    /// - Parameter documents: List of string documents.
    /// - Returns: A 2D array of shape [documents, vocabSize].
    public func fitTransform(_ documents: [String]) throws -> [[Double]] {
        try fit(documents)
        return try transform(documents)
    }

    /// Fits the model and transforms documents into sparse vectors.
    public func fitTransformSparse(_ documents: [String]) throws -> [SparseVector] {
        try fit(documents)
        return try transformSparse(documents)
    }

    /// Fits the vectorizer on a corpus of documents (labeled argument overload).
    public func fit(documents: [String]) throws {
        try fit(documents)
    }

    /// Transforms documents into a TF-IDF matrix (labeled argument overload).
    public func transform(documents: [String]) throws -> [[Double]] {
        try transform(documents)
    }

    /// Transforms documents into sparse vectors (labeled argument overload).
    public func transformSparse(documents: [String]) throws -> [SparseVector] {
        try transformSparse(documents)
    }

    /// Fits the model and transforms documents (labeled argument overload).
    public func fitTransform(documents: [String]) throws -> [[Double]] {
        try fitTransform(documents)
    }
}
