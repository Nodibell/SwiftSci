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
    /// - Returns: Array of computed numeric values.
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
/// TF-IDF Vectorizer for text feature extraction.
public final class TFIDFVectorizer: @unchecked Sendable {
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

    /// N-gram range for feature extraction (e.g. 1...1 for unigrams, 1...2 for unigrams + bigrams).
    public let ngramRange: ClosedRange<Int>

    /// Whether to apply sublinear TF scaling (1 + log(count)).
    public let sublinearTF: Bool

    /// Optional maximum document frequency threshold ratio (e.g. 0.85).
    public let maxDF: Double?

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
    ///   - ngramRange: Range of n-gram sizes to extract. Defaults to 1...1 (unigrams only).
    ///   - maxFeatures: Optional maximum number of features to retain by term frequency.
    ///   - minDF: Minimum document frequency threshold. Defaults to 1.
    ///   - maxDF: Optional maximum document frequency ratio (0.0 to 1.0). Terms exceeding this frequency are excluded.
    ///   - sublinearTF: Whether to apply sublinear term frequency scaling: `1 + log(tf)`. Defaults to false.
    ///   - language: Optional language identifier for built-in stop words.
    ///   - removeStopWords: Whether to prune stop words during tokenization. Defaults to true.
    ///   - customStopWords: Optional additional stop words to filter.
    ///   - stopWords: Optional custom set of stop words to filter. If nil and `removeStopWords` is true, defaults based on language.
    public init(
        ngramRange: ClosedRange<Int> = 1...1,
        maxFeatures: Int? = nil,
        minDF: Int = 1,
        maxDF: Double? = nil,
        sublinearTF: Bool = false,
        language: StopWords.Language? = nil,
        removeStopWords: Bool = true,
        customStopWords: Set<String>? = nil,
        stopWords: Set<String>? = nil
    ) {
        let lower = max(1, ngramRange.lowerBound)
        let upper = max(lower, ngramRange.upperBound)
        self.ngramRange = lower...upper
        self.maxFeatures = maxFeatures.map { max(1, $0) }
        self.minDF = max(1, minDF)
        self.maxDF = maxDF
        self.sublinearTF = sublinearTF
        if let custom = stopWords {
            self.stopWords = custom
        } else if !removeStopWords {
            self.stopWords = []
        } else if let lang = language {
            var stops = StopWords.set(for: lang)
            if let custom = customStopWords {
                stops.formUnion(custom)
            }
            self.stopWords = stops
        } else {
            self.stopWords = Self.defaultStopWords
        }
    }
    
    /// Tokenizes a preprocessed document string into tokens (unigrams and n-grams if configured).
    /// - Parameter doc: Input string document.
    /// - Returns: A list of clean tokens including n-grams.
    private func tokenize(_ doc: String) -> [String] {
        var unigrams: [String] = []
        var current = ""
        current.reserveCapacity(16)
        for char in doc.lowercased() {
            if char.isLetter || char.isNumber {
                current.append(char)
            } else if !current.isEmpty {
                if !self.stopWords.contains(current) {
                    unigrams.append(current)
                }
                current.removeAll(keepingCapacity: true)
            }
        }
        if !current.isEmpty && !self.stopWords.contains(current) {
            unigrams.append(current)
        }
        
        if ngramRange == 1...1 {
            return unigrams
        }
        
        var tokens: [String] = []
        let minN = ngramRange.lowerBound
        let maxN = ngramRange.upperBound
        let count = unigrams.count
        
        for n in minN...maxN {
            if n == 1 {
                tokens.append(contentsOf: unigrams)
            } else if n <= count {
                for i in 0...(count - n) {
                    tokens.append(unigrams[i..<(i + n)].joined(separator: " "))
                }
            }
        }
        return tokens
    }

    private func buildVocabulary(dfMap: [String: Int], numDocs: Int) throws {
        let maxDocLimit = maxDF.map { Int(Double(numDocs) * $0) }
        var validTerms = dfMap.filter { (term, df) in
            if df < minDF { return false }
            if let maxLimit = maxDocLimit, df > maxLimit { return false }
            return true
        }.keys.map { String($0) }
        guard !validTerms.isEmpty else {
            throw NLPError.invalidVocabulary
        }

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

        let sortedVocab = validTerms.sorted()
        var vocabMap = [String: Int]()
        vocabMap.reserveCapacity(sortedVocab.count)
        for (idx, word) in sortedVocab.enumerated() {
            vocabMap[word] = idx
        }

        let n = Double(numDocs)
        var idfValues = [Double](repeating: 0.0, count: sortedVocab.count)
        for (idx, word) in sortedVocab.enumerated() {
            let df = Double(dfMap[word] ?? 0)
            idfValues[idx] = log((1.0 + n) / (1.0 + df)) + 1.0
        }

        self.vocabulary = vocabMap
        self.idfs = idfValues
    }

    private func transformTokenizedDocs(_ docTokensList: [[String]]) -> [[Double]] {
        let vocabSize = vocabulary.count
        var result = [[Double]]()
        result.reserveCapacity(docTokensList.count)

        for docTokens in docTokensList {
            var termCounts = [String: Int]()
            termCounts.reserveCapacity(docTokens.count)
            for tok in docTokens {
                termCounts[tok, default: 0] += 1
            }

            var vector = [Double](repeating: 0.0, count: vocabSize)
            if !docTokens.isEmpty {
                let docCount = Double(docTokens.count)
                for (tok, count) in termCounts {
                    if let idx = vocabulary[tok] {
                        let tf: Double
                        if sublinearTF {
                            tf = 1.0 + log(Double(count))
                        } else {
                            tf = Double(count) / docCount
                        }
                        vector[idx] = tf * idfs[idx]
                    }
                }
            }
            result.append(vector)
        }
        return result
    }

    private func transformSparseTokenizedDocs(_ docTokensList: [[String]]) -> [SparseVector] {
        let vocabSize = vocabulary.count
        var sparseVectors = [SparseVector]()
        sparseVectors.reserveCapacity(docTokensList.count)

        for docTokens in docTokensList {
            var termCounts = [String: Int]()
            termCounts.reserveCapacity(docTokens.count)
            for tok in docTokens {
                termCounts[tok, default: 0] += 1
            }

            var activeIndices = [Int]()
            var activeValues = [Double]()

            if !docTokens.isEmpty {
                let docCount = Double(docTokens.count)
                let uniqueDocTerms = termCounts.keys.sorted()
                for tok in uniqueDocTerms {
                    if let idx = vocabulary[tok], let count = termCounts[tok] {
                        let tf: Double
                        if sublinearTF {
                            tf = 1.0 + log(Double(count))
                        } else {
                            tf = Double(count) / docCount
                        }
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
    
    /// Fits the vectorizer on a corpus of documents, building the vocabulary and computing IDFs.
    /// - Parameter documents: List of string documents.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    public func fit(_ documents: [String]) throws {
        guard !documents.isEmpty else {
            throw NLPError.emptyInput
        }
        
        var dfMap = [String: Int]()
        for doc in documents {
            let tokens = tokenize(doc)
            let tokenSet = Set(tokens)
            for tok in tokenSet {
                dfMap[tok, default: 0] += 1
            }
        }
        
        try buildVocabulary(dfMap: dfMap, numDocs: documents.count)
    }
    
    /// Transforms the documents into a dense TF-IDF matrix.
    /// - Parameter documents: List of string documents.
    /// - Returns: A 2D array of shape [documents, vocabSize].
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    public func transform(_ documents: [String]) throws -> [[Double]] {
        guard !vocabulary.isEmpty, !idfs.isEmpty else {
            throw NLPError.fittingRequired
        }
        guard !documents.isEmpty else {
            throw NLPError.emptyInput
        }
        let docTokensList = documents.map { tokenize($0) }
        return transformTokenizedDocs(docTokensList)
    }

    /// Transforms documents into memory-efficient `SparseVector` representations.
    /// - Parameter documents: List of string documents.
    /// - Returns: An array of `SparseVector` objects.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    public func transformSparse(_ documents: [String]) throws -> [SparseVector] {
        guard !vocabulary.isEmpty, !idfs.isEmpty else {
            throw NLPError.fittingRequired
        }
        guard !documents.isEmpty else {
            throw NLPError.emptyInput
        }
        let docTokensList = documents.map { tokenize($0) }
        return transformSparseTokenizedDocs(docTokensList)
    }
    
    /// Fits the model and transforms the documents into a dense matrix.
    /// - Parameter documents: List of string documents.
    /// - Returns: A 2D array of shape [documents, vocabSize].
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    public func fitTransform(_ documents: [String]) throws -> [[Double]] {
        guard !documents.isEmpty else { throw NLPError.emptyInput }
        var allTokens: [[String]] = []
        allTokens.reserveCapacity(documents.count)
        var dfMap = [String: Int]()

        for doc in documents {
            let tokens = tokenize(doc)
            allTokens.append(tokens)
            let tokenSet = Set(tokens)
            for tok in tokenSet {
                dfMap[tok, default: 0] += 1
            }
        }

        try buildVocabulary(dfMap: dfMap, numDocs: documents.count)
        return transformTokenizedDocs(allTokens)
    }

    /// Fits the model and transforms documents into sparse vectors.
    /// - Parameters:
    ///   - documents: Collection of text documents to process.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    /// - Returns: The computed [SparseVector] result instance.
    public func fitTransformSparse(_ documents: [String]) throws -> [SparseVector] {
        guard !documents.isEmpty else { throw NLPError.emptyInput }
        var allTokens: [[String]] = []
        allTokens.reserveCapacity(documents.count)
        var dfMap = [String: Int]()

        for doc in documents {
            let tokens = tokenize(doc)
            allTokens.append(tokens)
            let tokenSet = Set(tokens)
            for tok in tokenSet {
                dfMap[tok, default: 0] += 1
            }
        }

        try buildVocabulary(dfMap: dfMap, numDocs: documents.count)
        return transformSparseTokenizedDocs(allTokens)
    }

    /// Fits the vectorizer on a corpus of documents (labeled argument overload).
    /// - Parameters:
    ///   - documents: Collection of text documents to process.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    public func fit(documents: [String]) throws {
        try fit(documents)
    }

    /// Transforms documents into a TF-IDF matrix (labeled argument overload).
    /// - Parameters:
    ///   - documents: Collection of text documents to process.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    /// - Returns: 2D numerical matrix of shape `[N, P]`.
    public func transform(documents: [String]) throws -> [[Double]] {
        try transform(documents)
    }

    /// Transforms documents into sparse vectors (labeled argument overload).
    /// - Parameters:
    ///   - documents: Collection of text documents to process.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    /// - Returns: The computed [SparseVector] result instance.
    public func transformSparse(documents: [String]) throws -> [SparseVector] {
        try transformSparse(documents)
    }

    /// Fits the model and transforms documents (labeled argument overload).
    /// - Parameters:
    ///   - documents: Collection of text documents to process.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    /// - Returns: 2D numerical matrix of shape `[N, P]`.
    public func fitTransform(documents: [String]) throws -> [[Double]] {
        try fitTransform(documents)
    }
}
