import Foundation

/// Term frequency tuple recording a vocabulary term and its occurrence frequency.
public struct TermFrequency: Sendable, Codable, Equatable {
    /// The vocabulary term string.
    public let term: String
    /// The absolute occurrence frequency count.
    public let count: Int

    /// Initializes a term frequency entry.
    /// - Parameters:
    ///   - term: The vocabulary term.
    ///   - count: The occurrence count.
    public init(term: String, count: Int) {
        self.term = term
        self.count = count
    }
}

/// Comprehensive column-level text profiling metrics including lexical diversity, character distributions, and stopword pruning.
public struct ColumnTextProfile: Sendable, Codable, Equatable {
    /// The name of the analyzed DataFrame column.
    public let columnName: String
    /// Total number of rows/documents evaluated.
    public let totalDocuments: Int
    /// Number of rows with non-empty text content.
    public let nonEmptyDocuments: Int
    /// Total word tokens analyzed before stopword pruning.
    public let totalWords: Int
    /// Number of distinct (unique) vocabulary terms before pruning.
    public let uniqueWords: Int
    /// Total word tokens remaining after stopword pruning.
    public let totalWordsPruned: Int
    /// Distinct vocabulary terms remaining after stopword pruning.
    public let uniqueWordsPruned: Int
    /// Number of stopword tokens removed during pruning.
    public let stopwordCount: Int
    /// Type-Token Ratio before stopword pruning (uniqueWords / totalWords).
    public let typeTokenRatio: Double
    /// Type-Token Ratio after stopword pruning (uniqueWordsPruned / totalWordsPruned).
    public let prunedTypeTokenRatio: Double
    /// Number of hapax legomena (words appearing exactly once in the corpus).
    public let hapaxLegomenaCount: Int
    /// Fraction of unique vocabulary consisting of hapax legomena (hapaxLegomenaCount / uniqueWords).
    public let hapaxLegomenaRatio: Double
    /// Total character count across all documents.
    public let characterCount: Int
    /// Mean character length per document.
    public let meanDocumentLength: Double
    /// Sample standard deviation of document character lengths.
    public let stdDocumentLength: Double
    /// Minimum document character length.
    public let minDocumentLength: Int
    /// Maximum document character length.
    public let maxDocumentLength: Int
    /// Mean character length per word token.
    public let meanWordLength: Double
    /// Shannon information entropy of the post-pruning term distribution in bits ($-\sum p_i \log_2 p_i$).
    public let shannonEntropy: Double
    /// Top most frequent terms after stopword pruning.
    public let topTerms: [TermFrequency]
    /// Primary language identifier configured for stopword pruning.
    public let language: String

    /// Initializes a column text profile result.
    /// - Parameters:
    ///   - columnName: Name of the text column.
    ///   - totalDocuments: Total rows evaluated.
    ///   - nonEmptyDocuments: Non-empty document count.
    ///   - totalWords: Total word count before pruning.
    ///   - uniqueWords: Unique word count before pruning.
    ///   - totalWordsPruned: Total word count after pruning.
    ///   - uniqueWordsPruned: Unique word count after pruning.
    ///   - stopwordCount: Number of pruned stop words.
    ///   - typeTokenRatio: Type-Token ratio before pruning.
    ///   - prunedTypeTokenRatio: Type-Token ratio after pruning.
    ///   - hapaxLegomenaCount: Count of single-occurrence words.
    ///   - hapaxLegomenaRatio: Ratio of hapax legomena to unique words.
    ///   - characterCount: Total character count.
    ///   - meanDocumentLength: Average characters per document.
    ///   - stdDocumentLength: Standard deviation of document character lengths.
    ///   - minDocumentLength: Minimum document character length.
    ///   - maxDocumentLength: Maximum document character length.
    ///   - meanWordLength: Average characters per word.
    ///   - shannonEntropy: Shannon information entropy in bits.
    ///   - topTerms: Most frequent terms with counts.
    ///   - language: Configured language code string.
    public init(
        columnName: String,
        totalDocuments: Int,
        nonEmptyDocuments: Int,
        totalWords: Int,
        uniqueWords: Int,
        totalWordsPruned: Int,
        uniqueWordsPruned: Int,
        stopwordCount: Int,
        typeTokenRatio: Double,
        prunedTypeTokenRatio: Double,
        hapaxLegomenaCount: Int,
        hapaxLegomenaRatio: Double,
        characterCount: Int,
        meanDocumentLength: Double,
        stdDocumentLength: Double,
        minDocumentLength: Int,
        maxDocumentLength: Int,
        meanWordLength: Double,
        shannonEntropy: Double,
        topTerms: [TermFrequency],
        language: String
    ) {
        self.columnName = columnName
        self.totalDocuments = totalDocuments
        self.nonEmptyDocuments = nonEmptyDocuments
        self.totalWords = totalWords
        self.uniqueWords = uniqueWords
        self.totalWordsPruned = totalWordsPruned
        self.uniqueWordsPruned = uniqueWordsPruned
        self.stopwordCount = stopwordCount
        self.typeTokenRatio = typeTokenRatio
        self.prunedTypeTokenRatio = prunedTypeTokenRatio
        self.hapaxLegomenaCount = hapaxLegomenaCount
        self.hapaxLegomenaRatio = hapaxLegomenaRatio
        self.characterCount = characterCount
        self.meanDocumentLength = meanDocumentLength
        self.stdDocumentLength = stdDocumentLength
        self.minDocumentLength = minDocumentLength
        self.maxDocumentLength = maxDocumentLength
        self.meanWordLength = meanWordLength
        self.shannonEntropy = shannonEntropy
        self.topTerms = topTerms
        self.language = language
    }
}
