import Foundation
import SwiftDataFrame

extension DataFrame {
    private func extractDocuments(column name: String) throws -> [String] {
        guard let col = self[column: name, as: String.self] else {
            throw SwiftMLError.columnNotFound(name)
        }
        return col.values.map { $0 ?? "" }
    }

    /// Fits a TFIDFVectorizer on the specified text column.
    /// - Parameters:
    ///   - name: Name or identifier string.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    /// - Returns: The computed TFIDFVectorizer result instance.
    public func fitTFIDF(column name: String) async throws -> TFIDFVectorizer {
        let documents = try extractDocuments(column: name)
        let vectorizer = TFIDFVectorizer()
        try await vectorizer.fit(documents)
        return vectorizer
    }

    /// Tokenizes a text column using the specified `Tokenizer`.
    /// - Parameters:
    ///   - name: Name or identifier string.
    ///   - targetColumn: Name of the target column in the dataset.
    ///   - tokenizer: Tokenizer instance or strategy used for lexical decomposition.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    /// - Returns: A new `DataFrame` containing the transformed columns and computed results.
    public func tokenizeColumn(_ name: String, targetColumn: String = "tokens", tokenizer: any Tokenizer = AppleWordTokenizer()) throws -> DataFrame {
        let docs = try extractDocuments(column: name)
        let tokenizedDocs = docs.map { tokenizer.tokenize(text: $0).joined(separator: " ") }
        return try withColumn(targetColumn, column: TypedColumn<String>(name: targetColumn, values: tokenizedDocs))
    }

    /// Stems word tokens in a text column using `PorterStemmer`.
    /// - Parameters:
    ///   - name: Name or identifier string.
    ///   - targetColumn: Name of the target column in the dataset.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    /// - Returns: A new `DataFrame` containing the transformed columns and computed results.
    public func stemColumn(_ name: String, targetColumn: String = "stemmed") throws -> DataFrame {
        let docs = try extractDocuments(column: name)
        let stemmer = PorterStemmer()
        let stemmedDocs = docs.map { text in
            let tokens = text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
            return stemmer.stem(tokens: tokens).joined(separator: " ")
        }
        return try withColumn(targetColumn, column: TypedColumn<String>(name: targetColumn, values: stemmedDocs))
    }

    /// Evaluates sentiment on a text column using VADER sentiment analyzer.
    /// - Parameters:
    ///   - name: Name or identifier string.
    ///   - targetColumn: Name of the target column in the dataset.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    /// - Returns: A new `DataFrame` containing the transformed columns and computed results.
    public func analyzeSentiment(column name: String, targetColumn: String = "sentiment_compound") throws -> DataFrame {
        let docs = try extractDocuments(column: name)
        let analyzer = VADERSentimentAnalyzer()
        let scores = docs.map { analyzer.polarityScores(text: $0).compound }
        return try withColumn(targetColumn, column: TypedColumn<Double>(name: targetColumn, values: scores))
    }

    /// Detects language code on a text column using AppleLanguageDetector.
    /// - Parameters:
    ///   - name: Name or identifier string.
    ///   - targetColumn: Name of the target column in the dataset.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    /// - Returns: A new `DataFrame` containing the transformed columns and computed results.
    public func detectLanguage(column name: String, targetColumn: String = "language") throws -> DataFrame {
        let docs = try extractDocuments(column: name)
        let detector = AppleLanguageDetector()
        let languages: [String?] = docs.map { (try? detector.detectLanguage(text: $0)) ?? nil }
        return try withColumn(targetColumn, column: TypedColumn<String>(name: targetColumn, values: languages))
    }

    /// Extracts named entities from a text column using AppleNamedEntityRecognizer.
    /// - Parameters:
    ///   - name: Name or identifier string.
    ///   - targetColumn: Name of the target column in the dataset.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    /// - Returns: A new `DataFrame` containing the transformed columns and computed results.
    public func extractEntities(fromColumn name: String, targetColumn: String = "entities") throws -> DataFrame {
        let docs = try extractDocuments(column: name)
        let recognizer = AppleNamedEntityRecognizer()
        let extracted: [String] = docs.map { text in
            guard let entities = try? recognizer.extractEntities(from: text) else { return "" }
            return entities.map { "\($0.text)(\($0.category.rawValue))" }.joined(separator: ", ")
        }
        return try withColumn(targetColumn, column: TypedColumn<String>(name: targetColumn, values: extracted))
    }

    /// Profiles a text column by computing comprehensive lexical statistics, character distributions, and pruned vocabulary metrics.
    /// - Parameters:
    ///   - name: The name of the text column to analyze.
    ///   - language: The stop-words language dictionary to apply (default: `.english`).
    ///   - customStopWords: Optional additional stop words to filter out during profiling.
    ///   - topKTerms: Maximum number of high-frequency vocabulary terms to retain in `topTerms` (default: 10).
    ///   - tokenizer: Tokenizer implementation used to segment text (defaults to `AppleWordTokenizer`).
    /// - Throws: `SwiftMLError.columnNotFound` if column does not exist or cannot be cast to String.
    /// - Returns: A `ColumnTextProfile` instance containing lexical, character, and term frequency statistics.
    public func profileTextColumn(
        _ name: String,
        language: StopWords.Language = .english,
        customStopWords: Set<String>? = nil,
        topKTerms: Int = 10,
        tokenizer: (any Tokenizer)? = nil
    ) throws -> ColumnTextProfile {
        let docs = try extractDocuments(column: name)
        guard !docs.isEmpty else {
            return ColumnTextProfile(
                columnName: name,
                totalDocuments: 0,
                nonEmptyDocuments: 0,
                totalWords: 0,
                uniqueWords: 0,
                totalWordsPruned: 0,
                uniqueWordsPruned: 0,
                stopwordCount: 0,
                typeTokenRatio: 0.0,
                prunedTypeTokenRatio: 0.0,
                hapaxLegomenaCount: 0,
                hapaxLegomenaRatio: 0.0,
                characterCount: 0,
                meanDocumentLength: 0.0,
                stdDocumentLength: 0.0,
                minDocumentLength: 0,
                maxDocumentLength: 0,
                meanWordLength: 0.0,
                shannonEntropy: 0.0,
                topTerms: [],
                language: language.rawValue
            )
        }

        var stopSet = StopWords.set(for: language)
        if let custom = customStopWords {
            stopSet.formUnion(custom.map { $0.lowercased() })
        }

        let activeTokenizer = tokenizer ?? AppleWordTokenizer()

        var totalChars = 0
        var minDocLen = Int.max
        var maxDocLen = 0
        var nonEmptyDocs = 0
        var docLengths: [Double] = []
        docLengths.reserveCapacity(docs.count)

        var totalWords = 0
        var totalWordsPruned = 0
        var stopwordCount = 0
        var totalWordChars = 0

        var rawFrequencies: [String: Int] = [:]
        var prunedFrequencies: [String: Int] = [:]

        for doc in docs {
            let docLen = doc.count
            docLengths.append(Double(docLen))
            totalChars += docLen
            if docLen > 0 {
                nonEmptyDocs += 1
            }
            if docLen < minDocLen {
                minDocLen = docLen
            }
            if docLen > maxDocLen {
                maxDocLen = docLen
            }

            let tokens = activeTokenizer.tokenize(text: doc)
            totalWords += tokens.count

            for tok in tokens {
                let lower = tok.lowercased()
                guard !lower.isEmpty else { continue }
                totalWordChars += tok.count
                rawFrequencies[lower, default: 0] += 1

                if stopSet.contains(lower) {
                    stopwordCount += 1
                } else {
                    prunedFrequencies[lower, default: 0] += 1
                    totalWordsPruned += 1
                }
            }
        }

        let nDocsDouble = Double(docs.count)
        let meanDocLen = Double(totalChars) / nDocsDouble

        // Calculate sample standard deviation of document lengths with 64-bit precision
        var varianceSum = 0.0
        for len in docLengths {
            let diff = len - meanDocLen
            varianceSum += diff * diff
        }
        let stdDocLen = docs.count > 1 ? sqrt(varianceSum / Double(docs.count - 1)) : 0.0

        let uniqueWords = rawFrequencies.count
        let uniqueWordsPruned = prunedFrequencies.count

        let ttr = totalWords > 0 ? (Double(uniqueWords) / Double(totalWords)) : 0.0
        let prunedTTR = totalWordsPruned > 0 ? (Double(uniqueWordsPruned) / Double(totalWordsPruned)) : 0.0

        var hapaxCount = 0
        for (_, count) in rawFrequencies {
            if count == 1 {
                hapaxCount += 1
            }
        }
        let hapaxRatio = uniqueWords > 0 ? (Double(hapaxCount) / Double(uniqueWords)) : 0.0
        let meanWordLen = totalWords > 0 ? (Double(totalWordChars) / Double(totalWords)) : 0.0

        // Compute Shannon entropy in bits (- sum p * log2(p)) over post-pruning term distribution
        var entropy = 0.0
        let log2Constant = log(2.0)
        let totalPrunedDouble = Double(totalWordsPruned)
        if totalWordsPruned > 0 {
            for (_, count) in prunedFrequencies {
                let p = Double(count) / totalPrunedDouble
                entropy -= p * (log(p) / log2Constant)
            }
        }

        // Top-K terms sorted descending by frequency, then alphabetically
        let sortedPruned = prunedFrequencies.sorted {
            if $0.value != $1.value {
                return $0.value > $1.value
            }
            return $0.key < $1.key
        }
        let topCount = max(0, topKTerms)
        let topTerms = sortedPruned.prefix(topCount).map {
            TermFrequency(term: $0.key, count: $0.value)
        }

        return ColumnTextProfile(
            columnName: name,
            totalDocuments: docs.count,
            nonEmptyDocuments: nonEmptyDocs,
            totalWords: totalWords,
            uniqueWords: uniqueWords,
            totalWordsPruned: totalWordsPruned,
            uniqueWordsPruned: uniqueWordsPruned,
            stopwordCount: stopwordCount,
            typeTokenRatio: ttr,
            prunedTypeTokenRatio: prunedTTR,
            hapaxLegomenaCount: hapaxCount,
            hapaxLegomenaRatio: hapaxRatio,
            characterCount: totalChars,
            meanDocumentLength: meanDocLen,
            stdDocumentLength: stdDocLen,
            minDocumentLength: minDocLen == Int.max ? 0 : minDocLen,
            maxDocumentLength: maxDocLen,
            meanWordLength: meanWordLen,
            shannonEntropy: entropy,
            topTerms: topTerms,
            language: language.rawValue
        )
    }
}
