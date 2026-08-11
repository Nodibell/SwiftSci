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
    public func fitTFIDF(column name: String) async throws -> TFIDFVectorizer {
        let documents = try extractDocuments(column: name)
        let vectorizer = TFIDFVectorizer()
        try await vectorizer.fit(documents)
        return vectorizer
    }

    /// Tokenizes a text column using the specified `Tokenizer`.
    public func tokenizeColumn(_ name: String, targetColumn: String = "tokens", tokenizer: any Tokenizer = AppleWordTokenizer()) throws -> DataFrame {
        let docs = try extractDocuments(column: name)
        let tokenizedDocs = docs.map { tokenizer.tokenize(text: $0).joined(separator: " ") }
        return try withColumn(targetColumn, column: TypedColumn<String>(name: targetColumn, values: tokenizedDocs))
    }

    /// Stems word tokens in a text column using `PorterStemmer`.
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
    public func analyzeSentiment(column name: String, targetColumn: String = "sentiment_compound") throws -> DataFrame {
        let docs = try extractDocuments(column: name)
        let analyzer = VADERSentimentAnalyzer()
        let scores = docs.map { analyzer.polarityScores(text: $0).compound }
        return try withColumn(targetColumn, column: TypedColumn<Double>(name: targetColumn, values: scores))
    }

    /// Detects language code on a text column using AppleLanguageDetector.
    public func detectLanguage(column name: String, targetColumn: String = "language") throws -> DataFrame {
        let docs = try extractDocuments(column: name)
        let detector = AppleLanguageDetector()
        let languages: [String?] = docs.map { (try? detector.detectLanguage(text: $0)) ?? nil }
        return try withColumn(targetColumn, column: TypedColumn<String>(name: targetColumn, values: languages))
    }

    /// Extracts named entities from a text column using AppleNamedEntityRecognizer.
    public func extractEntities(fromColumn name: String, targetColumn: String = "entities") throws -> DataFrame {
        let docs = try extractDocuments(column: name)
        let recognizer = AppleNamedEntityRecognizer()
        let extracted: [String] = docs.map { text in
            guard let entities = try? recognizer.extractEntities(from: text) else { return "" }
            return entities.map { "\($0.text)(\($0.category.rawValue))" }.joined(separator: ", ")
        }
        return try withColumn(targetColumn, column: TypedColumn<String>(name: targetColumn, values: extracted))
    }
}
