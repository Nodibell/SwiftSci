import Testing
import Foundation
import SwiftDataFrame
@testable import SwiftNLP

@Suite("DataFrame+NLP Extension Tests")
struct DataFrameNLPExtensionTests {

    @Test("fitTFIDF on text column returns fitted vectorizer")
    func testFitTFIDF() async throws {
        let df = try DataFrame(columns: [
            TypedColumn(name: "text", values: ["swift machine learning", "artificial intelligence in swift", "data science"])
        ])
        let tfidf = try await df.fitTFIDF(column: "text")
        let vec = try await tfidf.transform(["swift data"])
        #expect(!vec.isEmpty)
    }

    @Test("tokenizeColumn tokenizes text into target column")
    func testTokenizeColumn() throws {
        let df = try DataFrame(columns: [
            TypedColumn(name: "text", values: ["Hello, world!", "Swift science"])
        ])
        let transformed = try df.tokenizeColumn("text", targetColumn: "tokens")
        #expect(transformed.columnNames.contains("tokens"))
        let tokens = transformed[column: "tokens", as: String.self]?.values ?? []
        #expect(tokens.count == 2)
    }

    @Test("stemColumn applies Porter stemming")
    func testStemColumn() throws {
        let df = try DataFrame(columns: [
            TypedColumn(name: "words", values: ["connecting connection connections", "playing played player"])
        ])
        let stemmedDF = try df.stemColumn("words", targetColumn: "stemmed")
        #expect(stemmedDF.columnNames.contains("stemmed"))
        let stemmedValues = stemmedDF[column: "stemmed", as: String.self]?.values ?? []
        #expect(stemmedValues.count == 2)
    }

    @Test("analyzeSentiment calculates compound scores")
    func testAnalyzeSentiment() throws {
        let df = try DataFrame(columns: [
            TypedColumn(name: "review", values: ["I love this fantastic library!", "Terrible awful broken", "This is a book"])
        ])
        let scored = try df.analyzeSentiment(column: "review", targetColumn: "sentiment")
        #expect(scored.columnNames.contains("sentiment"))
        let scores = scored[column: "sentiment", as: Double.self]?.values ?? []
        #expect(scores.count == 3)
        if let pos = scores[0], let neg = scores[1] {
            #expect(pos > neg)
        }
    }

    @Test("detectLanguage detects language code")
    func testDetectLanguage() throws {
        let df = try DataFrame(columns: [
            TypedColumn(name: "text", values: ["This is an English text.", "Це український текст."])
        ])
        let detected = try df.detectLanguage(column: "text", targetColumn: "lang")
        #expect(detected.columnNames.contains("lang"))
        let langs = detected[column: "lang", as: String.self]?.values ?? []
        #expect(langs.count == 2)
    }

    @Test("extractEntities extracts named entities")
    func testExtractEntities() throws {
        let df = try DataFrame(columns: [
            TypedColumn(name: "news", values: ["Apple was founded in Cupertino, California.", "Tim Cook visited London."])
        ])
        let entitiesDF = try df.extractEntities(fromColumn: "news", targetColumn: "ner")
        #expect(entitiesDF.columnNames.contains("ner"))
        let ner = entitiesDF[column: "ner", as: String.self]?.values ?? []
        #expect(ner.count == 2)
    }

    @Test("Calling NLP operations on missing column throws columnNotFound")
    func testMissingColumnThrows() throws {
        let df = try DataFrame(columns: [
            TypedColumn(name: "id", values: [1, 2, 3])
        ])
        #expect(throws: SwiftMLError.self) {
            _ = try df.tokenizeColumn("nonexistent")
        }
        #expect(throws: SwiftMLError.self) {
            _ = try df.stemColumn("nonexistent")
        }
        #expect(throws: SwiftMLError.self) {
            _ = try df.analyzeSentiment(column: "nonexistent")
        }
    }
}
