import Testing
import Foundation
import SwiftDataFrame
@testable import SwiftNLP

@Suite("TFIDF Tests")
struct TFIDFTests {
    
    @Test("TF-IDF basic tokenization and calculation")
    func testTFIDFBasic() async throws {
        let corpus = [
            "The quick brown fox",
            "jumped over the lazy dog"
        ]
        
        let vectorizer = TFIDFVectorizer()
        try await vectorizer.fit(corpus)
        
        // "the" is a stop word, so it should be filtered out!
        // Remaining words:
        // Doc 0: quick, brown, fox
        // Doc 1: jumped, over, lazy, dog
        let vocab = await vectorizer.vocabulary
        #expect(vocab.count == 7)
        #expect(vocab["quick"] != nil)
        #expect(vocab["lazy"] != nil)
        #expect(vocab["the"] == nil) // verify stop word removal
        
        let tfidf = try await vectorizer.transform(corpus)
        #expect(tfidf.count == 2)
        #expect(tfidf[0].count == 7)
        
        // In doc 0, words "quick", "brown", "fox" should have positive values, others 0
        let quickIdx = vocab["quick"]!
        let lazyIdx = vocab["lazy"]!
        
        #expect(tfidf[0][quickIdx] > 0.0)
        #expect(tfidf[0][lazyIdx] == 0.0)
        
        #expect(tfidf[1][quickIdx] == 0.0)
        #expect(tfidf[1][lazyIdx] > 0.0)
    }
    
    @Test("TF-IDF fit and transform errors")
    func testTFIDFErrorHandling() async throws {
        let vectorizer = TFIDFVectorizer()
        
        // Empty corpus
        await #expect(throws: NLPError.self) {
            try await vectorizer.fit([])
        }
        
        // Transform before fit
        await #expect(throws: NLPError.self) {
            _ = try await vectorizer.transform(["hello world"])
        }
        
        // Corpus with only stop words (resulting in empty vocabulary)
        await #expect(throws: NLPError.self) {
            try await vectorizer.fit(["the and of", "a an above"])
        }
    }

    @Test("TFIDF fitting on DataFrame")
    func testNLPGlue() async throws {
        let text = TypedColumn<String>(name: "text", values: [
            "The quick brown fox",
            "jumped over the lazy dog"
        ])
        let df = try DataFrame(columns: [text])

        // TFIDF
        let tfidf = try await df.fitTFIDF(column: "text")
        let vocab = await tfidf.vocabulary
        #expect(vocab.count == 7)
    }
}
