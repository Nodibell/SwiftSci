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

    @Test("TF-IDF fitting on DataFrame")
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

    @Test("TF-IDF maxFeatures limits vocabulary size")
    func testMaxFeatures() async throws {
        // 4 unique terms across docs, maxFeatures=2 keeps top 2 by doc frequency
        let corpus = [
            "cat dog bird",
            "cat dog fish",
            "cat lion"
        ]
        let vectorizer = TFIDFVectorizer(maxFeatures: 2)
        try await vectorizer.fit(corpus)
        let vocab = await vectorizer.vocabulary
        // "cat" appears in 3 docs, "dog" in 2 → top 2
        #expect(vocab.count == 2)
        #expect(vocab["cat"] != nil)
        #expect(vocab["dog"] != nil)
    }

    @Test("TF-IDF minDF filters rare terms")
    func testMinDF() async throws {
        // "rare" appears in only 1 doc; minDF=2 should exclude it
        let corpus = [
            "cat dog rare",
            "cat dog",
            "cat dog"
        ]
        let vectorizer = TFIDFVectorizer(minDF: 2)
        try await vectorizer.fit(corpus)
        let vocab = await vectorizer.vocabulary
        #expect(vocab["rare"] == nil)
        #expect(vocab["cat"] != nil)
        #expect(vocab["dog"] != nil)
    }

    @Test("TF-IDF fitTransform labeled overload produces same result as fit+transform")
    func testFitTransformLabeledOverload() async throws {
        let corpus = ["hello world", "world swift"]
        let v1 = TFIDFVectorizer()
        let matrix1 = try await v1.fitTransform(documents: corpus)

        let v2 = TFIDFVectorizer()
        try await v2.fit(documents: corpus)
        let matrix2 = try await v2.transform(documents: corpus)

        #expect(matrix1.count == matrix2.count)
        for (row1, row2) in zip(matrix1, matrix2) {
            for (v1, v2) in zip(row1, row2) {
                #expect(abs(v1 - v2) < 1e-9)
            }
        }
    }
}
