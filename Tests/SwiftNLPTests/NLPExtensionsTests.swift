import Testing
import Foundation
@testable import SwiftNLP
import SwiftDataFrame

@Suite("SwiftNLP Ecosystem Extensions Suite")
struct NLPExtensionsTests {

    @Test("TextPipeline - End-to-End Fitting and Prediction")
    func testTextPipeline() async throws {
        let pipeline = TextPipeline(alpha: 1.0)
        let docs = [
            "soccer match goal score penalty stadium",
            "football player tournament champion world cup",
            "python swift code algorithm function compiler",
            "binary tree sorting matrix GPU optimization"
        ]
        let labels = ["sports", "sports", "tech", "tech"]

        try await pipeline.fit(documents: docs, labels: labels)

        let sportsPred = try await pipeline.predict(document: "goal penalty football match")
        #expect(sportsPred == "sports")

        let techPred = try await pipeline.predict(document: "swift code function compiler")
        #expect(techPred == "tech")
    }

    @Test("LLM Context Window - Token Counting and Truncation")
    func testLLMContextWindow() {
        let contextWindow = LLMContextWindow(maxTokens: 5, tokenizer: AppleWordTokenizer())
        let longText = "SwiftSci 2.4.0 is a highly performant framework for data science and machine learning."
        
        let tokenCount = contextWindow.countTokens(in: longText)
        #expect(tokenCount > 5)

        let truncated = contextWindow.truncate(text: longText)
        let truncatedTokens = contextWindow.countTokens(in: truncated)
        #expect(truncatedTokens <= 5)
    }

    @Test("TextExplainer - Sentiment Token Importance")
    func testTextExplainer() {
        let explainer = TextExplainer()
        let text = "SwiftSci is fantastic and awesome, but the bug was terrible"
        let importance = explainer.explainSentimentTokens(in: text)
        
        #expect(!importance.isEmpty)
        let topToken = importance.first?.token.lowercased()
        #expect(topToken != nil)
    }

    @Test("DataFrame Text Vectorization for Clustering")
    func testDataFrameVectorizeTextColumn() async throws {
        let df = try DataFrame(columns: [
            TypedColumn<String>(name: "text", values: [
                "data science machine learning AI",
                "quantum physics astronomy cosmology"
            ])
        ])

        let (matrix, vocab) = try await df.vectorizeTextColumn(column: "text")
        #expect(matrix.count == 2)
        #expect(!vocab.isEmpty)
    }
}
