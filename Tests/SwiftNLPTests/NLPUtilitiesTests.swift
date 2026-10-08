import Testing
import Foundation
@testable import SwiftNLP

@Suite("SwiftNLP Utilities Tests")
struct NLPUtilitiesTests {
    @Test("StopWords and TextNormalizer NLP utilities")
    func testNLPUtilities() {
        let tokens = ["The", "quick", "brown", "fox", "and", "the", "lazy", "dog"]
        let filtered = StopWords.filter(tokens: tokens)
        #expect(!filtered.contains("The"))
        #expect(!filtered.contains("and"))
        #expect(filtered.contains("quick"))

        let normalizer = TextNormalizer(lowercase: true, removePunctuation: true)
        let text = "Hello, World! SwiftSci v1.6."
        let normalized = normalizer.normalize(text)
        #expect(normalized == "hello world swiftsci v16")
    }
}
