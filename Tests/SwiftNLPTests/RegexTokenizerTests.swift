import Testing
import Foundation
@testable import SwiftNLP

@Suite("RegexTokenizer Tests")
struct RegexTokenizerTests {

    @Test("RegexTokenizer encodes text into deterministic FNV-1a token IDs")
    func testRegexTokenizerDeterministicEncoding() {
        let tokenizer = RegexTokenizer()
        let text = "SwiftSci deterministic token hashing test."

        let enc1 = tokenizer.encode(text: text)
        let enc2 = tokenizer.encode(text: text)
        #expect(enc1 == enc2)
        #expect(!enc1.isEmpty)

        // Verify IDs match fnv1aTokenHash directly
        let words = tokenizer.tokenize(text: text)
        #expect(enc1 == words.map { fnv1aTokenHash($0) })
    }

    @Test("RegexTokenizer validates regex pattern and throws on invalid syntax")
    func testRegexTokenizerPatternValidation() {
        // Invalid regex syntax (unclosed character class)
        #expect(throws: NLPError.self) {
            _ = try RegexTokenizer(validatingPattern: "[a-z")
        }

        let invalidTokenizer = RegexTokenizer(pattern: "[a-z")
        #expect(throws: NLPError.self) {
            _ = try invalidTokenizer.tokenizeThrowing(text: "hello")
        }

        // Valid tokenizer tokenizeThrowing succeeds
        let validTokenizer = try? RegexTokenizer(validatingPattern: "\\w+")
        #expect(validTokenizer != nil)
        let tokens = try? validTokenizer?.tokenizeThrowing(text: "hello world")
        #expect(tokens == ["hello", "world"])
    }
}
