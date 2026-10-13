import Testing
import Foundation
@testable import SwiftNLP

@Suite("SentenceTokenizer Tests")
struct SentenceTokenizerTests {

    @Test("SentenceTokenizer encodes sentences into deterministic FNV-1a token IDs")
    func testSentenceTokenizerDeterministicEncoding() {
        let tokenizer = SentenceTokenizer()
        let text = "First sentence here. Second sentence starts now! Third sentence?"

        let enc1 = tokenizer.encode(text: text)
        let enc2 = tokenizer.encode(text: text)
        #expect(enc1 == enc2)
        #expect(!enc1.isEmpty)

        let sentences = tokenizer.tokenize(text: text)
        #expect(enc1 == sentences.map { fnv1aTokenHash($0) })
    }
}
