import Testing
import Foundation
@testable import SwiftNLP

// MARK: - NGramTokenizer Tests

@Suite("NGramTokenizer Tests")
struct NGramTokenizerTests {

    @Test("Default init produces unigrams and bigrams")
    func testDefaultInitParams() {
        let tokenizer = NGramTokenizer()
        #expect(tokenizer.minN == 1)
        #expect(tokenizer.maxN == 2)
        #expect(tokenizer.lowercase == true)
    }

    @Test("Custom init params are stored correctly")
    func testCustomInit() {
        let tokenizer = NGramTokenizer(minN: 2, maxN: 3, lowercase: false)
        #expect(tokenizer.minN == 2)
        #expect(tokenizer.maxN == 3)
        #expect(tokenizer.lowercase == false)
    }

    @Test("minN clamped to 1 when provided 0")
    func testMinNClampedToOne() {
        let tokenizer = NGramTokenizer(minN: 0, maxN: 1)
        #expect(tokenizer.minN == 1)
    }

    @Test("maxN always >= minN")
    func testMaxNAtLeastMinN() {
        let tokenizer = NGramTokenizer(minN: 3, maxN: 1)
        #expect(tokenizer.maxN >= tokenizer.minN)
    }

    @Test("Unigram tokenization returns words")
    func testUnigramTokenization() {
        let tokenizer = NGramTokenizer(minN: 1, maxN: 1)
        let tokens = tokenizer.tokenize("hello world foo")
        #expect(tokens == ["hello", "world", "foo"])
    }

    @Test("Bigram tokenization returns correct bigrams")
    func testBigramTokenization() {
        let tokenizer = NGramTokenizer(minN: 2, maxN: 2)
        let tokens = tokenizer.tokenize("hello world foo")
        #expect(tokens == ["hello world", "world foo"])
    }

    @Test("Unigram and bigram tokenization returns both")
    func testUnigramAndBigramTokenization() {
        let tokenizer = NGramTokenizer(minN: 1, maxN: 2)
        let tokens = tokenizer.tokenize("hello world")
        #expect(tokens.contains("hello"))
        #expect(tokens.contains("world"))
        #expect(tokens.contains("hello world"))
    }

    @Test("Trigram tokenization returns correct trigrams")
    func testTrigramTokenization() {
        let tokenizer = NGramTokenizer(minN: 3, maxN: 3)
        let tokens = tokenizer.tokenize("a b c d")
        #expect(tokens.contains("a b c"))
        #expect(tokens.contains("b c d"))
        #expect(!tokens.contains("a b"))
    }

    @Test("Lowercase normalization works")
    func testLowercaseNormalization() {
        let tokenizer = NGramTokenizer(minN: 1, maxN: 1, lowercase: true)
        let tokens = tokenizer.tokenize("Hello WORLD Foo")
        #expect(tokens == ["hello", "world", "foo"])
    }

    @Test("No lowercase when disabled")
    func testNoLowercaseWhenDisabled() {
        let tokenizer = NGramTokenizer(minN: 1, maxN: 1, lowercase: false)
        let tokens = tokenizer.tokenize("Hello WORLD")
        #expect(tokens.contains("Hello"))
        #expect(tokens.contains("WORLD"))
    }

    @Test("Empty string returns empty array")
    func testEmptyString() {
        let tokenizer = NGramTokenizer(minN: 1, maxN: 2)
        let tokens = tokenizer.tokenize("")
        #expect(tokens.isEmpty)
    }

    @Test("String with only whitespace returns empty array")
    func testWhitespaceOnlyString() {
        let tokenizer = NGramTokenizer(minN: 1, maxN: 2)
        let tokens = tokenizer.tokenize("   \t\n  ")
        #expect(tokens.isEmpty)
    }

    @Test("Single word returns one unigram and no bigrams")
    func testSingleWord() {
        let tokenizer = NGramTokenizer(minN: 1, maxN: 2)
        let tokens = tokenizer.tokenize("hello")
        #expect(tokens == ["hello"])
    }

    @Test("Punctuation is used as delimiter")
    func testPunctuationDelimiter() {
        let tokenizer = NGramTokenizer(minN: 1, maxN: 1)
        let tokens = tokenizer.tokenize("hello, world! foo.")
        #expect(tokens.contains("hello"))
        #expect(tokens.contains("world"))
        #expect(tokens.contains("foo"))
        #expect(!tokens.contains(","))
        #expect(!tokens.contains("!"))
    }

    @Test("Two words produces one bigram only when minN=maxN=2")
    func testTwoWordsBigram() {
        let tokenizer = NGramTokenizer(minN: 2, maxN: 2)
        let tokens = tokenizer.tokenize("one two")
        #expect(tokens == ["one two"])
    }

    @Test("Bigram on single word is empty")
    func testBigramOnSingleWordIsEmpty() {
        let tokenizer = NGramTokenizer(minN: 2, maxN: 2)
        let tokens = tokenizer.tokenize("only")
        #expect(tokens.isEmpty)
    }
}

// MARK: - HashingVectorizer Tests

@Suite("HashingVectorizer Tests")
struct HashingVectorizerTests {

    @Test("Default init stores correct params")
    func testDefaultInit() {
        let vec = HashingVectorizer()
        #expect(vec.nFeatures == 1024)
        #expect(vec.ngramRange.min == 1)
        #expect(vec.ngramRange.max == 1)
        #expect(vec.lowercase == true)
    }

    @Test("Custom init stores nFeatures and ngramRange")
    func testCustomInit() {
        let vec = HashingVectorizer(nFeatures: 512, ngramRange: (2, 3), lowercase: false)
        #expect(vec.nFeatures == 512)
        #expect(vec.ngramRange.min == 2)
        #expect(vec.ngramRange.max == 3)
        #expect(vec.lowercase == false)
    }

    @Test("nFeatures clamped to at least 1")
    func testNFeaturesClampedToOne() {
        let vec = HashingVectorizer(nFeatures: 0)
        #expect(vec.nFeatures >= 1)
    }

    @Test("Transform returns one row per document")
    func testTransformRowCount() {
        let vec = HashingVectorizer(nFeatures: 64)
        let docs = ["hello world", "foo bar baz", "swift science"]
        let result = vec.transform(documents: docs)
        #expect(result.count == 3)
    }

    @Test("Transform row length equals nFeatures")
    func testTransformRowLength() {
        let vec = HashingVectorizer(nFeatures: 128)
        let result = vec.transform(documents: ["hello world foo"])
        #expect(result[0].count == 128)
    }

    @Test("Transform empty documents list returns empty result")
    func testTransformEmptyDocuments() {
        let vec = HashingVectorizer(nFeatures: 64)
        let result = vec.transform(documents: [])
        #expect(result.isEmpty)
    }

    @Test("Transform empty document string returns zero vector")
    func testTransformEmptyDocumentString() {
        let vec = HashingVectorizer(nFeatures: 64)
        let result = vec.transform(documents: [""])
        #expect(result[0].allSatisfy { $0 == 0.0 })
    }

    @Test("Transform produces non-zero values for non-empty input")
    func testTransformProducesNonZeroValues() {
        let vec = HashingVectorizer(nFeatures: 256)
        let result = vec.transform(documents: ["hello world swift science"])
        #expect(result[0].contains { $0 > 0.0 })
    }

    @Test("Same document produces same hash vector (determinism)")
    func testTransformDeterministic() {
        let vec = HashingVectorizer(nFeatures: 128)
        let result1 = vec.transform(documents: ["swift machine learning"])
        let result2 = vec.transform(documents: ["swift machine learning"])
        #expect(result1[0] == result2[0])
    }

    @Test("Different documents produce different vectors")
    func testTransformDifferentDocsDifferentVectors() {
        let vec = HashingVectorizer(nFeatures: 128)
        let result = vec.transform(documents: ["hello world", "completely different text"])
        #expect(result[0] != result[1])
    }

    @Test("Bigram ngramRange produces more non-zero features than unigram")
    func testBigramRangeProducesMoreFeatures() {
        let docs = ["hello world foo bar"]
        let unigramVec = HashingVectorizer(nFeatures: 512, ngramRange: (1, 1))
        let bigramVec = HashingVectorizer(nFeatures: 512, ngramRange: (1, 2))
        let unigramSum = unigramVec.transform(documents: docs)[0].reduce(0, +)
        let bigramSum = bigramVec.transform(documents: docs)[0].reduce(0, +)
        #expect(bigramSum >= unigramSum)
    }

    @Test("Lowercase normalization treats same tokens equivalently")
    func testLowercaseNormalizationEquivalence() {
        let vec = HashingVectorizer(nFeatures: 256, ngramRange: (1, 1), lowercase: true)
        let r1 = vec.transform(documents: ["hello world"])
        let r2 = vec.transform(documents: ["Hello World"])
        #expect(r1[0] == r2[0])
    }

    @Test("Transform all-whitespace document returns zero vector")
    func testTransformWhitespaceOnlyReturnsZeros() {
        let vec = HashingVectorizer(nFeatures: 64)
        let result = vec.transform(documents: ["   \t\n  "])
        #expect(result[0].allSatisfy { $0 == 0.0 })
    }
}
