import Testing
import SwiftNLP

@Suite("Corpus Lexical Profiler Tests")
struct LexicalProfilerTests {

    @Test("Token array profiling computes correct counts and TTR")
    func testTokenProfiling() throws {
        let tokens = ["apple", "banana", "apple", "cherry", "date", "banana"]
        let profile = CorpusLexicalProfiler.profile(tokens: tokens)
        
        #expect(profile.totalTokens == 6)
        #expect(profile.uniqueTokens == 4)
        #expect(abs(profile.typeTokenRatio - (4.0 / 6.0)) < 1e-6)
        // cherry and date appear once -> 2 / 4 = 0.5
        #expect(abs(profile.hapaxLegomenaRatio - 0.5) < 1e-6)
        #expect(profile.shannonEntropy > 0.0)
    }

    @Test("Document strings profiling with default tokenizer")
    func testTextProfiling() throws {
        let docs = [
            "SwiftSci is lightning fast on Apple Silicon.",
            "Apple Silicon unified memory powers scientific ML."
        ]
        let profile = CorpusLexicalProfiler.profile(texts: docs)
        #expect(profile.totalTokens > 0)
        #expect(profile.uniqueTokens > 0)
        #expect(profile.typeTokenRatio > 0.0 && profile.typeTokenRatio <= 1.0)
        #expect(profile.averageTokenLength > 0.0)
    }

    @Test("Empty input returns zero profile without crashing")
    func testEmptyInput() throws {
        let profile = CorpusLexicalProfiler.profile(tokens: [])
        #expect(profile.totalTokens == 0)
        #expect(profile.uniqueTokens == 0)
        #expect(profile.typeTokenRatio == 0.0)
        #expect(profile.shannonEntropy == 0.0)
    }
}
