#if os(macOS)
import Testing
import Foundation
import MLX
@testable import SwiftLLM
import SwiftNLP

@Suite("KVCache and LLM Streaming Tests")
struct KVCacheTests {
    
    @Test("KVCache accumulation and reset")
    func testKVCacheAccumulation() {
        let cache = KVCache()
        #expect(cache.keys == nil)
        #expect(cache.values == nil)
        
        let k1 = MLXArray([1.0, 2.0] as [Float]).reshaped([1, 2])
        let v1 = MLXArray([3.0, 4.0] as [Float]).reshaped([1, 2])
        
        let (resK1, resV1) = cache.update(keys: k1, values: v1)
        #expect(resK1.shape == [1, 2])
        #expect(resV1.shape == [1, 2])
        
        let k2 = MLXArray([5.0, 6.0] as [Float]).reshaped([1, 2])
        let v2 = MLXArray([7.0, 8.0] as [Float]).reshaped([1, 2])
        
        let (resK2, resV2) = cache.update(keys: k2, values: v2)
        #expect(resK2.shape == [1, 4])
        #expect(resV2.shape == [1, 4])
        
        cache.reset()
        #expect(cache.keys == nil)
        #expect(cache.values == nil)
    }

    @Test("TransformerDecoder generateStream produces token stream")
    func testGenerateStream() async throws {
        struct DummyTokenizer: Tokenizer {
            func tokenize(text: String) -> [String] { ["token"] }
            func encode(text: String) -> [Int] { [1, 2, 3] }
            func decode(tokens: [Int]) -> String { "token" }
            func tokenToString(token: Int) -> String? { "token" }
            func stringToToken(string: String) -> Int? { 1 }
        }
        
        let decoder = TransformerDecoder(vocabSize: 10, tokenizer: DummyTokenizer(), dimensions: 16, numHeads: 2, maxSeqLen: 32)
        let stream = decoder.generateStream(prompt: "Hello", options: LLMOptions(maxTokens: 3))
        
        var count = 0
        for try await token in stream {
            #expect(!token.isEmpty)
            count += 1
        }
        
        #expect(count == 3)
    }
}
#endif
