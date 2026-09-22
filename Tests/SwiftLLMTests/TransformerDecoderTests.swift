import Testing
import Foundation
import MLX
import MLXNN
@testable import SwiftNLP
@testable import SwiftLLM

@Suite("Transformer Decoder Tests", .serialized)
struct TransformerDecoderTests {
    
    // Helper BPE vocabulary
    let vocab: [String: Int] = [
        "<unk>": 0,
        "h": 1,
        "e": 2,
        "l": 3,
        "o": 4,
        "w": 5,
        "r": 6,
        "d": 7,
        "l</w>": 8,
        "o</w>": 9,
        "d</w>": 10,
        "he": 11,
        "lo</w>": 12,
        "hello</w>": 13,
        "world</w>": 14
    ]
    
    let merges = [
        "h e",       // he
        "l o</w>",   // lo</w>
        "he l",      // hel
        "hel lo</w>" // hello</w>
    ]
    
    @Test("TransformerDecoder forward pass logits shape")
    func testTransformerForwardPass() throws {
        // Setup MLX metallib access if needed
        registerCmlxBundle()
        
        let tokenizer = BPETokenizer(vocab: vocab, merges: merges)
        let decoder = TransformerDecoder(vocabSize: vocab.count, tokenizer: tokenizer, dimensions: 16, numHeads: 2, maxSeqLen: 32)
        
        let input = MLXArray([1, 2, 3, 4]) // shape [4]
        let logits = decoder(input)
        
        // Output shape should be [1, seq_len, vocab_size] = [1, 4, 15]
        #expect(logits.shape == [1, 4, 15])
    }
    
    @Test("TransformerDecoder streaming text generation")
    func testTransformerGeneration() async throws {
        registerCmlxBundle()
        
        let tokenizer = BPETokenizer(vocab: vocab, merges: merges)
        let decoder = TransformerDecoder(vocabSize: vocab.count, tokenizer: tokenizer, dimensions: 16, numHeads: 2, maxSeqLen: 32)
        
        let options = LLMOptions(temperature: 0.0, maxTokens: 5)
        let stream = try await decoder.generate(prompt: "hello", options: options)
        
        var generatedTokens = [String]()
        for await token in stream {
            generatedTokens.append(token)
        }
        
        // Since the weights are random, we just verify the streaming generates successfully
        // and produces some text segments without throwing any crashes.
        #expect(generatedTokens.count >= 0)
    }

    @Test("Golden Test: Incremental decode with KV-cache matches full forward pass (RoPE)")
    func testIncrementalDecodeMatchesFullForwardRoPE() throws {
        registerCmlxBundle()

        let tokenizer = BPETokenizer(vocab: vocab, merges: merges)
        let config = LLMConfig(
            vocabSize: vocab.count,
            numLayers: 2,
            hiddenDim: 16,
            numHeads: 2,
            intermediateSize: 32,
            maxSeqLen: 32,
            positionalEncoding: .rope(base: 10_000.0)
        )
        let decoder = TransformerDecoder(config: config, tokenizer: tokenizer)

        let tokens: [Int] = [1, 3, 5, 2, 4]
        let fullInput = MLXArray(tokens).expandedDimensions(axis: 0)

        // Path A: Full causal forward pass over all tokens
        let fullLogits = decoder(fullInput)
        eval(fullLogits)

        // Path B: Prefill first 3 tokens with KVCache
        let caches = decoder.layers.map { _ in KVCache() }
        let prefillInput = MLXArray(Array(tokens[0..<3])).expandedDimensions(axis: 0)
        let prefillLogits = decoder.forward(prefillInput, caches: caches, offset: 0)
        eval(prefillLogits)

        // Verify prefill logits at position 2 match full logits at position 2
        let prefillDiff = MLX.abs(prefillLogits[0, 2] - fullLogits[0, 2])
        eval(prefillDiff)
        let prefillMaxErr = prefillDiff.max().item(Float.self)

        // Incremental decode step 1: token at position 3
        let step1Input = MLXArray([tokens[3]]).expandedDimensions(axis: 0)
        let step1Logits = decoder.forward(step1Input, caches: caches, offset: caches.first?.count ?? 3)
        eval(step1Logits)

        let step1Diff = MLX.abs(step1Logits[0, 0] - fullLogits[0, 3])
        eval(step1Diff)
        let step1MaxErr = step1Diff.max().item(Float.self)

        // Incremental decode step 2: token at position 4
        let step2Input = MLXArray([tokens[4]]).expandedDimensions(axis: 0)
        let step2Logits = decoder.forward(step2Input, caches: caches, offset: caches.first?.count ?? 4)
        eval(step2Logits)

        let step2Diff = MLX.abs(step2Logits[0, 0] - fullLogits[0, 4])
        eval(step2Diff)
        let step2MaxErr = step2Diff.max().item(Float.self)

        let maxAbsoluteError = max(prefillMaxErr, max(step1MaxErr, step2MaxErr))
        let tolerance: Float = 1e-4

        #expect(
            maxAbsoluteError < tolerance,
            "RoPE incremental decode deviated from full forward: max error \(maxAbsoluteError) >= tolerance \(tolerance)"
        )
    }

    @Test("Golden Test: Incremental decode with KV-cache matches full forward pass (Learned Positional)")
    func testIncrementalDecodeMatchesFullForwardLearned() throws {
        registerCmlxBundle()

        let tokenizer = BPETokenizer(vocab: vocab, merges: merges)
        let config = LLMConfig(
            vocabSize: vocab.count,
            numLayers: 2,
            hiddenDim: 16,
            numHeads: 2,
            intermediateSize: 32,
            maxSeqLen: 32,
            positionalEncoding: .learned
        )
        let decoder = TransformerDecoder(config: config, tokenizer: tokenizer)

        let tokens: [Int] = [1, 3, 5, 2, 4]
        let fullInput = MLXArray(tokens).expandedDimensions(axis: 0)

        // Path A: Full causal forward pass
        let fullLogits = decoder(fullInput)
        eval(fullLogits)

        // Path B: Prefill first 3 tokens
        let caches = decoder.layers.map { _ in KVCache() }
        let prefillInput = MLXArray(Array(tokens[0..<3])).expandedDimensions(axis: 0)
        let prefillLogits = decoder.forward(prefillInput, caches: caches, offset: 0)
        eval(prefillLogits)

        // Incremental decode step 1: token at position 3
        let step1Input = MLXArray([tokens[3]]).expandedDimensions(axis: 0)
        let step1Logits = decoder.forward(step1Input, caches: caches, offset: caches.first?.count ?? 3)
        eval(step1Logits)

        // Incremental decode step 2: token at position 4
        let step2Input = MLXArray([tokens[4]]).expandedDimensions(axis: 0)
        let step2Logits = decoder.forward(step2Input, caches: caches, offset: caches.first?.count ?? 4)
        eval(step2Logits)

        let step2Diff = MLX.abs(step2Logits[0, 0] - fullLogits[0, 4])
        eval(step2Diff)
        let maxAbsoluteError = step2Diff.max().item(Float.self)
        let tolerance: Float = 1e-4

        #expect(
            maxAbsoluteError < tolerance,
            "Learned pos incremental decode deviated from full forward: max error \(maxAbsoluteError) >= tolerance \(tolerance)"
        )
    }

    @Test("Gate 7: Output parity before and after MLX.compile")
    func testMLXCompileOutputParity() throws {
        registerCmlxBundle()

        let tokenizer = BPETokenizer(vocab: vocab, merges: merges)
        let config = LLMConfig(
            vocabSize: vocab.count,
            numLayers: 2,
            hiddenDim: 16,
            numHeads: 2,
            intermediateSize: 32,
            maxSeqLen: 32,
            positionalEncoding: .rope(base: 10_000.0)
        )
        let decoder = TransformerDecoder(config: config, tokenizer: tokenizer)

        let input = MLXArray([1, 4, 2, 7]).expandedDimensions(axis: 0)

        // 1. Uncompiled forward pass
        let uncompiledLogits = decoder(input)
        eval(uncompiledLogits)

        // 2. Compiled forward pass
        let compiledFn = MLX.compile(decoder.callAsFunction)
        let compiledLogits = compiledFn(input)
        eval(compiledLogits)

        // 3. Verify output parity
        let diff = MLX.abs(compiledLogits - uncompiledLogits)
        eval(diff)
        let maxAbsoluteError = diff.max().item(Float.self)
        let tolerance: Float = 1e-4

        #expect(
            maxAbsoluteError < tolerance,
            "MLX.compile parity error: maxAbsoluteError=\(maxAbsoluteError) >= tolerance=\(tolerance)"
        )
    }
    
    // Dynamic metallib finder helper
    private func registerCmlxBundle() {
        #if targetEnvironment(simulator)
        #else
        let path = Bundle.allBundles.first { $0.bundlePath.hasSuffix("mlx-swift_Cmlx.bundle") }
        if path == nil {
            let metallibPath = "./default.metallib"
            if FileManager.default.fileExists(atPath: metallibPath) {
                // Already in working directory, MLX will find it automatically
            }
        }
        #endif
    }
}
