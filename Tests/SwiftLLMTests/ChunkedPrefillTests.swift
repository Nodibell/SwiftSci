import Testing
import MLX
import MLXNN
import SwiftNLP
@testable import SwiftLLM

@Suite("Cached prompt prefill", .serialized)
struct ChunkedPrefillTests {
    private func model(tied: Bool = false, learned: Bool = false) -> TransformerDecoder {
        let config = LLMConfig(vocabSize: 12, numLayers: 2, hiddenDim: 16,
            numHeads: 4, numKVHeads: 2, intermediateSize: 24, maxSeqLen: 32,
            positionalEncoding: learned ? .learned : .llama3RoPE(scaling: Llama3RoPEScaling(factor: 32)),
            tieWordEmbeddings: tied)
        let model = TransformerDecoder(config: config, tokenizer: BPETokenizer(vocab: ["a": 0], merges: []))
        model.update(parameters: NestedDictionary.unflattened(model.parameters().flattened().map { name, tensor in
            let salt = name.utf8.reduce(0) { $0 + Int($1) }
            return (name, MLXArray((0..<tensor.size).map { Float(($0 * 7 + salt) % 29 - 14) / 32 }, tensor.shape))
        }))
        return model
    }

    @Test("Cached chunks preserve all logits and cannot attend to future tokens", arguments: [false, true])
    func cachedChunks(gpu: Bool) throws {
        try Device.withDefaultDevice(gpu ? .gpu : .cpu) {
            for learned in [false, true] {
                for tied in [false, true] {
                    let model = model(tied: tied, learned: learned)
                    let ids = [0, 1, 2, 3, 4, 5, 6]
                    let expected = model.forward(MLXArray(ids, [1, ids.count]))
                    eval(expected)
                    for size in [1, 2, 3, 7] {
                        let caches = model.layers.map { _ in KVCache() }
                        var chunks: [MLXArray] = []
                        for start in stride(from: 0, to: ids.count, by: size) {
                            let end = min(start + size, ids.count)
                            let chunk = model.forward(MLXArray(Array(ids[start..<end]), [1, end-start]),
                                caches: caches, offset: start)
                            eval(chunk)
                            #expect(chunk.shape == [1, end-start, 12])
                            #expect(caches.allSatisfy { $0.count == end })
                            chunks.append(chunk)
                        }
                        let actual = concatenated(chunks, axis: 1)
                        #expect(abs(actual - expected).max().item(Float.self) < 0.00001)
                    }
                    let changed = model.forward(MLXArray([0,1,2,3,11,10,9], [1,7]))
                    #expect(abs(changed[0,0..<4] - expected[0,0..<4]).max().item(Float.self) < 0.00001)
                }
            }
        }
    }
}
