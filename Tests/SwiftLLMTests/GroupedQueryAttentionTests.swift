import Foundation
import Testing
import MLX
import MLXNN
@testable import SwiftNLP
@testable import SwiftLLM

@Suite("Grouped-query attention", .serialized)
struct GroupedQueryAttentionTests {
    @Test("KV projection and cache widths follow the configured KV head count")
    func projectionShapes() throws {
        try Device.withDefaultDevice(.cpu) {
            let config = LLMConfig(vocabSize: 12, numLayers: 1, hiddenDim: 16,
                                   numHeads: 4, numKVHeads: 2, intermediateSize: 8,
                                   maxSeqLen: 8, positionalEncoding: .learned)
            let block = TransformerBlock(config: config)
            let weights = Dictionary(uniqueKeysWithValues: block.parameters().flattened())
            let query = try #require(weights["attention.query_proj.weight"])
            #expect(query.shape == [16, 16])
            let key = try #require(weights["attention.key_proj.weight"])
            #expect(key.shape == [8, 16])
            let value = try #require(weights["attention.value_proj.weight"])
            #expect(value.shape == [8, 16])
            let outputProjection = try #require(weights["attention.out_proj.weight"])
            #expect(outputProjection.shape == [16, 16])
            let cache = KVCache()
            let output = block.forward(MLXArray.zeros([2, 1, 16]), cache: cache)
            eval(output)
            #expect(cache.keys?.shape == [2, 2, 1, 4])
            #expect(cache.values?.shape == [2, 2, 1, 4])
        }
    }
    private func fixedWeights(_ module: Module) {
        let parameters = module.parameters().flattened().map { name, tensor in
            let salt = name.utf8.reduce(0) { $0 + Int($1) }
            let values = (0..<tensor.size).map { Float(($0 * 7 + salt) % 29 - 14) / 32 }
            return (name, MLXArray(values, tensor.shape))
        }
        module.update(parameters: NestedDictionary.unflattened(parameters))
    }

    @Test("Direct attention matches an independent Double grouped-head calculation",
          arguments: [1, 2, 4], [false, true])
    func scalarOracle(kvHeads: Int, gpu: Bool) throws {
        try Device.withDefaultDevice(gpu ? .gpu : .cpu) {
            for (dimensions, queries) in [(8, 3), (256, 1)] {
                let heads = 4, headDim = dimensions / heads, batch = 2, keys = 4
                let block = TransformerBlock(config: LLMConfig(vocabSize: 12, numLayers: 1,
                    hiddenDim: dimensions, numHeads: heads, numKVHeads: kvHeads,
                    intermediateSize: 8, positionalEncoding: .learned))
                let attention = block.attention
                fixedWeights(attention)
                let weights = Dictionary(uniqueKeysWithValues: attention.parameters().flattened())
                let qw = try #require(weights["query_proj.weight"]).asArray(Float.self).map(Double.init)
                let kw = try #require(weights["key_proj.weight"]).asArray(Float.self).map(Double.init)
                let vw = try #require(weights["value_proj.weight"]).asArray(Float.self).map(Double.init)
                let ow = try #require(weights["out_proj.weight"]).asArray(Float.self).map(Double.init)
                func data(_ count: Int, _ salt: Int) -> [Float] {
                    (0..<count).map { Float(($0 * 3 + salt) % 23 - 11) / 16 }
                }
                let qInput = data(batch * queries * dimensions, 1)
                let kInput = data(batch * keys * dimensions, 4)
                let vInput = data(batch * keys * dimensions, 9)
                func project(_ input: [Double], _ weight: [Double], rows: Int, outputs: Int) -> [Double] {
                    var result = [Double](repeating: 0, count: rows * outputs)
                    for row in 0..<rows {
                        for output in 0..<outputs {
                            for column in 0..<dimensions {
                                result[row * outputs + output] += input[row * dimensions + column] * weight[output * dimensions + column]
                            }
                        }
                    }
                    return result
                }
                let q = project(qInput.map(Double.init), qw, rows: batch * queries, outputs: dimensions)
                let k = project(kInput.map(Double.init), kw, rows: batch * keys, outputs: kvHeads * headDim)
                let v = project(vInput.map(Double.init), vw, rows: batch * keys, outputs: kvHeads * headDim)
                var joined = [Double](repeating: 0, count: batch * queries * dimensions)
                for b in 0..<batch {
                    for h in 0..<heads {
                        let kh = h / (heads / kvHeads)
                        for query in 0..<queries {
                            let visibleKeys = query + 2
                            var scores = [Double](repeating: 0, count: visibleKeys)
                            for key in 0..<visibleKeys {
                                for d in 0..<headDim {
                                    scores[key] += q[(b * queries + query) * dimensions + h * headDim + d]
                                        * k[(b * keys + key) * kvHeads * headDim + kh * headDim + d]
                                }
                                scores[key] /= Double(headDim).squareRoot()
                            }
                            let maximum = scores.max()!
                            let exponentials = scores.map { Foundation.exp($0 - maximum) }
                            let denominator = exponentials.reduce(0, +)
                            for key in 0..<visibleKeys {
                                for d in 0..<headDim {
                                    joined[(b * queries + query) * dimensions + h * headDim + d] +=
                                        exponentials[key] / denominator * v[(b * keys + key) * kvHeads * headDim + kh * headDim + d]
                                }
                            }
                        }
                    }
                }
                let expected = project(joined, ow, rows: batch * queries, outputs: dimensions)
                let mask: [Float] = (0..<queries).flatMap { query in
                    (0..<keys).map { $0 <= query + 1 ? Float(0) : -Float.infinity }
                }
                let actual = attention(MLXArray(qInput, [batch, queries, dimensions]),
                    keys: MLXArray(kInput, [batch, keys, dimensions]),
                    values: MLXArray(vInput, [batch, keys, dimensions]),
                    mask: MLXArray(mask, [queries, keys])).asArray(Float.self)
                let error = zip(actual, expected).map { abs(Double($0) - $1) }.max()!
                #expect(error < 2e-5, "KV heads: \(kvHeads), GPU: \(gpu), max error: \(error)")
            }
        }
    }

    @Test("Grouped KV caches preserve batched rotary and learned-position logits",
          arguments: [1, 2, 4], [false, true])
    func cachedDecode(kvHeads: Int, gpu: Bool) {
        Device.withDefaultDevice(gpu ? .gpu : .cpu) {
            for positional in [PositionalEncodingScheme.learned, .rope(base: 10_000)] {
                for batch in [1, 3] {
                    let config = LLMConfig(vocabSize: 12, numLayers: 2, hiddenDim: 256,
                        numHeads: 4, numKVHeads: kvHeads, intermediateSize: 10,
                        maxSeqLen: 8, positionalEncoding: positional)
                    let model = TransformerDecoder(config: config,
                        tokenizer: BPETokenizer(vocab: ["<unk>": 0], merges: []))
                    fixedWeights(model)
                    let tokens = (0..<batch).map { b in (0..<6).map { ($0 + b * 2) % 12 } }
                    let full = model(MLXArray(tokens.flatMap { $0 }, [batch, 6]))
                    eval(full)
                    let caches = model.layers.map { _ in KVCache() }
                    var chunks: [MLXArray] = []
                    for range in [0..<2, 2..<3, 3..<4, 4..<5, 5..<6] {
                        let input = MLXArray(tokens.flatMap { Array($0[range]) }, [batch, range.count])
                        let result = model.forward(input, caches: caches, offset: range.lowerBound)
                        eval(result)
                        chunks.append(result)
                        for cache in caches {
                            #expect(cache.keys?.shape == [batch, kvHeads, range.upperBound, 64])
                            #expect(cache.values?.shape == [batch, kvHeads, range.upperBound, 64])
                            #expect(cache.count == range.upperBound)
                        }
                    }
                    let error = MLX.abs(concatenated(chunks, axis: 1) - full).max().item(Float.self)
                    // Use the existing full-decoder Float32 parity budget. Different GEMM
                    // shapes in prefill and single-token decoding change accumulation order.
                    #expect(error < 1e-4, "KV heads: \(kvHeads), batch: \(batch), GPU: \(gpu)")
                }
            }
        }
    }

    @Test("Default KV heads follow query heads and Llama presets use eight KV heads")
    func configuration() throws {
        var config = LLMConfig(vocabSize: 12, numLayers: 1, hiddenDim: 8, numHeads: 4)
        #expect(config.numKVHeads == nil)
        config.numHeads = 2
        let block = TransformerBlock(config: config)
        let key = try #require(Dictionary(uniqueKeysWithValues: block.parameters().flattened())["attention.key_proj.weight"])
        #expect(key.shape == [8, 8])
        #expect(LLMConfig.llama1B.numKVHeads == 8)
        #expect(LLMConfig.llama8B.numKVHeads == 8)
    }

    @Test("Public weight loading replaces grouped projections without changing parameter paths")
    func loadGroupedWeights() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = TransformerDecoder(config: LLMConfig(vocabSize: 12, numLayers: 1,
                hiddenDim: 8, numHeads: 4, numKVHeads: 2, intermediateSize: 8, maxSeqLen: 8),
                tokenizer: BPETokenizer(vocab: ["<unk>": 0], merges: []))
            let keys = (0..<32).map { Float($0) / 32 }
            let values = (0..<32).map { Float(31 - $0) / 32 }
            let missing = model.loadWeights([
                "model.layers.0.self_attn.k_proj.weight": MLXArray(keys, [4, 8]),
                "model.layers.0.self_attn.v_proj.weight": MLXArray(values, [4, 8])
            ])
            #expect(!missing.contains("model.layers.0.self_attn.k_proj.weight"))
            #expect(!missing.contains("model.layers.0.self_attn.v_proj.weight"))
            let loaded = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            let k = try #require(loaded["layers.0.attention.key_proj.weight"])
            let v = try #require(loaded["layers.0.attention.value_proj.weight"])
            #expect(k.shape == [4, 8] && k.asArray(Float.self) == keys)
            #expect(v.shape == [4, 8] && v.asArray(Float.self) == values)
            let cache = KVCache()
            let output = model.forward(MLXArray([0, 1]), caches: [cache])
            eval(output)
            #expect(output.shape == [1, 2, 12])
            #expect(cache.keys?.shape == [1, 2, 2, 2])
        }
    }

}
