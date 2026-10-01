import Foundation
import Testing
import MLX
import MLXNN
import SwiftNLP
@testable import SwiftLLM

@Suite("Tied output embeddings", .serialized)
struct TiedEmbeddingTests {
    @Test("Llama 3.2 checkpoint loads without a separate output head and projects with its embeddings")
    func checkpointTiedOutput() {
        Device.withDefaultDevice(.cpu) {
            var config = LLMConfig.llama1B
            config.vocabSize = 3
            config.numLayers = 0
            config.hiddenDim = 2
            config.numHeads = 1
            config.numKVHeads = nil
            config.maxSeqLen = 4
            let model = TransformerDecoder(config: config,
                tokenizer: BPETokenizer(vocab: ["<unk>": 0], merges: []))
            let weights: [Float] = [1, 0, 0, 2, 3, 4]
            let missing = model.loadWeights([
                "model.embed_tokens.weight": MLXArray(weights, [3, 2]),
                "model.norm.weight": MLXArray.ones([2])
            ])
            #expect(!missing.contains("lm_head.weight"))
            let actual = model(MLXArray([0, 1])).asArray(Float.self)
            var expected: [Double] = []
            for token in [0, 1] {
                let row = [Double(weights[2 * token]), Double(weights[2 * token + 1])]
                let denominator = sqrt(row.map { $0 * $0 }.reduce(0, +) / 2 + Double(config.rmsNormEps))
                for word in 0..<3 {
                    expected.append((row[0] * Double(weights[word * 2]) + row[1] * Double(weights[word * 2 + 1])) / denominator)
                }
            }
            let error = zip(actual, expected).map { abs(Double($0) - $1) }.max()!
            #expect(error < 2e-5)
        }
    }

    private func tinyModel(tied: Bool, dimensions: Int = 2) -> TransformerDecoder {
        TransformerDecoder(config: LLMConfig(vocabSize: 3, numLayers: 0,
            hiddenDim: dimensions, numHeads: 1, maxSeqLen: 4, tieWordEmbeddings: tied),
            tokenizer: BPETokenizer(vocab: ["<unk>": 0], merges: []))
    }

    @Test("Tied models contain no independent output parameters; ordinary models retain their head")
    func parameterOwnership() {
        Device.withDefaultDevice(.cpu) {
            let tied = tinyModel(tied: true)
            let ordinary = tinyModel(tied: false)
            #expect(tied.lmHead == nil)
            #expect(ordinary.lmHead != nil)
            #expect(tied.parameters().flattened().allSatisfy { !$0.0.hasPrefix("lmHead.") })
            #expect(tied.trainableParameters().flattened().allSatisfy { !$0.0.hasPrefix("lmHead.") })
            let tiedCount = tied.parameters().flattened().reduce(0) { $0 + $1.1.size }
            let ordinaryCount = ordinary.parameters().flattened().reduce(0) { $0 + $1.1.size }
            #expect(ordinaryCount - tiedCount == 6)
            #expect(ordinary.loadWeights([String: MLXArray]()).contains("lm_head.weight"))
            #expect(!LLMConfig.debug.tieWordEmbeddings)
            #expect(!LLMConfig.llama8B.tieWordEmbeddings)
            #expect(LLMConfig.llama1B.tieWordEmbeddings)
        }
    }

    @Test("Tied output follows checkpoint reloads, parameter updates, and module replacement",
          arguments: [false, true])
    func liveEmbedding(gpu: Bool) {
        Device.withDefaultDevice(gpu ? .gpu : .cpu) {
            let model = tinyModel(tied: true)
            let first = MLXArray([Float(1), 0, 0, 2, 3, 4], [3, 2])
            _ = model.loadWeights(["model.embed_tokens.weight": first, "model.norm.weight": MLXArray.ones([2]),
                "lm_head.weight": MLXArray.zeros([3, 2])])
            func check(_ expectedWeights: MLXArray) {
                let tokens = MLXArray([0, 1])
                let hidden = model.finalNorm(expectedWeights[tokens])
                let expected = matmul(hidden, expectedWeights.T)
                let actual = model(tokens).reshaped([2, 3])
                #expect(MLX.abs(actual - expected).max().item(Float.self) < 2e-5)
                #expect(model.lmHead == nil)
            }
            check(first)
            let second = MLXArray([Float(2), 1, 1, 3, 4, 2], [3, 2])
            _ = model.loadWeights(["model.embed_tokens.weight": second, "model.norm.weight": MLXArray.ones([2])])
            check(second)
            let third = MLXArray([Float(3), 2, 4, 1, 1, 5], [3, 2])
            model.update(parameters: NestedDictionary.unflattened(["embedding.weight": third]))
            check(third)
            let replacement = Embedding(weight: first)
            let modules: [String: Module] = ["embedding": replacement]
            model.update(modules: NestedDictionary.unflattened(modules))
            check(first)
            let missing = model.loadWeights(["lm_head.weight": second])
            #expect(missing.contains("model.embed_tokens.weight"))
            check(first)
        }
    }

    @Test("One embedding gradient accumulates input and output contributions", arguments: [false, true])
    func sharedGradient(gpu: Bool) throws {
        try Device.withDefaultDevice(gpu ? .gpu : .cpu) {
            let model = tinyModel(tied: true)
            _ = model.loadWeights([
                "model.embed_tokens.weight": MLXArray([Float(1), 0, 0, 2, 3, 4], [3, 2]),
                "model.norm.weight": MLXArray.ones([2])
            ])
            let valueGradient = valueAndGrad(model: model) { (model: TransformerDecoder, inputs: [MLXArray]) in
                [model(inputs[0]).sum()]
            }
            let (_, gradients) = valueGradient(model, [MLXArray([0])])
            let flat = Dictionary(uniqueKeysWithValues: gradients.flattened())
            let gradient = try #require(flat["embedding.weight"])
            let actual = gradient.asArray(Float.self)
            let radius = sqrt(0.5 + Double(model.config.rmsNormEps))
            // The selected row contributes through RMSNorm and through the output dot product.
            let expected = [5 / radius - 2 / pow(radius, 3), 6 / radius,
                            1 / radius, 0, 1 / radius, 0]
            let error = zip(actual, expected).map { abs(Double($0) - $1) }.max()!
            #expect(error < 2e-5)
            #expect(flat.keys.allSatisfy { !$0.hasPrefix("lmHead.") })
            let updated = model.embedding.weight - gradient * 0.01
            eval(updated)
            model.update(parameters: NestedDictionary.unflattened(["embedding.weight": updated]))
            let reference = matmul(model.finalNorm(updated[MLXArray([0])]), updated.T)
            #expect(MLX.abs(model(MLXArray([0])).reshaped([1, 3]) - reference).max().item(Float.self) < 2e-5)
        }
    }

    @Test("Tied output uses a replaced quantized embedding for projection", arguments: [false, true])
    func quantizedEmbedding(gpu: Bool) {
        Device.withDefaultDevice(gpu ? .gpu : .cpu) {
            let model = tinyModel(tied: true, dimensions: 64)
            let weight = MLXArray((0..<192).map { Float(($0 * 3) % 17 - 8) / 16 }, [3, 64])
            let embedding = QuantizedEmbedding(weight: weight, groupSize: 64, bits: 4)
            let modules: [String: Module] = ["embedding": embedding]
            model.update(modules: NestedDictionary.unflattened(modules))
            let tokens = MLXArray([0, 1])
            let dequantized = embedding(MLXArray([0, 1, 2]))
            let hidden = model.finalNorm(dequantized[tokens])
            let expected = matmul(hidden, dequantized.T)
            let actual = model(tokens).reshaped([2, 3])
            #expect(MLX.abs(actual - expected).max().item(Float.self) < 1e-4)
            #expect(model.lmHead == nil)
        }
    }
}
