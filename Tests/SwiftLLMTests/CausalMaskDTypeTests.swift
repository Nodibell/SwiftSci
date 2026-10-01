import Testing
import MLX
import MLXNN
import SwiftNLP
@testable import SwiftLLM

@Suite("Automatic causal-mask dtype", .serialized)
struct CausalMaskDTypeTests {
    @Test("Decoder prefill and cached decoding accept low-precision weights", arguments: [false, true])
    func decoderMasks(gpu: Bool) {
        Device.withDefaultDevice(gpu ? .gpu : .cpu) {
            for dtype in [DType.float16, .bfloat16] {
                let config = LLMConfig(vocabSize: 8, numLayers: 1, hiddenDim: 64,
                    numHeads: 1, intermediateSize: 16, maxSeqLen: 8)
                let model = TransformerDecoder(config: config,
                    tokenizer: BPETokenizer(vocab: ["<unk>": 0], merges: []))
                let parameters = model.parameters().flattened().map { name, value in
                    (name, MLXArray((0..<value.size).map { Float(($0 * 3) % 17 - 8) / 32 }, value.shape).asType(dtype))
                }
                model.update(parameters: NestedDictionary.unflattened(parameters))
                let full = model(MLXArray([0, 1, 2, 3]))
                eval(full)
                #expect(full.dtype == dtype)
                #expect(full.shape == [1, 4, 8])
                #expect(isFinite(full).all().item(Bool.self))
                let caches = [KVCache()]
                let prefill = model.forward(MLXArray([0, 1, 2]), caches: caches)
                let decoded = model.forward(MLXArray([3]), caches: caches, offset: 3)
                eval(prefill, decoded)
                let error = MLX.abs(concatenated([prefill, decoded], axis: 1).asType(.float32) - full.asType(.float32)).max().item(Float.self)
                // BF16 and F16 round intermediate matrix products at different sequence shapes.
                #expect(error < (dtype == .bfloat16 ? 0.01 : 0.002))
            }
        }
    }

    @Test("Direct blocks construct masks in the activation dtype", arguments: [false, true])
    func blockMasks(gpu: Bool) {
        Device.withDefaultDevice(gpu ? .gpu : .cpu) {
            for dtype in [DType.float16, .bfloat16] {
                let config = LLMConfig(vocabSize: 8, numLayers: 1, hiddenDim: 64,
                    numHeads: 1, intermediateSize: 16)
                let block = TransformerBlock(config: config)
                block.update(parameters: NestedDictionary.unflattened(block.parameters().flattened().map {
                    ($0.0, $0.1.asType(dtype))
                }))
                let input = MLXArray.ones([1, 3, 64], dtype: dtype)
                let actual = block(input)
                let explicitMask = MultiHeadAttention.createAdditiveCausalMask(3).asType(dtype)
                let expected = block.forward(input, mask: explicitMask)
                #expect(actual.dtype == dtype)
                #expect(actual.asArray(Float.self) == expected.asArray(Float.self))
            }
        }
    }
}
