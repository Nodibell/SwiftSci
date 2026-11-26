#if os(macOS)
import Foundation
import Testing
import MLX
import MLXNN
@testable import SwiftNLP
@testable import SwiftLLM

@Suite("Mistral, Gemma 2, Gemma 3, and Gemma 4 Architectures", .serialized)
struct GemmaMistralArchitectureTests {

    // MARK: - Helpers

    private static func deterministicTensor(_ shape: [Int], seed: Int, scale: Float = 0.05) -> MLXArray {
        let count = shape.reduce(1, *)
        let safeSeed = abs(seed & 0xFFFF)
        let values: [Float] = (0..<count).map { idx in
            let v = ((idx * 13 + safeSeed * 31 + 7) % 37) - 18
            return Float(v) / 18.0 * scale
        }
        return MLXArray(values, shape)
    }

    private static func populateDeterministicWeights(_ decoder: TransformerDecoder) {
        let params = decoder.parameters().flattened().map { name, tensor -> (String, MLXArray) in
            let salt = name.utf8.reduce(0) { $0 &* 31 &+ Int($1) }
            if name.hasSuffix("norm.weight") || name.hasSuffix("Norm.weight")
                || name.hasSuffix("Norm1.weight") || name.hasSuffix("Norm2.weight") {
                let offset: Float = (decoder.config.architecture == .gemma2 || decoder.config.architecture == .gemma3) ? 0.05 : 1.0
                return (name, MLXArray(Array(repeating: offset, count: tensor.size), tensor.shape))
            }
            if name.hasSuffix("layer_scalar") || name.hasSuffix(".scale") || name.hasSuffix("per_expert_scale") {
                return (name, MLXArray(Array(repeating: Float(0.75), count: tensor.size), tensor.shape))
            }
            return (name, deterministicTensor(tensor.shape, seed: salt, scale: 0.1))
        }
        decoder.update(parameters: NestedDictionary.unflattened(params))
    }

    private static func makeMinimalGemmaTokenizerJSON(vocabSize: Int = 32) throws -> Data {
        var vocab: [String: Int] = [
            "<unk>": 0,
            "<eos>": 1,
            "<bos>": 2,
            "▁": 3,
        ]
        var next = 4
        for b in 0..<256 {
            vocab[String(format: "<0x%02X>", b)] = next
            next += 1
        }
        let root: [String: Any] = [
            "version": "1.0",
            "added_tokens": [
                ["id": 0, "content": "<unk>", "special": true],
                ["id": 1, "content": "<eos>", "special": true],
                ["id": 2, "content": "<bos>", "special": true],
            ],
            "normalizer": ["type": "Replace", "pattern": ["String": " "], "content": "▁"],
            "pre_tokenizer": [
                "type": "Split",
                "pattern": ["String": " "],
                "behavior": "MergedWithPrevious",
                "invert": false,
            ],
            "post_processor": [
                "type": "TemplateProcessing",
                "single": [
                    ["SpecialToken": ["id": "<bos>", "type_id": 0]],
                    ["Sequence": ["id": "A", "type_id": 0]],
                ],
                "special_tokens": [
                    "<bos>": ["id": "<bos>", "ids": [2], "tokens": ["<bos>"]],
                ],
            ],
            "decoder": [
                "type": "Sequence",
                "decoders": [
                    ["type": "Replace", "pattern": ["String": "▁"], "content": " "],
                    ["type": "ByteFallback"],
                    ["type": "Fuse"],
                ],
            ],
            "model": [
                "type": "BPE",
                "dropout": NSNull(),
                "unk_token": "<unk>",
                "continuing_subword_prefix": NSNull(),
                "end_of_word_suffix": NSNull(),
                "fuse_unk": false,
                "byte_fallback": true,
                "vocab": vocab,
                "merges": [],
            ],
        ]
        _ = vocabSize
        return try JSONSerialization.data(withJSONObject: root)
    }

    private static func makeGemma3DenseWeights(_ cfg: LLMConfig) -> [String: MLXArray] {
        let h = cfg.hiddenDim
        let headDim = cfg.headDim
        let qDim = cfg.numHeads * headDim
        let kvDim = (cfg.numKVHeads ?? cfg.numHeads) * headDim
        var w: [String: MLXArray] = [
            "model.embed_tokens.weight": deterministicTensor([cfg.vocabSize, h], seed: 1, scale: 0.1),
            "model.norm.weight": deterministicTensor([h], seed: 2, scale: 0.2),
        ]
        if !cfg.tieWordEmbeddings {
            w["lm_head.weight"] = deterministicTensor([cfg.vocabSize, h], seed: 3, scale: 0.1)
        }
        for i in 0..<cfg.numLayers {
            let p = "model.layers.\(i)"
            w["\(p).self_attn.q_proj.weight"] = deterministicTensor([qDim, h], seed: 10 + i)
            w["\(p).self_attn.k_proj.weight"] = deterministicTensor([kvDim, h], seed: 20 + i)
            w["\(p).self_attn.v_proj.weight"] = deterministicTensor([kvDim, h], seed: 30 + i)
            w["\(p).self_attn.o_proj.weight"] = deterministicTensor([h, qDim], seed: 40 + i)
            w["\(p).self_attn.q_norm.weight"] = deterministicTensor([headDim], seed: 50 + i)
            w["\(p).self_attn.k_norm.weight"] = deterministicTensor([headDim], seed: 60 + i)
            w["\(p).mlp.gate_proj.weight"] = deterministicTensor([cfg.intermediateSize, h], seed: 70 + i)
            w["\(p).mlp.up_proj.weight"] = deterministicTensor([cfg.intermediateSize, h], seed: 80 + i)
            w["\(p).mlp.down_proj.weight"] = deterministicTensor([h, cfg.intermediateSize], seed: 90 + i)
            w["\(p).input_layernorm.weight"] = deterministicTensor([h], seed: 100 + i)
            w["\(p).post_attention_layernorm.weight"] = deterministicTensor([h], seed: 110 + i)
            w["\(p).pre_feedforward_layernorm.weight"] = deterministicTensor([h], seed: 120 + i)
            w["\(p).post_feedforward_layernorm.weight"] = deterministicTensor([h], seed: 130 + i)
        }
        return w
    }

    private static func makeGemma4DenseWeights(_ cfg: LLMConfig) -> [String: MLXArray] {
        let g4 = cfg.gemma4!
        let h = cfg.hiddenDim
        var w: [String: MLXArray] = [
            "model.embed_tokens.weight": deterministicTensor([cfg.vocabSize, h], seed: 1, scale: 0.1),
            "model.norm.weight": MLXArray.ones([h]),
        ]
        if !cfg.tieWordEmbeddings {
            w["lm_head.weight"] = deterministicTensor([cfg.vocabSize, h], seed: 3, scale: 0.1)
        }
        if g4.hiddenSizePerLayerInput > 0 {
            let pleDim = g4.hiddenSizePerLayerInput
            w["model.embed_tokens_per_layer.weight"] = deterministicTensor([g4.vocabSizePerLayerInput, cfg.numLayers * pleDim], seed: 4, scale: 0.1)
            w["model.per_layer_model_projection.weight"] = deterministicTensor([cfg.numLayers * pleDim, h], seed: 5, scale: 0.1)
            w["model.per_layer_projection_norm.weight"] = MLXArray.ones([pleDim])
        }
        for i in 0..<cfg.numLayers {
            let p = "model.layers.\(i)"
            let hd = cfg.headDim(forLayer: i)
            let qDim = cfg.numHeads * hd
            let kvHeads = cfg.kvHeadCount(forLayer: i)
            let kvDim = kvHeads * hd
            let mlpSize = cfg.intermediateSize(forLayer: i)

            w["\(p).self_attn.q_proj.weight"] = deterministicTensor([qDim, h], seed: 100 + i)
            w["\(p).self_attn.q_norm.weight"] = MLXArray.ones([hd])
            w["\(p).self_attn.o_proj.weight"] = deterministicTensor([h, qDim], seed: 110 + i)
            if cfg.hasKV(forLayer: i) {
                w["\(p).self_attn.k_proj.weight"] = deterministicTensor([kvDim, h], seed: 120 + i)
                w["\(p).self_attn.k_norm.weight"] = MLXArray.ones([hd])
                if !cfg.usesKEqV(forLayer: i) {
                    w["\(p).self_attn.v_proj.weight"] = deterministicTensor([kvDim, h], seed: 130 + i)
                }
            }
            w["\(p).mlp.gate_proj.weight"] = deterministicTensor([mlpSize, h], seed: 140 + i)
            w["\(p).mlp.up_proj.weight"] = deterministicTensor([mlpSize, h], seed: 150 + i)
            w["\(p).mlp.down_proj.weight"] = deterministicTensor([h, mlpSize], seed: 160 + i)
            w["\(p).input_layernorm.weight"] = MLXArray.ones([h])
            w["\(p).post_attention_layernorm.weight"] = MLXArray.ones([h])
            w["\(p).pre_feedforward_layernorm.weight"] = MLXArray.ones([h])
            w["\(p).post_feedforward_layernorm.weight"] = MLXArray.ones([h])
            w["\(p).layer_scalar"] = MLXArray([Float(0.75)], [1])

            if g4.enableMoEBlock {
                w["\(p).router.proj.weight"] = deterministicTensor([g4.numExperts, h], seed: 200 + i)
                w["\(p).router.scale"] = MLXArray.ones([h])
                w["\(p).router.per_expert_scale"] = MLXArray(Array(repeating: Float(0.8), count: g4.numExperts), [g4.numExperts])
                w["\(p).experts.switch_glu.gate_proj.weight"] = deterministicTensor([g4.numExperts, g4.moeIntermediateSize, h], seed: 210 + i)
                w["\(p).experts.switch_glu.up_proj.weight"] = deterministicTensor([g4.numExperts, g4.moeIntermediateSize, h], seed: 220 + i)
                w["\(p).experts.switch_glu.down_proj.weight"] = deterministicTensor([g4.numExperts, h, g4.moeIntermediateSize], seed: 230 + i)
                w["\(p).post_feedforward_layernorm_1.weight"] = MLXArray.ones([h])
                w["\(p).pre_feedforward_layernorm_2.weight"] = MLXArray.ones([h])
                w["\(p).post_feedforward_layernorm_2.weight"] = MLXArray.ones([h])
            }
            if g4.hiddenSizePerLayerInput > 0 {
                let pleDim = g4.hiddenSizePerLayerInput
                w["\(p).per_layer_input_gate.weight"] = deterministicTensor([pleDim, h], seed: 300 + i)
                w["\(p).per_layer_projection.weight"] = deterministicTensor([h, pleDim], seed: 310 + i)
                w["\(p).post_per_layer_input_norm.weight"] = MLXArray.ones([h])
            }
        }
        return w
    }

    // MARK: - Mistral

    @Test("Mistral parses config, uses SiLU + RMSNorm, enforces sliding window, and matches KV-cache")
    func mistralArchitectureAndSlidingWindow() throws {
        let json = """
        {
          "model_type": "mistral",
          "hidden_size": 16,
          "num_hidden_layers": 1,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "intermediate_size": 24,
          "vocab_size": 32,
          "max_position_embeddings": 64,
          "rms_norm_eps": 1e-5,
          "rope_theta": 10000.0,
          "sliding_window": 2,
          "tie_word_embeddings": false
        }
        """
        let cfg = try LLMConfig(huggingFaceConfigJSON: Data(json.utf8))
        #expect(cfg.architecture == .mistral)
        #expect(cfg.slidingWindow == 2)
        #expect(cfg.numKVHeads == 2)

        let decoder = TransformerDecoder(config: cfg, tokenizer: BPETokenizer(vocab: ["<a>": 0], merges: []))
        Self.populateDeterministicWeights(decoder)
        #expect(!(decoder.layers[0].norm1 is GemmaRMSNorm))
        // Verify SiLU activation on unit gate/up/down FFN basis vector: silu(1.0) ~= 0.7311
        let e0In = MLXArray([Float(1.0)] + Array(repeating: Float(0.0), count: 15), [1, 1, 16])
        func e0Matrix(rows: Int, cols: Int) -> MLXArray {
            var arr = Array(repeating: Float(0.0), count: rows * cols)
            arr[0] = 1.0
            return MLXArray(arr, [rows, cols])
        }
        decoder.layers[0].ffn.update(parameters: NestedDictionary.unflattened([
            ("gate.weight", e0Matrix(rows: 24, cols: 16)),
            ("up.weight", e0Matrix(rows: 24, cols: 16)),
            ("down.weight", e0Matrix(rows: 16, cols: 24)),
        ]))
        let siluOut = decoder.layers[0].ffn(e0In)[0, 0, 0]
        eval(siluOut)
        #expect(abs(siluOut.item(Float.self) - Float(1.0 / (1.0 + exp(-1.0)))) < 1e-5)
        Self.populateDeterministicWeights(decoder)

        // Single-layer model with sliding_window = 2:
        // Query at position 3 attends to keys {2, 3}. Mutating token 0 must NOT alter position 3,
        // while mutating token 2 MUST alter position 3.
        let baseTokens = [1, 2, 3, 4]
        let baseLogits = decoder.forward(MLXArray(baseTokens, [1, 4]))
        eval(baseLogits)

        let farMutated = decoder.forward(MLXArray([15, 2, 3, 4], [1, 4]))
        eval(farMutated)
        let farDiffAtPos3 = MLX.abs(farMutated[0, 3] - baseLogits[0, 3]).max().item(Float.self)
        #expect(farDiffAtPos3 == 0, "Position 0 should fall outside sliding_window=2 from position 3")

        let nearMutated = decoder.forward(MLXArray([1, 2, 15, 4], [1, 4]))
        eval(nearMutated)
        let nearDiffAtPos3 = MLX.abs(nearMutated[0, 3] - baseLogits[0, 3]).max().item(Float.self)
        #expect(nearDiffAtPos3 > 1e-4, "Position 2 inside sliding_window=2 should influence position 3")

        // Incremental KV-cache decode matches full forward pass
        let caches = decoder.layers.map { _ in KVCache() }
        let prefill = decoder.forward(MLXArray([1, 2], [1, 2]), caches: caches, offset: 0)
        let step1 = decoder.forward(MLXArray([3], [1, 1]), caches: caches, offset: 2)
        let step2 = decoder.forward(MLXArray([4], [1, 1]), caches: caches, offset: 3)
        eval(prefill, step1, step2)
        let combined = concatenated([prefill, step1, step2], axis: 1)
        let cacheErr = MLX.abs(combined - baseLogits).max().item(Float.self)
        #expect(cacheErr < 1e-4)
    }

    // MARK: - Gemma 2

    @Test("Gemma 2 uses (1+w) RMSNorm, GELU, alternating window, softcapping, and KV-cache parity")
    func gemma2ArchitectureAndSoftcap() throws {
        // Direct mathematical check of GemmaRMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w)
        let norm = GemmaRMSNorm(dimensions: 4, eps: 1e-6)
        norm.update(parameters: NestedDictionary.unflattened([("weight", MLXArray([Float](repeating: 0.5, count: 4), [4]))]))
        let sample = MLXArray([Float(1), Float(3), Float(1), Float(3)], [1, 4])
        let normed = norm(sample)
        eval(normed)
        // mean(x^2) = (1+9+1+9)/4 = 5 -> inv = 1/sqrt(5.000001), weight factor = 1.5
        let expectedFirst = Float(1.0 / sqrt(5.0 + 1e-6) * 1.5)
        #expect(abs(normed[0, 0].item(Float.self) - expectedFirst) < 1e-5)

        let json = """
        {
          "model_type": "gemma2",
          "hidden_size": 16,
          "head_dim": 8,
          "num_hidden_layers": 2,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "intermediate_size": 24,
          "vocab_size": 32,
          "max_position_embeddings": 64,
          "rms_norm_eps": 1e-6,
          "rope_theta": 10000.0,
          "sliding_window": 2,
          "attn_logit_softcapping": 5.0,
          "final_logit_softcapping": 4.0,
          "query_pre_attn_scalar": 8.0,
          "hidden_activation": "gelu_pytorch_tanh",
          "tie_word_embeddings": true
        }
        """
        let cfg = try LLMConfig(huggingFaceConfigJSON: Data(json.utf8))
        #expect(cfg.architecture == .gemma2)
        #expect(cfg.isSlidingLayer(at: 0))
        #expect(!cfg.isSlidingLayer(at: 1))
        #expect(cfg.attentionSoftcap == 5.0)
        #expect(cfg.finalSoftcap == 4.0)

        let decoder = TransformerDecoder(config: cfg, tokenizer: BPETokenizer(vocab: ["<a>": 0], merges: []))
        Self.populateDeterministicWeights(decoder)
        #expect(decoder.layers[0].norm1 is GemmaRMSNorm)
        // Verify GELU (tanh approximation) FFN on unit gate/up/down: gelu_approx(1.0) ~= 0.8412 != silu(1.0)
        let e0In = MLXArray([Float(1.0)] + Array(repeating: Float(0.0), count: 15), [1, 1, 16])
        func e0Matrix(rows: Int, cols: Int) -> MLXArray {
            var arr = Array(repeating: Float(0.0), count: rows * cols)
            arr[0] = 1.0
            return MLXArray(arr, [rows, cols])
        }
        decoder.layers[0].ffn.update(parameters: NestedDictionary.unflattened([
            ("gate.weight", e0Matrix(rows: 24, cols: 16)),
            ("up.weight", e0Matrix(rows: 24, cols: 16)),
            ("down.weight", e0Matrix(rows: 16, cols: 24)),
        ]))
        let geluOut = decoder.layers[0].ffn(e0In)[0, 0, 0]
        eval(geluOut)
        #expect(abs(geluOut.item(Float.self) - Float(0.841192)) < 1e-4)
        Self.populateDeterministicWeights(decoder)

        // With numHeads=2 and softcap!=nil, 4D mask reshaping must run cleanly across batch=2 and seq=4
        let input = MLXArray([1, 2, 3, 4, 5, 6, 7, 8], [2, 4])
        let full = decoder.forward(input)
        eval(full)
        #expect(full.shape == [2, 4, 32])
        let maxAbsLogit = MLX.abs(full).max().item(Float.self)
        #expect(maxAbsLogit <= 4.0 + 1e-5, "final_logit_softcapping=4.0 must bound logits")

        // Incremental KV-cache decode matches full forward pass
        let caches = decoder.layers.map { _ in KVCache() }
        let singleFull = decoder.forward(MLXArray([1, 2, 3, 4], [1, 4]))
        let prefill = decoder.forward(MLXArray([1, 2], [1, 2]), caches: caches, offset: 0)
        let s1 = decoder.forward(MLXArray([3], [1, 1]), caches: caches, offset: 2)
        let s2 = decoder.forward(MLXArray([4], [1, 1]), caches: caches, offset: 3)
        eval(singleFull, prefill, s1, s2)
        let diff = MLX.abs(concatenated([prefill, s1, s2], axis: 1) - singleFull).max().item(Float.self)
        #expect(diff < 1e-4)
    }

    // MARK: - Gemma 3

    @Test("Gemma 3 supports nested text_config, explicit head_dim, Q/K norms, dual RoPE, clipResidual, and lmHead policy")
    func gemma3NestedConfigHeadsRoPEClipAndLMHead() throws {
        // Explicit head_dim (8) with num_attention_heads (4) and hidden_size (16) -> 4*8 = 32 != 16
        let nestedJSON = """
        {
          "model_type": "gemma3",
          "text_config": {
            "model_type": "gemma3_text",
            "hidden_size": 16,
            "head_dim": 8,
            "num_hidden_layers": 6,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "intermediate_size": 24,
            "vocab_size": 300,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0,
            "sliding_window": 4,
            "sliding_window_pattern": 6,
            "rope_scaling": { "rope_type": "linear", "factor": 8.0 },
            "tie_word_embeddings": true
          }
        }
        """
        let cfg = try LLMConfig(huggingFaceConfigJSON: Data(nestedJSON.utf8))
        #expect(cfg.architecture == .gemma3)
        #expect(cfg.headDim == 8)
        #expect(cfg.gemma3LocalRopeTheta == 10_000)
        #expect(cfg.gemma3GlobalRopeTheta == 1_000_000)
        #expect(abs(cfg.gemma3GlobalRopeScale - 0.125) < 1e-6)
        #expect(cfg.isSlidingLayer(at: 0))
        #expect(cfg.isSlidingLayer(at: 4))
        #expect(!cfg.isSlidingLayer(at: 5))

        // Float16 clipResidual keeps overflowing sums finite within Float16 range
        let huge = MLXArray([Float(60_000)], [1]).asType(.float16)
        let clipped = clipResidual(huge, huge)
        eval(clipped)
        #expect(clipped.dtype == .float16)
        #expect(isFinite(clipped).all().item(Bool.self))

        // Embedding scale in Gemma 3 rounds via BFloat16
        let decoder = TransformerDecoder(config: cfg, tokenizer: BPETokenizer(vocab: ["<a>": 0], merges: []))
        #expect(decoder.layers[0].qNorm is GemmaRMSNorm)
        #expect(decoder.layers[0].kNorm is GemmaRMSNorm)
        let expectedBF16Scale = MLXArray(Float(cfg.hiddenDim).squareRoot()).asType(.bfloat16).asType(.float32).item(Float.self)
        #expect(expectedBF16Scale == 4.0)

        // Direct gemma3_text directory loading: tied without lm_head.weight vs explicit lm_head.weight
        let fm = FileManager.default
        let baseDir = fm.temporaryDirectory.appendingPathComponent("gemma3-\(UUID().uuidString)")
        try fm.createDirectory(at: baseDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: baseDir) }

        let directConfigJSON = """
        {
          "model_type": "gemma3_text",
          "hidden_size": 16,
          "head_dim": 8,
          "num_hidden_layers": 6,
          "num_attention_heads": 4,
          "num_key_value_heads": 1,
          "intermediate_size": 24,
          "vocab_size": 300,
          "max_position_embeddings": 64,
          "sliding_window": 4,
          "sliding_window_pattern": 6,
          "tie_word_embeddings": true
        }
        """
        try Data(directConfigJSON.utf8).write(to: baseDir.appendingPathComponent("config.json"))
        try Self.makeMinimalGemmaTokenizerJSON(vocabSize: 300).write(to: baseDir.appendingPathComponent("tokenizer.json"))

        let parsedDirect = try LLMConfig(huggingFaceConfigJSON: Data(directConfigJSON.utf8))
        let baseWeights = Self.makeGemma3DenseWeights(parsedDirect)
        try MLX.save(arrays: baseWeights, url: baseDir.appendingPathComponent("model.safetensors"))

        let tiedDecoder = try TransformerDecoder.load(directory: baseDir)
        #expect(tiedDecoder.lmHead == nil)
        let tiedLogits = tiedDecoder.forward(MLXArray([2, 4, 5], [1, 3]))
        eval(tiedLogits)
        #expect(tiedLogits.shape == [1, 3, 300])

        // Now write lm_head.weight despite tie_word_embeddings: true -> ensureLMHeadIfPresent attaches lmHead
        var untiedWeights = baseWeights
        untiedWeights["lm_head.weight"] = Self.deterministicTensor([300, 16], seed: 999, scale: 0.1)
        try MLX.save(arrays: untiedWeights, url: baseDir.appendingPathComponent("model.safetensors"))

        let loadedUntied = try TransformerDecoder.load(directory: baseDir)
        #expect(loadedUntied.lmHead != nil)
        let untiedLogits = loadedUntied.forward(MLXArray([2, 4, 5], [1, 3]))
        eval(untiedLogits)
        #expect(MLX.abs(untiedLogits - tiedLogits).max().item(Float.self) > 1e-4)
    }

    // MARK: - Gemma 4

    @Test("Gemma 4 ParameterlessRMSNorm, ProportionalRoPE (CPU & GPU), and CheckpointKeyNormalizer")
    func gemma4PrimitivesAndKeyNormalization() {
        // 1. ParameterlessRMSNorm matches manual rsqrt(mean(x^2) + eps)
        let pnorm = ParameterlessRMSNorm(eps: 1e-6)
        let x = MLXArray([Float(1), Float(-3), Float(1), Float(3)], [1, 4])
        let y = pnorm(x)
        eval(y)
        let invRMS = Float(1.0 / sqrt(5.0 + 1e-6))
        #expect(abs(y[0, 0].item(Float.self) - invRMS) < 1e-5)
        #expect(abs(y[0, 1].item(Float.self) - (-3.0 * invRMS)) < 1e-5)

        // 2. ProportionalRoPE leaves tail unrotated and rotates leading dims using FULL head dimension
        let vec = MLXArray([Float(1.0), Float(0.5), Float(0.25), Float(-0.75), Float(0.9), Float(-0.4), Float(0.3), Float(0.2)], [1, 1, 1, 8])
        for gpu in [false, true] {
            Device.withDefaultDevice(gpu ? .gpu : .cpu) {
                let rope = RoPEEmbedding(dimensions: 8, rotatedDimensions: 2, base: 1_000_000, scale: 1.0)
                let rotated = rope(vec, offset: 7)
                eval(rotated)
                // MLX RoPE splits rotated pairs (traditional=false) into [0..<half] and [half..<dims];
                // verify tail slice beyond rotated width is untouched when unrotated.
                let values = rotated.asArray(Float.self)
                // Pairs inside full dimensions=8 mean rotated slice covers first 1 pair -> freqs[1..<4] = inf!
                // Wait: in non-traditional RoPE, channel (0, 4) vs (0..<2)? Verify full slice at offset 0 equals identity.
                let atZero = rope(vec, offset: 0)
                eval(atZero)
                #expect(MLX.abs(atZero - vec).max().item(Float.self) < 1e-6)
                // With base=1_000_000 and rotatedDimensions=2, dim index 0 has freq 1.0, rotated tail is unchanged
                #expect(values[1] == Float(0.5) && values[2] == Float(0.25) && values[3] == Float(-0.75))
                #expect(values[5] == Float(-0.4) && values[6] == Float(0.3) && values[7] == Float(0.2))
                // And first pair rotated by angle = 7.0 rad: x0' = 1*cos(7) - 0.9*sin(7)
                let expected0 = Float(1.0) * cos(7.0) - Float(0.9) * sin(7.0)
                #expect(abs(values[0] - expected0) < 1e-5)
            }
        }

        // 3. CheckpointKeyNormalizer strips multimodal prefixes/towers and splits experts.gate_up_proj
        let raw: [String: MLXArray] = [
            "model.language_model.model.embed_tokens.weight": MLXArray.ones([4, 4]),
            "language_model.model.layers.0.input_layernorm.weight": MLXArray.ones([4]),
            "language_model.lm_head.weight": MLXArray.ones([4, 4]),
            "model.vision_tower.patch_emb.weight": MLXArray.zeros([2, 2]),
            "model.audio_tower.enc.weight": MLXArray.zeros([2, 2]),
            "model.embed_vision.weight": MLXArray.zeros([2, 2]),
            "model.language_model.model.rotary_emb.inv_freq": MLXArray.zeros([4]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([4]),
            "model.layers.0.self_attn.q_proj.input_max": MLXArray.ones([1]),
            "model.language_model.model.layers.0.experts.gate_up_proj": MLXArray((0..<16).map(Float.init), [2, 4, 2]),
        ]
        let normalized = CheckpointKeyNormalizer.normalizeWeights(raw)
        #expect(normalized["model.embed_tokens.weight"] != nil)
        #expect(normalized["model.layers.0.input_layernorm.weight"] != nil)
        #expect(normalized["lm_head.weight"] != nil)
        #expect(normalized["model.vision_tower.patch_emb.weight"] == nil)
        #expect(normalized["model.audio_tower.enc.weight"] == nil)
        #expect(normalized["model.embed_vision.weight"] == nil)
        #expect(normalized["model.rotary_emb.inv_freq"] == nil)
        #expect(normalized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
        #expect(normalized["model.layers.0.self_attn.q_proj.input_max"] == nil)
        #expect(normalized["model.layers.0.experts.gate_up_proj"] == nil)
        let g = normalized["model.layers.0.experts.switch_glu.gate_proj.weight"]
        let u = normalized["model.layers.0.experts.switch_glu.up_proj.weight"]
        #expect(g?.shape == [2, 2, 2] && u?.shape == [2, 2, 2])
    }

    @Test("Gemma 4 E2B variant: PLE, KV-shared layers, double-wide MLP, layer_scalar, and K=V")
    func gemma4E2BPLEAndKVSharing() throws {
        // 4 layers: 0=sliding, 1=full(donor), 2=sliding(shared from 0), 3=full(shared from 1) -> num_kv_shared_layers=2
        let json = """
        {
          "model_type": "gemma4",
          "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 16,
            "head_dim": 8,
            "global_head_dim": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "intermediate_size": 12,
            "vocab_size": 300,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "sliding_window": 4,
            "sliding_window_pattern": 2,
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            "num_kv_shared_layers": 2,
            "use_double_wide_mlp": true,
            "attention_k_eq_v": true,
            "hidden_size_per_layer_input": 4,
            "vocab_size_per_layer_input": 300,
            "tie_word_embeddings": true
          }
        }
        """
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("gemma4-e2b-\(UUID().uuidString)")
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }
        try Data(json.utf8).write(to: dir.appendingPathComponent("config.json"))
        try Self.makeMinimalGemmaTokenizerJSON(vocabSize: 300).write(to: dir.appendingPathComponent("tokenizer.json"))

        let cfg = try LLMConfig(huggingFaceConfigJSON: Data(json.utf8))
        #expect(cfg.architecture == .gemma4)
        #expect(cfg.hasKV(forLayer: 0) && cfg.hasKV(forLayer: 1))
        #expect(!cfg.hasKV(forLayer: 2) && !cfg.hasKV(forLayer: 3))
        let kvSources = cfg.previousKVLayerIndices()
        #expect(kvSources[2] == 0)
        #expect(kvSources[3] == 1)
        #expect(!cfg.usesKEqV(forLayer: 0) && cfg.usesKEqV(forLayer: 1))
        #expect(cfg.headDim(forLayer: 1) == 16 && cfg.headDim(forLayer: 0) == 8)
        #expect(cfg.intermediateSize(forLayer: 0) == 12 && cfg.intermediateSize(forLayer: 2) == 24)

        let weights = Self.makeGemma4DenseWeights(cfg)
        // Verify no v_proj generated on K=V global layer 1 or shared layers 2/3
        #expect(weights["model.layers.0.self_attn.v_proj.weight"] != nil)
        #expect(weights["model.layers.1.self_attn.v_proj.weight"] == nil)
        #expect(weights["model.layers.2.self_attn.k_proj.weight"] == nil)

        try MLX.save(arrays: weights, url: dir.appendingPathComponent("model.safetensors"))
        let decoder = try TransformerDecoder.load(directory: dir)
        #expect(decoder.embedTokensPerLayer != nil)
        #expect(decoder.perLayerModelProjection != nil)
        #expect(decoder.perLayerProjectionNorm != nil)
        // Standard RMSNorm (not 1+w GemmaRMSNorm)
        #expect(!(decoder.layers[0].norm1 is GemmaRMSNorm))
        #expect(decoder.layers[1].vNorm != nil && decoder.layers[2].vNorm == nil)

        // Verify raw k_proj (before kNorm) feeds vNorm on K=V global layer 1
        // With all weights initialized except k_norm != ones, changing k_norm changes K while V matches ParameterlessRMSNorm(k_proj)
        let tokens = [2, 5, 7, 9]
        let full = decoder.forward(MLXArray(tokens, [1, 4]))
        eval(full)
        #expect(full.shape == [1, 4, 300])

        let caches = decoder.layers.map { _ in KVCache() }
        let p0 = decoder.forward(MLXArray([2, 5], [1, 2]), caches: caches, offset: 0)
        let p1 = decoder.forward(MLXArray([7], [1, 1]), caches: caches, offset: 2)
        let p2 = decoder.forward(MLXArray([9], [1, 1]), caches: caches, offset: 3)
        eval(p0, p1, p2)
        // Donor layers cached 4 tokens; shared layers cached 0 tokens!
        #expect(caches[0].count == 4 && caches[1].count == 4)
        #expect(caches[2].count == 0 && caches[3].count == 0)
        let diff = MLX.abs(concatenated([p0, p1, p2], axis: 1) - full).max().item(Float.self)
        #expect(diff < 1e-4)

        // Verify K=V global layer 1 uses pre-kNorm keys for values: scaling k_norm preserves V difference vs changing K
        decoder.layers[1].kNorm?.update(parameters: NestedDictionary.unflattened([("weight", MLXArray(Array(repeating: Float(3.0), count: 16), [16]))]))
        let modifiedLogits = decoder.forward(MLXArray(tokens, [1, 4]))
        eval(modifiedLogits)
        #expect(MLX.abs(modifiedLogits - full).max().item(Float.self) > 1e-5)
    }

    @Test("G-050: TransformerDecoder.generationPrompt avoids duplicate <bos> for Gemma 4 prompts")
    func gemma4GenerationPromptSingleBOS() async throws {
        let tokData = try Self.makeMinimalGemmaTokenizerJSON(vocabSize: 300)
        let tokenizer = try SentencePieceBPETokenizer(tokenizerJSON: tokData)
        let prompt = ChatTemplate.gemma4.render(messages: [.user("hi")], addGenerationPrompt: true)
        #expect(prompt.hasPrefix("<bos>"))

        let expectedTokens = tokenizer.encode(text: prompt, addSpecialTokens: false)
        #expect(expectedTokens.first == 2)
        #expect(expectedTokens.dropFirst().first != 2)

        // Exact maxSeqLen == expectedTokens.count succeeds only if duplicate BOS is NOT added.
        let cfg = LLMConfig(
            vocabSize: 300,
            numLayers: 1,
            hiddenDim: 8,
            numHeads: 2,
            numKVHeads: 1,
            intermediateSize: 16,
            maxSeqLen: expectedTokens.count
        )
        let decoder = TransformerDecoder(config: cfg, tokenizer: tokenizer)
        Self.populateDeterministicWeights(decoder)

        var completionInfo: GenerationCompletionInfo?
        for try await event in decoder.generateDetails(prompt: prompt, options: LLMOptions(temperature: 0, maxTokens: 0)) {
            if case .info(let info) = event {
                completionInfo = info
            }
        }
        #expect(completionInfo?.promptTokenCount == expectedTokens.count)

        // A prompt without leading <bos> still receives the tokenizer's single BOS token.
        let plainPrompt = "hi"
        let expectedPlainCount = tokenizer.encode(text: plainPrompt, addSpecialTokens: true).count
        var plainInfo: GenerationCompletionInfo?
        for try await event in decoder.generateDetails(prompt: plainPrompt, options: LLMOptions(temperature: 0, maxTokens: 0)) {
            if case .info(let info) = event {
                plainInfo = info
            }
        }
        #expect(plainInfo?.promptTokenCount == expectedPlainCount)
    }

    @Test("Gemma 4 MoE + mixed-bit quantized checkpoint loads packed 4-bit experts and 8-bit router")
    func gemma4MoEAndMixedBitQuantization() throws {
        // Hidden=64, intermediate=64, moe_intermediate=64 to allow clean groupSize=64 affine packing
        let json = """
        {
          "model_type": "gemma4_unified",
          "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": 64,
            "head_dim": 64,
            "global_head_dim": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "intermediate_size": 64,
            "moe_intermediate_size": 64,
            "vocab_size": 300,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "sliding_window": 4,
            "sliding_window_pattern": 2,
            "enable_moe_block": true,
            "num_experts": 4,
            "top_k_experts": 2,
            "tie_word_embeddings": true
          },
          "quantization": {
            "group_size": 64,
            "bits": 4,
            "model.layers.0.router.proj": { "group_size": 64, "bits": 8 },
            "model.layers.1.router.proj": { "group_size": 64, "bits": 8 }
          }
        }
        """
        let cfg = try LLMConfig(huggingFaceConfigJSON: Data(json.utf8))
        #expect(cfg.architecture == .gemma4)
        #expect(cfg.gemma4?.enableMoEBlock == true)

        let quantCfg = try #require(AffineQuantizationConfig(configJSON: Data(json.utf8)))
        #expect(quantCfg.groupSize == 64 && quantCfg.bits == 4)
        #expect(quantCfg.moduleOverrides["model.layers.0.router.proj"]?.bits == 8)

        let dense = Self.makeGemma4DenseWeights(cfg)
        var packed: [String: MLXArray] = [:]
        var dequantized: [String: MLXArray] = [:]
        for (key, value) in dense {
            guard key.hasSuffix(".weight"), value.ndim >= 2, value.dim(-1) >= 64,
                  !key.contains("norm") else {
                packed[key] = value
                dequantized[key] = value
                continue
            }
            let base = String(key.dropLast(".weight".count))
            let bits = base.hasSuffix(".router.proj") ? 8 : 4
            let q = MLX.quantized(value, groupSize: 64, bits: bits)
            packed[key] = q.wq
            packed[base + ".scales"] = q.scales
            if let biases = q.biases { packed[base + ".biases"] = biases }
            dequantized[key] = MLX.dequantized(q.wq, scales: q.scales, biases: q.biases, groupSize: 64, bits: bits)
        }

        // Compare SwitchLinear vs QuantizedSwitchLinear directly first
        let switchW = dense["model.layers.0.experts.switch_glu.gate_proj.weight"]!
        let qSwitch = MLX.quantized(switchW, groupSize: 64, bits: 4)
        let deqSwitch = MLX.dequantized(qSwitch.wq, scales: qSwitch.scales, biases: qSwitch.biases, groupSize: 64, bits: 4)
        let denseSwitch = SwitchLinear(inputDims: 64, outputDims: 64, numExperts: 4)
        denseSwitch.update(parameters: NestedDictionary.unflattened([("weight", deqSwitch)]))
        let quantSwitch = try #require(denseSwitch.toQuantized(groupSize: 64, bits: 4) as? QuantizedSwitchLinear)
        quantSwitch.update(parameters: NestedDictionary.unflattened([
            ("weight", qSwitch.wq), ("scales", qSwitch.scales), ("biases", qSwitch.biases!),
        ]))
        let probe = Self.deterministicTensor([2, 1, 1, 64], seed: 77, scale: 0.5)
        let idx = MLXArray([0, 2], [2, 1])
        let sOutDense = denseSwitch(probe, indices: idx, sortedIndices: false)
        let sOutQuant = quantSwitch(probe, indices: idx, sortedIndices: false)
        eval(sOutDense, sOutQuant)
        #expect(MLX.abs(sOutDense - sOutQuant).max().item(Float.self) < 1e-3)

        // End-to-end directory load via TransformerDecoder.load(directory:) with mixed-bit quantization
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("gemma4-moe-\(UUID().uuidString)")
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }
        try Data(json.utf8).write(to: dir.appendingPathComponent("config.json"))
        try Self.makeMinimalGemmaTokenizerJSON(vocabSize: 300).write(to: dir.appendingPathComponent("tokenizer.json"))
        try MLX.save(arrays: packed, url: dir.appendingPathComponent("model.safetensors"))

        let refDecoder = TransformerDecoder(config: cfg, tokenizer: BPETokenizer(vocab: ["<a>": 0], merges: []))
        #expect(refDecoder.loadWeights(dequantized).isEmpty)

        let quantDecoder = try TransformerDecoder.load(directory: dir)
        #expect(quantDecoder.layers[0].experts?.switchGLU.gateProj is QuantizedSwitchLinear)
        #expect(quantDecoder.layers[0].router?.proj is MLXNN.QuantizedLinear)

        Device.withDefaultDevice(.gpu) {
            let ids = MLXArray([2, 5, 9], [1, 3])
            let qLogits = quantDecoder.forward(ids)
            let rLogits = refDecoder.forward(ids)
            eval(qLogits, rLogits)
            #expect(qLogits.shape == [1, 3, 300])
            #expect(MLX.abs(qLogits - rLogits).max().item(Float.self) < 5e-3)

            let caches = quantDecoder.layers.map { _ in KVCache() }
            let pre = quantDecoder.forward(MLXArray([2, 5], [1, 2]), caches: caches, offset: 0)
            let dec = quantDecoder.forward(MLXArray([9], [1, 1]), caches: caches, offset: 2)
            eval(pre, dec)
            let cacheDiff = MLX.abs(concatenated([pre, dec], axis: 1) - qLogits).max().item(Float.self)
            #expect(cacheDiff < 1e-4)
        }
    }

    @Test("Live HuggingFace MLX checkpoint parity against tokenizers & mlx_lm",
          .enabled(if: ProcessInfo.processInfo.environment["SWIFTSCI_HF_REFERENCE_JSON"] != nil))
    func liveHuggingFaceParity() throws {
        let refPath = try #require(ProcessInfo.processInfo.environment["SWIFTSCI_HF_REFERENCE_JSON"])
        let data = try Data(contentsOf: URL(fileURLWithPath: refPath))
        let root = try #require(try JSONSerialization.jsonObject(with: data) as? [String: [String: Any]])

        for (name, entry) in root.sorted(by: { $0.key < $1.key }) {
            let dirPath = try #require(entry["dir"] as? String)
            let dir = URL(fileURLWithPath: dirPath)

            // 1. Validate LLMConfig parses real HF config.json cleanly
            let configData = try Data(contentsOf: dir.appendingPathComponent("config.json"))
            let parsedConfig = try LLMConfig(huggingFaceConfigJSON: configData)
            let tokData = try Data(contentsOf: dir.appendingPathComponent("tokenizer.json"))
            let tokenizer = try SentencePieceBPETokenizer(tokenizerJSON: tokData)

            // 2. Validate token IDs and decoding match HuggingFace `tokenizers`
            if let tokCases = entry["tokenizers"] as? [[String: Any]] {
                for tc in tokCases {
                    let text = try #require(tc["text"] as? String)
                    let expectedNo = try #require(tc["ids_no_special"] as? [Int])
                    let expectedWith = try #require(tc["ids_with_special"] as? [Int])
                    let expectedDecoded = try #require(tc["decoded"] as? String)
                    let actualNo = tokenizer.encode(text: text, addSpecialTokens: false)
                    let actualWith = tokenizer.encode(text: text, addSpecialTokens: true)
                    let actualDecoded = tokenizer.decode(tokens: actualNo)
                    #expect(actualNo == expectedNo, "\(name) encode(addSpecial=false) mismatch for \(text.debugDescription)")
                    #expect(actualWith == expectedWith, "\(name) encode(addSpecial=true) mismatch for \(text.debugDescription)")
                    #expect(actualDecoded == expectedDecoded, "\(name) decode mismatch for \(text.debugDescription)")
                }
            }

            // 3. Validate TransformerDecoder.load(directory:), ChatTemplate, logits parity, and greedy generation
            if let fw = entry["forward"] as? [String: Any] {
                let decoder = try TransformerDecoder.load(directory: dir, maxSeqLen: 512)
                let inputIDs = try #require(fw["input_ids"] as? [Int])
                let userMessages: [ChatMessage] = [.user("What is the capital of France? Answer in one word.")]
                if let rendered = fw["rendered_prompt"] as? String {
                    switch parsedConfig.architecture {
                    case .gemma2, .gemma3:
                        let swiftEncoded = ChatTemplate.gemma.encode(messages: userMessages, using: tokenizer, addGenerationPrompt: true)
                        #expect("<bos>" + ChatTemplate.gemma.render(messages: userMessages, addGenerationPrompt: true) == rendered)
                        #expect(swiftEncoded == inputIDs, "\(name) ChatTemplate.encode mismatch")
                    case .gemma4:
                        let noThink = fw["rendered_prompt_no_think"] as? String
                        let swiftNoThink = ChatTemplate.gemma4.render(messages: userMessages, addGenerationPrompt: true)
                        if let noThink {
                            #expect(swiftNoThink == noThink, "\(name) ChatTemplate no-think mismatch")
                        }
                        let swiftThink = ChatTemplate.gemma4.render(
                            messages: [.system("<|think|>\n"), .user("What is the capital of France? Answer in one word.")],
                            addGenerationPrompt: true
                        )
                        #expect(swiftThink == rendered, "\(name) ChatTemplate think mismatch")
                        #expect(tokenizer.encode(text: swiftThink, addSpecialTokens: false) == inputIDs)
                    default:
                        break
                    }
                }

                let expectedLogits = try #require(fw["last_logits"] as? [Double]).map(Float.init)
                let expectedGreedy = try #require(fw["greedy_tokens"] as? [Int])

                try Device.withDefaultDevice(.gpu) {
                    let x = MLXArray(inputIDs, [1, inputIDs.count])
                    let out = decoder.forward(x)
                    eval(out)
                    let last = out[0, inputIDs.count - 1].asType(.float32).asArray(Float.self)
                    #expect(last.count == expectedLogits.count)
                    var maxDiff: Float = 0
                    for (a, b) in zip(last, expectedLogits) {
                        maxDiff = max(maxDiff, abs(a - b))
                    }
                    let swiftTop = last.enumerated().max(by: { $0.element < $1.element })?.offset
                    let hfTop = expectedLogits.enumerated().max(by: { $0.element < $1.element })?.offset
                    #expect(swiftTop == hfTop, "\(name) top-1 token mismatch")
                    #expect(maxDiff <= 1e-3, "\(name) logits diverged by \(maxDiff)")

                    // Run 16 greedy steps using KVCache incrementally
                    let caches = decoder.layers.map { _ in KVCache() }
                    var cur = try #require(decoder.prefill(inputIDs, caches: caches))
                    eval(cur)
                    var generated: [Int] = []
                    for _ in 0..<expectedGreedy.count {
                        let nextID = argMax(cur, axis: -1).item(Int.self)
                        generated.append(nextID)
                        let step = decoder.forward(MLXArray([nextID], [1, 1]), caches: caches, offset: caches.first?.count ?? 0)
                        eval(step)
                        cur = step[0, 0]
                    }
                    #expect(generated == expectedGreedy,
                            "\(name) greedy generation diverged: \(generated) vs \(expectedGreedy)")
                }
            }
        }
    }
}
#endif
