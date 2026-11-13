#if os(macOS)
import Testing
import Foundation
import MLX
import MLXNN
@testable import SwiftNLP
@testable import SwiftLLM

private func makeDenseWeights(_ config: LLMConfig) -> [String: MLXArray] {
    let hidden = config.hiddenDim
    let kv = hidden / config.numHeads * (config.numKVHeads ?? config.numHeads)
    func rand(_ shape: [Int]) -> MLXArray { MLXRandom.normal(shape) * 0.05 }
    var w: [String: MLXArray] = ["model.embed_tokens.weight": rand([config.vocabSize, hidden]),
                                 "model.norm.weight": MLXArray.ones([hidden])]
    if !config.tieWordEmbeddings { w["lm_head.weight"] = rand([config.vocabSize, hidden]) }
    for i in 0..<config.numLayers {
        let p = "model.layers.\(i)"
        w["\(p).self_attn.q_proj.weight"] = rand([hidden, hidden])
        w["\(p).self_attn.k_proj.weight"] = rand([kv, hidden])
        w["\(p).self_attn.v_proj.weight"] = rand([kv, hidden])
        w["\(p).self_attn.o_proj.weight"] = rand([hidden, hidden])
        w["\(p).mlp.gate_proj.weight"] = rand([config.intermediateSize, hidden])
        w["\(p).mlp.up_proj.weight"] = rand([config.intermediateSize, hidden])
        w["\(p).mlp.down_proj.weight"] = rand([hidden, config.intermediateSize])
        w["\(p).input_layernorm.weight"] = MLXArray.ones([hidden])
        w["\(p).post_attention_layernorm.weight"] = MLXArray.ones([hidden])
    }
    return w
}

/// Quantizes every projection and embedding. Returns the packed dictionary and the
/// dense equivalent obtained by dequantizing it.
private func quantize(_ dense: [String: MLXArray], groupSize: Int, bits: Int)
    -> (packed: [String: MLXArray], dequantized: [String: MLXArray]) {
    var packed: [String: MLXArray] = [:]
    var back: [String: MLXArray] = [:]
    for (key, value) in dense {
        guard key.hasSuffix(".weight"), value.ndim == 2 else { packed[key] = value; back[key] = value; continue }
        let base = String(key.dropLast(".weight".count))
        let q = MLX.quantized(value, groupSize: groupSize, bits: bits)
        packed[key] = q.wq
        packed[base + ".scales"] = q.scales
        if let biases = q.biases { packed[base + ".biases"] = biases }
        back[key] = MLX.dequantized(q.wq, scales: q.scales, biases: q.biases, groupSize: groupSize, bits: bits)
    }
    return (packed, back)
}

private func llama3Tokenizer() throws -> Data {
    let visible = Array(33...126) + Array(161...172) + Array(174...255)
    var vocab: [String: Int] = [:]
    var next = 256
    for byte in 0...255 {
        let scalar = visible.contains(byte) ? byte : next
        if !visible.contains(byte) { next += 1 }
        vocab[String(UnicodeScalar(scalar)!)] = byte
    }
    let special: [[String: Any]] = [[
        "id": 1000, "content": "<|begin_of_text|>", "special": true,
        "single_word": false, "lstrip": false, "rstrip": false, "normalized": false]]
    let regex = #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    let root: [String: Any] = [
        "version": "1.0", "normalizer": NSNull(), "truncation": NSNull(), "padding": NSNull(),
        "added_tokens": special,
        "model": ["type": "BPE", "vocab": vocab, "merges": [], "ignore_merges": true,
                  "dropout": NSNull(), "unk_token": NSNull(), "continuing_subword_prefix": NSNull(),
                  "end_of_word_suffix": NSNull(), "fuse_unk": false, "byte_fallback": false],
        "pre_tokenizer": ["type": "Sequence", "pretokenizers": [
            ["type": "Split", "pattern": ["Regex": regex], "behavior": "Isolated", "invert": false],
            ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false]]],
        "decoder": ["type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true],
        // Bare ByteLevel, as written by MLX community exports.
        "post_processor": ["type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true]
    ]
    return try JSONSerialization.data(withJSONObject: root)
}

@Suite("Native quantized checkpoints", .serialized)
struct QuantizedDecoderTests {
    private func config() -> LLMConfig { LLMConfig.debug }

    @Test("Packed 4-bit weights match the dequantized dense model without expanding layers",
          arguments: [(64, 4), (64, 8)])
    func nativeQuantizedMatchesDense(groupSize: Int, bits: Int) throws {
        MLXRandom.seed(7)
        let cfg = config()
        let (packed, dense) = quantize(makeDenseWeights(cfg), groupSize: groupSize, bits: bits)
        let tokenizer = BPETokenizer(vocab: ["a</w>": 1], merges: [])

        let reference = TransformerDecoder(config: cfg, tokenizer: tokenizer)
        #expect(reference.loadWeights(dense).isEmpty)

        let native = TransformerDecoder(config: cfg, tokenizer: tokenizer)
        // Auto-detected through the regular loader: no new call needed by existing code.
        #expect(native.loadWeights(packed).isEmpty)
        #expect(native.layers[0].ffn.gate is MLXNN.QuantizedLinear)
        #expect(native.layers[1].attention.queryProjection is MLXNN.QuantizedLinear)
        #expect(native.lmHead is MLXNN.QuantizedLinear)
        #expect(native.embedding is QuantizedEmbedding)

        let input = MLXArray([1, 2, 3, 4, 5], [1, 5])
        let a = reference(input), b = native(input)
        eval(a, b)
        #expect(a.shape == b.shape)
        let delta = abs(a - b).max().item(Float.self)
        #expect(delta < 1e-3, "max |reference - native| = \(delta)")
    }

    @Test("Quantization parameters are inferred; mismatched explicit parameters are reported")
    func inferenceAndValidation() throws {
        MLXRandom.seed(3)
        let cfg = config()
        let (packed, _) = quantize(makeDenseWeights(cfg), groupSize: 64, bits: 4)
        let tokenizer = BPETokenizer(vocab: ["a</w>": 1], merges: [])
        let decoder = TransformerDecoder(config: cfg, tokenizer: tokenizer)
        let missing = decoder.loadQuantizedWeights(packed, quantization: AffineQuantizationConfig(groupSize: 32, bits: 8))
        #expect(!missing.isEmpty)
        #expect(missing.allSatisfy { $0.hasSuffix("(incompatible quantization)") })
        #expect(decoder.loadQuantizedWeights(packed).isEmpty)

        let json = Data(#"{"quantization": {"group_size": 64, "bits": 4}}"#.utf8)
        #expect(AffineQuantizationConfig(configJSON: json) == AffineQuantizationConfig(groupSize: 64, bits: 4))
        #expect(AffineQuantizationConfig(configJSON: Data("{}".utf8)) == nil)
    }

    @Test("Dense checkpoints keep the original loading path")
    func denseUnchanged() {
        let cfg = config()
        let decoder = TransformerDecoder(config: cfg, tokenizer: BPETokenizer(vocab: ["a</w>": 1], merges: []))
        #expect(decoder.loadWeights(makeDenseWeights(cfg)).isEmpty)
        #expect(!(decoder.layers[0].ffn.gate is MLXNN.QuantizedLinear))
    }

    @Test("Sharded quantized checkpoints match single-dict loading even when weight/scales/biases split across shards")
    func crossShardSplitQuantizedPairs() throws {
        MLXRandom.seed(19)
        let cfg = config()
        let (packed, _) = quantize(makeDenseWeights(cfg), groupSize: 64, bits: 4)
        let tokenizer = BPETokenizer(vocab: ["a</w>": 1], merges: [])

        let single = TransformerDecoder(config: cfg, tokenizer: tokenizer)
        #expect(single.loadWeights(packed).isEmpty)

        // Split q_proj (`weight` in shard1, `scales` + `biases` in shard2) and
        // k_proj (`weight` + `scales` in shard1, `biases` in shard2), leaving
        // shard2 with no `.scales` only for late `.biases` coverage too.
        var shard1: [String: MLXArray] = [:]
        var shard2: [String: MLXArray] = [:]
        let half = packed.count / 2
        for (idx, (key, value)) in packed.sorted(by: { $0.key < $1.key }).enumerated() {
            if key == "model.layers.0.self_attn.q_proj.weight" {
                shard1[key] = value
            } else if key == "model.layers.0.self_attn.q_proj.scales" || key == "model.layers.0.self_attn.q_proj.biases" {
                shard2[key] = value
            } else if key == "model.layers.0.self_attn.k_proj.weight" || key == "model.layers.0.self_attn.k_proj.scales" {
                shard1[key] = value
            } else if key == "model.layers.0.self_attn.k_proj.biases" {
                shard2[key] = value
            } else if idx < half {
                shard1[key] = value
            } else {
                shard2[key] = value
            }
        }

        let sharded = TransformerDecoder(config: cfg, tokenizer: tokenizer)
        let m1 = Set(sharded.loadWeights(shard1))
        #expect(m1.contains("model.layers.0.self_attn.q_proj.weight"), "q_proj should stay missing until scales arrive")
        let m2 = Set(sharded.loadWeights(shard2))
        #expect(m1.intersection(m2).isEmpty)

        let input = MLXArray([1, 2, 3, 4], [1, 4])
        let outSingle = single(input)
        let outSharded = sharded(input)
        eval(outSingle, outSharded)
        let delta = abs(outSingle - outSharded).max().item(Float.self)
        #expect(delta == 0, "expected bit-exact sharded output, got max delta \(delta)")
    }

    @Test("Multi-layer sharded checkpoints tolerate skipped layer indices across shards")
    func sparseLayerShardsAcrossFourLayers() throws {
        MLXRandom.seed(42)
        let cfg = LLMConfig(vocabSize: 128, numLayers: 4, hiddenDim: 128, numHeads: 4, maxSeqLen: 64)
        let (packed, _) = quantize(makeDenseWeights(cfg), groupSize: 64, bits: 4)
        let tokenizer = BPETokenizer(vocab: ["a</w>": 1], merges: [])

        let single = TransformerDecoder(config: cfg, tokenizer: tokenizer)
        #expect(single.loadWeights(packed).isEmpty)

        // Shard 1 touches layers.0 (partial q_proj) and layers.3 (skipping layers.1 & 2).
        // Shard 2 touches layers.2 only (skipping index 0 completely).
        // Shard 3 finishes layers.0.q_proj.biases and layers.1.
        var shards: [[String: MLXArray]] = [[:], [:], [:]]
        for (key, value) in packed {
            if key == "model.layers.0.self_attn.q_proj.biases" || key.hasPrefix("model.layers.1.") || key == "model.norm.weight" {
                shards[2][key] = value
            } else if key.hasPrefix("model.layers.2.") {
                shards[1][key] = value
            } else {
                shards[0][key] = value
            }
        }

        let sharded = TransformerDecoder(config: cfg, tokenizer: tokenizer)
        var cumulativeMissing: Set<String>?
        for shard in shards {
            let m = Set(sharded.loadWeights(shard))
            cumulativeMissing = cumulativeMissing.map { $0.intersection(m) } ?? m
        }
        #expect(cumulativeMissing == [])

        let input = MLXArray([1, 2, 3, 4], [1, 4])
        let outSingle = single(input)
        let outSharded = sharded(input)
        eval(outSingle, outSharded)
        let delta = abs(outSingle - outSharded).max().item(Float.self)
        #expect(delta == 0, "expected bit-exact sparse-shard output, got max delta \(delta)")
    }
}

@Suite("SafeTensors dtypes and hardening")
struct SafeTensorsHardeningTests {
    private func file(header: String, payload: Data) throws -> URL {
        var data = Data()
        var length = UInt64(header.utf8.count).littleEndian
        withUnsafeBytes(of: &length) { data.append(contentsOf: $0) }
        data.append(Data(header.utf8))
        data.append(payload)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("st-\(UUID().uuidString).safetensors")
        try data.write(to: url)
        return url
    }

    @Test("Unsigned and boolean tensors parse")
    func unsignedTypes() throws {
        var payload = Data()
        payload.append(contentsOf: [1, 2, 3])                                   // U8
        func append<T>(_ values: [T]) { values.withUnsafeBytes { payload.append(contentsOf: $0) } }
        append([UInt16(4), 5])                                                  // U16
        append([UInt32(0xFFFF_FFFF), 7])                                        // U32
        append([UInt64(9), 10])                                                 // U64
        payload.append(contentsOf: [1, 0])                                      // BOOL
        let header = """
        {"a":{"dtype":"U8","shape":[3],"data_offsets":[0,3]},
         "b":{"dtype":"U16","shape":[2],"data_offsets":[3,7]},
         "c":{"dtype":"U32","shape":[2],"data_offsets":[7,15]},
         "d":{"dtype":"U64","shape":[2],"data_offsets":[15,31]},
         "e":{"dtype":"BOOL","shape":[2],"data_offsets":[31,33]}}
        """
        let url = try file(header: header, payload: payload)
        defer { try? FileManager.default.removeItem(at: url) }
        let t = try SafeTensorsParser.parse(url: url)
        #expect(t["a"]?.dtype == .uint8 && t["a"]?.asArray(UInt8.self) == [1, 2, 3])
        #expect(t["b"]?.dtype == .uint16 && t["b"]?.asArray(UInt16.self) == [4, 5])
        #expect(t["c"]?.dtype == .uint32 && t["c"]?.asArray(UInt32.self) == [0xFFFF_FFFF, 7])
        #expect(t["d"]?.dtype == .uint64 && t["d"]?.asArray(UInt64.self) == [9, 10])
        #expect(t["e"]?.dtype == .bool && t["e"]?.asArray(Bool.self) == [true, false])
    }

    @Test("Unknown dtypes, hostile header lengths, negative/overflow shapes, and truncated payloads throw instead of trapping")
    func malformed() throws {
        let unknown = try file(header: #"{"x":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]}}"#,
                               payload: Data([0]))
        let inverted = try file(header: #"{"x":{"dtype":"U8","shape":[1],"data_offsets":[4,2]}}"#,
                                payload: Data([0, 0, 0, 0]))
        let negative = try file(header: #"{"x":{"dtype":"U8","shape":[1],"data_offsets":[-1,1]}}"#,
                                payload: Data([0, 0]))
        let negativeShape = try file(header: #"{"x":{"dtype":"F32","shape":[-1,4],"data_offsets":[0,4]}}"#,
                                     payload: Data([0, 0, 0, 0]))
        let truncatedPayload = try file(header: #"{"x":{"dtype":"F32","shape":[2,2],"data_offsets":[0,8]}}"#,
                                        payload: Data(repeating: 0, count: 8))
        let overflowShape = try file(header: #"{"x":{"dtype":"F32","shape":[4611686018427387904,4],"data_offsets":[0,4]}}"#,
                                     payload: Data([0, 0, 0, 0]))
        let oversizedDimension = try file(
            header: #"{"x":{"dtype":"U8","shape":[0,2147483648],"data_offsets":[0,0]}}"#,
            payload: Data())
        var hostile = Data()
        var huge = UInt64.max.littleEndian
        withUnsafeBytes(of: &huge) { hostile.append(contentsOf: $0) }
        hostile.append(Data(repeating: 0, count: 8))
        let hostileURL = FileManager.default.temporaryDirectory.appendingPathComponent("st-\(UUID().uuidString).safetensors")
        try hostile.write(to: hostileURL)
        for url in [unknown, inverted, negative, negativeShape, truncatedPayload, overflowShape, oversizedDimension, hostileURL] {
            defer { try? FileManager.default.removeItem(at: url) }
            #expect(throws: (any Error).self) { try SafeTensorsParser.parse(url: url) }
        }
    }
}

@Suite("Zero-configuration model loading", .serialized)
struct ModelLoadingTests {
    private let configJSON = """
    {"model_type":"llama","hidden_size":128,"num_hidden_layers":2,"num_attention_heads":4,
     "num_key_value_heads":2,"intermediate_size":256,"vocab_size":1024,"max_position_embeddings":131072,
     "rms_norm_eps":1e-5,"rope_theta":500000.0,"tie_word_embeddings":true,"eos_token_id":[1,2],
     "rope_scaling":{"rope_type":"llama3","factor":32.0,"low_freq_factor":1.0,"high_freq_factor":4.0,
                     "original_max_position_embeddings":8192},
     "quantization":{"group_size":64,"bits":4}}
    """

    @Test("Hugging Face config.json maps to LLMConfig")
    func configParsing() throws {
        let cfg = try LLMConfig(huggingFaceConfigJSON: Data(configJSON.utf8))
        #expect(cfg.hiddenDim == 128 && cfg.numLayers == 2 && cfg.numHeads == 4 && cfg.numKVHeads == 2)
        #expect(cfg.intermediateSize == 256 && cfg.vocabSize == 1024)
        #expect(cfg.maxSeqLen == 8192, "context is capped to bound KV-cache memory")
        #expect(cfg.tieWordEmbeddings && cfg.eosTokenIDs == [1, 2])
        #expect(cfg.positionalEncoding == .llama3RoPE(base: 500_000, scaling: Llama3RoPEScaling(factor: 32)))
        #expect(try LLMConfig(huggingFaceConfigJSON: Data(configJSON.utf8), maxSeqLen: 512).maxSeqLen == 512)
    }

    @Test("Invalid configs throw typed errors instead of trapping")
    func configErrors() {
        for text in ["[]", #"{"hidden_size":130,"num_hidden_layers":1,"num_attention_heads":4,"intermediate_size":8,"vocab_size":8}"#,
                     #"{"hidden_size":128}"#,
                     #"{"model_type":"gpt2","hidden_size":128,"num_hidden_layers":1,"num_attention_heads":4,"intermediate_size":8,"vocab_size":8}"#,
                     #"{"hidden_size":128,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":3,"intermediate_size":8,"vocab_size":8}"#] {
            #expect(throws: LLMModelLoadingError.self) { try LLMConfig(huggingFaceConfigJSON: Data(text.utf8)) }
        }
    }

    @Test("Downloader rejects unsafe identifiers")
    func unsafeIdentifiers() async {
        let downloader = ModelDownloader(cacheDirectory: FileManager.default.temporaryDirectory)
        for repo in ["../etc", "owner/../x", "a/b/c", "onlyname", "owner/na me"] {
            await #expect(throws: LLMModelLoadingError.invalidIdentifier(repo)) {
                _ = try await downloader.download(repo: repo)
            }
        }
    }

    @Test("Download from a mirror and load a quantized checkpoint end to end")
    func downloadAndLoad() async throws {
        MLXRandom.seed(11)
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("mirror-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: root) }
        let remote = root.appendingPathComponent("remote/acme/tiny/resolve/main")
        try fm.createDirectory(at: remote, withIntermediateDirectories: true)
        try Data(configJSON.utf8).write(to: remote.appendingPathComponent("config.json"))
        try llama3Tokenizer().write(to: remote.appendingPathComponent("tokenizer.json"))

        let cfg = try LLMConfig(huggingFaceConfigJSON: Data(configJSON.utf8))
        let packed = quantize(makeDenseWeights(cfg), groupSize: 64, bits: 4).packed
        try MLX.save(arrays: packed, url: remote.appendingPathComponent("model.safetensors"))

        let downloader = ModelDownloader(endpoint: root.appendingPathComponent("remote"),
                                         cacheDirectory: root.appendingPathComponent("cache"))
        let folder = try await downloader.download(repo: "acme/tiny")
        #expect(fm.fileExists(atPath: folder.appendingPathComponent("model.safetensors").path))

        let decoder = try TransformerDecoder.load(directory: folder)
        #expect(decoder.layers[0].ffn.up is MLXNN.QuantizedLinear)
        let logits = decoder(MLXArray([1, 2, 3], [1, 3]))
        eval(logits)
        #expect(logits.shape == [1, 3, 1024])
        #expect((decoder.tokenizer as? BPETokenizer)?.encode(text: "hi", addSpecialTokens: true).first == 1000)

        // A second call reuses the cached files.
        try fm.removeItem(at: remote)
        _ = try await downloader.download(repo: "acme/tiny")
    }

    @Test("Missing affine biases or layer norms reject a checkpoint", arguments: [
        ("model.layers.0.self_attn.q_proj.biases", true),
        ("model.layers.0.input_layernorm.weight", true),
        ("model.layers.0.post_attention_layernorm.weight", true),
        ("model.layers.0.input_layernorm.weight", false),
        ("model.layers.0.post_attention_layernorm.weight", false)
    ])
    func incompleteWeights(key: String, quantized: Bool) throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("incomplete-\(UUID().uuidString)")
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }
        try Data(configJSON.utf8).write(to: dir.appendingPathComponent("config.json"))
        try llama3Tokenizer().write(to: dir.appendingPathComponent("tokenizer.json"))
        let cfg = try LLMConfig(huggingFaceConfigJSON: Data(configJSON.utf8))
        let dense = makeDenseWeights(cfg)
        var weights = quantized ? quantize(dense, groupSize: 64, bits: 4).packed : dense
        #expect(weights.removeValue(forKey: key) != nil)
        try MLX.save(arrays: weights, url: dir.appendingPathComponent("model.safetensors"))
        #expect(throws: LLMModelLoadingError.self) {
            _ = try TransformerDecoder.load(directory: dir)
        }
    }

    @Test("Different revisions have independent caches and reuse their own files")
    func revisionCaches() async throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("revisions-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: root) }
        for revision in ["version-a", "version-b"] {
            let remote = root.appendingPathComponent("remote/acme/tiny/resolve/\(revision)")
            try fm.createDirectory(at: remote, withIntermediateDirectories: true)
            for file in ["config.json", "tokenizer.json", "model.safetensors"] {
                try Data(revision.utf8).write(to: remote.appendingPathComponent(file))
            }
        }
        let downloader = ModelDownloader(endpoint: root.appendingPathComponent("remote"),
                                         cacheDirectory: root.appendingPathComponent("cache"))
        let a = try await downloader.download(repo: "acme/tiny", revision: "version-a")
        let b = try await downloader.download(repo: "acme/tiny", revision: "version-b")
        #expect(a != b)
        for (folder, revision) in [(a, "version-a"), (b, "version-b")] {
            #expect(try String(contentsOf: folder.appendingPathComponent("model.safetensors"),
                               encoding: .utf8) == revision)
        }
        try fm.removeItem(at: root.appendingPathComponent("remote"))
        // Index files are optional and probed again; file:// missing files are supported.
        #expect(try await downloader.download(repo: "acme/tiny", revision: "version-a") == a)
        #expect(try await downloader.download(repo: "acme/tiny", revision: "version-b") == b)
    }

    @Test("Loading an incomplete directory reports the missing file")
    func missingFiles() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: LLMModelLoadingError.missingFile("config.json")) {
            _ = try TransformerDecoder.load(directory: dir)
        }
    }
}
#endif
