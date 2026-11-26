#if os(macOS)
import Foundation
import MLX
import SwiftNLP

// MARK: - Errors

/// Failures while reading a checkpoint directory or downloading one.
public enum LLMModelLoadingError: Error, LocalizedError, Sendable, Equatable {
    /// A required checkpoint file is absent.
    case missingFile(String)
    /// `config.json` is malformed, incomplete, or describes an unsupported architecture.
    case invalidConfig(String)
    /// The checkpoint lacks tensors the architecture requires.
    case missingWeights([String])
    /// A repository identifier or file name is unsafe or malformed.
    case invalidIdentifier(String)
    /// The server answered with a non-success HTTP status.
    case httpStatus(Int, String)

    /// Human-readable description of the loading failure.
    public var errorDescription: String? {
        switch self {
        case .missingFile(let name): return "Checkpoint file '\(name)' is missing."
        case .invalidConfig(let reason): return "Invalid model config: \(reason)"
        case .missingWeights(let keys):
            return "Checkpoint is missing \(keys.count) tensor(s), first: \(keys.prefix(3).joined(separator: ", "))"
        case .invalidIdentifier(let id): return "Invalid identifier '\(id)'."
        case .httpStatus(let code, let name): return "HTTP \(code) while downloading '\(name)'."
        }
    }
}

// MARK: - Config parsing

extension LLMConfig {
    /// Builds a configuration from a HuggingFace Llama, Mistral, or Gemma 2 `config.json`.
    ///
    /// - Parameters:
    ///   - data: Raw `config.json` bytes.
    ///   - maxSeqLen: Context capacity to allocate. Defaults to the checkpoint's
    ///     `max_position_embeddings`, capped at 8,192 to bound KV-cache memory;
    ///     pass a value to opt into a different capacity.
    /// - Throws: ``LLMModelLoadingError/invalidConfig(_:)`` for missing or inconsistent fields.
    public init(huggingFaceConfigJSON data: Data, maxSeqLen: Int? = nil) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw LLMModelLoadingError.invalidConfig("not a JSON object")
        }
        let textConfig = (root["text_config"] as? [String: Any]) ?? root

        func number(_ dict: [String: Any], _ key: String) -> Double? {
            guard let n = dict[key] as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else { return nil }
            return n.doubleValue
        }
        func integer(_ dict: [String: Any], _ key: String) -> Int? {
            guard let d = number(dict, key), d.rounded() == d, abs(d) < 1e12 else { return nil }
            return Int(d)
        }
        func numVal(_ key: String) -> Double? { number(textConfig, key) ?? number(root, key) }
        func intVal(_ key: String) -> Int? { integer(textConfig, key) ?? integer(root, key) }
        func boolVal(_ key: String) -> Bool? { (textConfig[key] as? Bool) ?? (root[key] as? Bool) }
        func dictVal(_ key: String) -> [String: Any]? { (textConfig[key] as? [String: Any]) ?? (root[key] as? [String: Any]) }
        func positive(_ key: String, default fallback: Double) throws -> Float {
            let value = Float(numVal(key) ?? fallback)
            guard value.isFinite, value > 0 else { throw LLMModelLoadingError.invalidConfig("invalid \(key)") }
            return value
        }
        func required(_ key: String) throws -> Int {
            guard let value = intVal(key), value > 0 else {
                throw LLMModelLoadingError.invalidConfig("'\(key)' is missing or not a positive integer")
            }
            return value
        }

        let rawType = (textConfig["model_type"] as? String) ?? (root["model_type"] as? String) ?? "llama"
        let architecture: DecoderArchitecture
        switch rawType {
        case "llama": architecture = .llama
        case "mistral": architecture = .mistral
        case "gemma2": architecture = .gemma2
        case "gemma3", "gemma3_text": architecture = .gemma3
        case "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text": architecture = .gemma4
        default: throw LLMModelLoadingError.invalidConfig("unsupported model_type '\(rawType)'")
        }

        let hidden = try required("hidden_size")
        let layers = try required("num_hidden_layers")
        let heads = try required("num_attention_heads")
        let intermediate = try required("intermediate_size")
        let vocab = try required("vocab_size")
        let kvHeads = intVal("num_key_value_heads")
        let explicitHeadDim = intVal("head_dim")

        if let explicitHeadDim {
            guard explicitHeadDim > 0, explicitHeadDim <= Int(Int32.max) / heads, explicitHeadDim.isMultiple(of: 2) else {
                throw LLMModelLoadingError.invalidConfig("head_dim must be positive, even and in range")
            }
        } else {
            guard hidden % heads == 0 else {
                throw LLMModelLoadingError.invalidConfig("hidden_size must be divisible by num_attention_heads")
            }
        }
        if let kvHeads, kvHeads <= 0 || heads % kvHeads != 0 {
            throw LLMModelLoadingError.invalidConfig("num_attention_heads must be divisible by num_key_value_heads")
        }

        if architecture != .llama {
            let activation = (textConfig["hidden_activation"] as? String)
                ?? (textConfig["hidden_act"] as? String)
                ?? (root["hidden_activation"] as? String)
                ?? (root["hidden_act"] as? String)
            let isGemma = architecture == .gemma2 || architecture == .gemma3 || architecture == .gemma4
            let expected = isGemma ? "gelu_pytorch_tanh" : "silu"
            guard activation == nil || activation == expected,
                  boolVal("attention_bias") != true,
                  boolVal("mlp_bias") != true else {
                throw LLMModelLoadingError.invalidConfig("unsupported activation or biased projections")
            }
        }

        let declaredContext = intVal("max_position_embeddings") ?? 2048
        let context = maxSeqLen ?? min(declaredContext, 8192)
        guard context > 0 else { throw LLMModelLoadingError.invalidConfig("maxSeqLen must be positive") }

        let defaultTheta: Double = architecture == .gemma3 ? 1_000_000 : 10_000
        let theta = Float(numVal("rope_theta") ?? defaultTheta)
        var encoding = PositionalEncodingScheme.rope(base: theta)
        var parsedLinearScale: Float?
        if let scaling = dictVal("rope_scaling") {
            let kind = (scaling["rope_type"] ?? scaling["type"]) as? String
            if kind == "llama3" {
                guard let factor = number(scaling, "factor"), factor.isFinite, factor >= 1 else {
                    throw LLMModelLoadingError.invalidConfig("rope_scaling.factor must be at least 1")
                }
                let low = Float(number(scaling, "low_freq_factor") ?? 1)
                let high = Float(number(scaling, "high_freq_factor") ?? 4)
                guard low > 0, high > low, low.isFinite, high.isFinite else {
                    throw LLMModelLoadingError.invalidConfig("rope_scaling frequency factors must be positive and increasing")
                }
                let original = integer(scaling, "original_max_position_embeddings") ?? 8192
                guard original > 0 else {
                    throw LLMModelLoadingError.invalidConfig("rope_scaling.original_max_position_embeddings must be positive")
                }
                encoding = .llama3RoPE(base: theta, scaling: Llama3RoPEScaling(
                    factor: Float(factor), lowFrequencyFactor: low, highFrequencyFactor: high,
                    originalContextLength: original))
            } else if kind == "linear" {
                guard let factor = number(scaling, "factor"), factor.isFinite, factor >= 1 else {
                    throw LLMModelLoadingError.invalidConfig("rope_scaling.factor must be at least 1")
                }
                parsedLinearScale = Float(1.0 / factor)
            } else if let kind, kind != "default" {
                throw LLMModelLoadingError.invalidConfig("unsupported rope_scaling type '\(kind)'")
            }
        }

        var eos = Set<Int>()
        for container in [textConfig["eos_token_id"], root["eos_token_id"]] {
            eos.formUnion(Self.parseEOSTokenIDs(from: container))
        }

        let isGemmaFamily = architecture == .gemma2 || architecture == .gemma3 || architecture == .gemma4
        let defaultEps: Double = isGemmaFamily ? 1e-6 : 1e-5
        let defaultHeadDim: Int? = isGemmaFamily ? 256 : nil
        let resolvedHeadDim = explicitHeadDim ?? defaultHeadDim

        self.init(
            vocabSize: vocab,
            numLayers: layers,
            hiddenDim: hidden,
            numHeads: heads,
            numKVHeads: kvHeads,
            intermediateSize: intermediate,
            maxSeqLen: context,
            rmsNormEps: Float(numVal("rms_norm_eps") ?? defaultEps),
            positionalEncoding: encoding,
            tieWordEmbeddings: boolVal("tie_word_embeddings") ?? isGemmaFamily,
            eosTokenIDs: eos,
            checkpointHeadDim: resolvedHeadDim
        )
        self.architecture = architecture
        self.linearRoPEScale = parsedLinearScale

        if architecture != .llama, let window = intVal("sliding_window") {
            guard window > 0 else { throw LLMModelLoadingError.invalidConfig("sliding_window must be positive") }
            self.slidingWindow = window
        }

        switch architecture {
        case .llama, .mistral:
            break
        case .gemma2:
            self.slidingWindow = slidingWindow ?? 4096
            self.attentionSoftcap = try positive("attn_logit_softcapping", default: 50)
            self.finalSoftcap = try positive("final_logit_softcapping", default: 30)
            self.queryScale = 1 / (try positive("query_pre_attn_scalar", default: 256)).squareRoot()
        case .gemma3:
            let pattern = intVal("sliding_window_pattern") ?? intVal("_sliding_window_pattern") ?? 6
            guard pattern > 0 else { throw LLMModelLoadingError.invalidConfig("sliding_window_pattern must be positive") }
            self.slidingWindowPattern = pattern
            self.slidingWindow = slidingWindow ?? 512
            self.queryScale = 1 / (try positive("query_pre_attn_scalar", default: 256)).squareRoot()
            let ropeParams = dictVal("rope_parameters")
            let slidingDict = ropeParams?["sliding_attention"] as? [String: Any]
            let fullDict = ropeParams?["full_attention"] as? [String: Any]
            self.gemma3LocalRopeTheta = Float(
                (slidingDict.flatMap { number($0, "rope_theta") })
                ?? numVal("rope_local_base_freq")
                ?? 10_000
            )
            self.gemma3GlobalRopeTheta = Float(
                (fullDict.flatMap { number($0, "rope_theta") })
                ?? Double(theta)
            )
            if let fullFactor = fullDict.flatMap({ number($0, "factor") }), fullFactor.isFinite, fullFactor >= 1 {
                self.gemma3GlobalRopeScale = Float(1.0 / fullFactor)
            } else {
                self.gemma3GlobalRopeScale = parsedLinearScale ?? 1.0
            }
        case .gemma4:
            let pattern = intVal("sliding_window_pattern") ?? 5
            guard pattern > 0 else { throw LLMModelLoadingError.invalidConfig("sliding_window_pattern must be positive") }
            self.slidingWindowPattern = pattern
            self.slidingWindow = slidingWindow ?? 512
            self.finalSoftcap = try positive("final_logit_softcapping", default: 30)

            let globalHeadDim = intVal("global_head_dim") ?? (resolvedHeadDim ?? 512)
            guard globalHeadDim > 0, globalHeadDim.isMultiple(of: 2) else {
                throw LLMModelLoadingError.invalidConfig("global_head_dim must be positive and even")
            }
            let attentionKEqV = boolVal("attention_k_eq_v") ?? false
            let numGlobalKVHeads = intVal("num_global_key_value_heads")
            if let gkv = numGlobalKVHeads {
                guard gkv > 0, heads % gkv == 0 else {
                    throw LLMModelLoadingError.invalidConfig("num_attention_heads must be divisible by num_global_key_value_heads")
                }
            }
            let numKVShared = intVal("num_kv_shared_layers") ?? 0
            guard numKVShared >= 0, numKVShared < layers else {
                throw LLMModelLoadingError.invalidConfig("num_kv_shared_layers out of range")
            }
            let enableMoE = boolVal("enable_moe_block") ?? false
            let numExperts = intVal("num_experts") ?? 0
            let topKExperts = intVal("top_k_experts") ?? 0
            if enableMoE {
                guard numExperts > 0, topKExperts > 0, topKExperts <= numExperts else {
                    throw LLMModelLoadingError.invalidConfig("invalid Gemma 4 MoE expert configuration")
                }
            }

            let layerTypes: [String]
            if let explicit = textConfig["layer_types"] as? [String], explicit.count == layers {
                layerTypes = explicit
            } else {
                layerTypes = (0..<layers).map { idx in
                    (idx + 1) % pattern == 0 ? "full_attention" : "sliding_attention"
                }
            }

            let ropeParams = dictVal("rope_parameters") ?? [:]
            func parseRoPE(_ key: String, defaultTheta: Float, defaultType: String, defaultPartial: Float) throws -> Gemma4Config.RoPEParameters {
                let d = ropeParams[key] as? [String: Any] ?? [:]
                let rTheta = Float(number(d, "rope_theta") ?? Double(defaultTheta))
                let rType = (d["rope_type"] as? String) ?? defaultType
                let rPartial = Float(number(d, "partial_rotary_factor") ?? Double(defaultPartial))
                let factor = number(d, "factor") ?? 1.0
                guard factor.isFinite, factor >= 1 else {
                    throw LLMModelLoadingError.invalidConfig("rope_parameters.\(key).factor must be at least 1")
                }
                return Gemma4Config.RoPEParameters(
                    ropeTheta: rTheta,
                    ropeType: rType,
                    partialRotaryFactor: rPartial,
                    scale: Float(1.0 / factor)
                )
            }
            let defaultPartial = Float(numVal("global_partial_rotary_factor") ?? 0.25)
            let slidingRoPE = try parseRoPE("sliding_attention", defaultTheta: 10_000, defaultType: "default", defaultPartial: 1.0)
            let fullRoPE = try parseRoPE("full_attention", defaultTheta: 1_000_000, defaultType: "proportional", defaultPartial: defaultPartial)

            self.gemma4 = Gemma4Config(
                layerTypes: layerTypes,
                globalHeadDim: globalHeadDim,
                attentionKEqV: attentionKEqV,
                numGlobalKVHeads: numGlobalKVHeads,
                numKVSharedLayers: numKVShared,
                useDoubleWideMLP: boolVal("use_double_wide_mlp") ?? false,
                hiddenSizePerLayerInput: intVal("hidden_size_per_layer_input") ?? 0,
                vocabSizePerLayerInput: intVal("vocab_size_per_layer_input") ?? vocab,
                enableMoEBlock: enableMoE,
                numExperts: numExperts,
                topKExperts: topKExperts,
                moeIntermediateSize: intVal("moe_intermediate_size") ?? intermediate,
                slidingRoPE: slidingRoPE,
                fullRoPE: fullRoPE
            )
        }
    }

    fileprivate static func parseEOSTokenIDs(from container: Any?) -> Set<Int> {
        var eos = Set<Int>()
        if let num = container as? NSNumber, CFGetTypeID(num) != CFBooleanGetTypeID() {
            let d = num.doubleValue
            if d.rounded() == d, abs(d) < 1e12 {
                eos.insert(Int(d))
            }
        } else if let many = container as? [Any] {
            for case let value as NSNumber in many where CFGetTypeID(value) != CFBooleanGetTypeID() {
                let d = value.doubleValue
                if d.rounded() == d, abs(d) < 1e12 {
                    eos.insert(Int(d))
                }
            }
        }
        return eos
    }
}

// MARK: - Directory loading

extension TransformerDecoder {
    /// Loads a complete HuggingFace-style Llama, Mistral, Gemma 2, Gemma 3, or Gemma 4 checkpoint directory.
    ///
    /// The directory must contain `config.json`, `tokenizer.json`, and either
    /// `model.safetensors` or the shards listed by `model.safetensors.index.json`.
    /// MLX affine-quantized checkpoints stay packed in memory (see
    /// ``loadQuantizedWeights(_:quantization:)``), so no external inference server is needed.
    ///
    /// - Parameters:
    ///   - directory: Folder holding the checkpoint, for example from ``ModelDownloader``.
    ///   - maxSeqLen: Optional context capacity override; see ``LLMConfig/init(huggingFaceConfigJSON:maxSeqLen:)``.
    /// - Throws: ``LLMModelLoadingError`` or the underlying parser/tokenizer error.
    /// - Returns: A decoder with all weights evaluated and ready for generation.
    public static func load(directory: URL, maxSeqLen: Int? = nil) throws -> TransformerDecoder {
        let fm = FileManager.default
        func read(_ name: String) throws -> Data {
            let url = directory.appendingPathComponent(name)
            guard fm.fileExists(atPath: url.path) else { throw LLMModelLoadingError.missingFile(name) }
            return try Data(contentsOf: url)
        }
        let configData = try read("config.json")
        var config = try LLMConfig(huggingFaceConfigJSON: configData, maxSeqLen: maxSeqLen)
        let generationConfigURL = directory.appendingPathComponent("generation_config.json")
        if fm.fileExists(atPath: generationConfigURL.path),
           let generationConfigData = try? Data(contentsOf: generationConfigURL),
           let generationConfigRoot = try? JSONSerialization.jsonObject(with: generationConfigData) as? [String: Any] {
            config.eosTokenIDs.formUnion(LLMConfig.parseEOSTokenIDs(from: generationConfigRoot["eos_token_id"]))
        }
        let tokenizer: any Tokenizer
        if config.architecture == .llama {
            tokenizer = try BPETokenizer(llama3TokenizerJSON: read("tokenizer.json"))
        } else {
            tokenizer = try SentencePieceBPETokenizer(tokenizerJSON: read("tokenizer.json"))
        }
        let shards = try ModelDownloader.shardNames(in: directory)
        let quantization = AffineQuantizationConfig(configJSON: configData)

        let decoder = TransformerDecoder(config: config, tokenizer: tokenizer)
        var missingAcrossShards: Set<String>?
        var missingNorms = Set(decoder.expectedDenseParameterKeys)
        for shard in shards {
            let rawShard = try SafeTensorsParser.parse(url: directory.appendingPathComponent(shard))
            let shardWeights = CheckpointKeyNormalizer.normalizeWeights(rawShard)
            missingNorms.subtract(shardWeights.keys)
            let missing = if let quantization {
                decoder.loadQuantizedWeights(shardWeights, quantization: quantization)
            } else {
                decoder.loadWeights(shardWeights)
            }
            let shardMissing = Set(missing)
            missingAcrossShards = missingAcrossShards.map { $0.intersection(shardMissing) } ?? shardMissing
        }
        let missing = (missingAcrossShards.map { $0.union(missingNorms).sorted() }) ?? ["model.safetensors"]
        guard missingAcrossShards != nil, missing.isEmpty, !decoder.hasPendingQuantizedTensors else {
            throw LLMModelLoadingError.missingWeights(
                missing.isEmpty ? decoder.pendingQuantizedKeys : missing
            )
        }
        eval(decoder)
        return decoder
    }
}

// MARK: - Downloader

/// Zero-configuration downloader for HuggingFace-style model repositories.
///
/// Files are fetched into a local cache directory and reused on later calls, so
/// `ReActAgent`-style applications can run on-device generation without a separate
/// Ollama or LM Studio server:
///
/// ```swift
/// let downloader = ModelDownloader()
/// let folder = try await downloader.download(repo: "mlx-community/Llama-3.2-1B-Instruct-4bit")
/// let decoder = try TransformerDecoder.load(directory: folder)
/// ```
public struct ModelDownloader: Sendable {
    /// Server root; repositories resolve to `<endpoint>/<repo>/resolve/<revision>/<file>`.
    public var endpoint: URL
    /// Session used for transfers.
    public var session: URLSession
    /// Parent directory for downloaded repositories.
    public var cacheDirectory: URL

    /// Default cache: `~/Library/Application Support/SwiftSci/Models`.
    public static var defaultCacheDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return base.appendingPathComponent("SwiftSci/Models", isDirectory: true)
    }

    /// Creates a downloader.
    /// - Parameters:
    ///   - endpoint: Server root. Defaults to `https://huggingface.co`; `file://` roots work for offline mirrors.
    ///   - cacheDirectory: Where repositories are stored. Defaults to ``defaultCacheDirectory``.
    ///   - session: URL session to use.
    public init(endpoint: URL = URL(string: "https://huggingface.co")!,
                cacheDirectory: URL = ModelDownloader.defaultCacheDirectory,
                session: URLSession = .shared) {
        self.endpoint = endpoint
        self.cacheDirectory = cacheDirectory
        self.session = session
    }

    /// Downloads `config.json`, `tokenizer.json`, and all weight shards of a repository.
    ///
    /// Files that already exist and are non-empty are kept, so interrupted downloads resume
    /// at file granularity.
    ///
    /// - Parameters:
    ///   - repo: `owner/name` repository identifier.
    ///   - revision: Branch, tag, or commit. Defaults to `main`.
    ///   - progress: Called with the fraction of files completed.
    /// - Throws: ``LLMModelLoadingError`` or a transfer error.
    /// - Returns: The local directory, ready for ``TransformerDecoder/load(directory:maxSeqLen:)``.
    public func download(repo: String, revision: String = "main",
                         progress: (@Sendable (Double) -> Void)? = nil) async throws -> URL {
        guard Self.isSafeRepo(repo) else { throw LLMModelLoadingError.invalidIdentifier(repo) }
        guard Self.isSafeComponent(revision) else { throw LLMModelLoadingError.invalidIdentifier(revision) }
        let directory = cacheDirectory
            .appendingPathComponent(repo, isDirectory: true)
            .appendingPathComponent(revision, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        try await fetch("config.json", repo: repo, revision: revision, into: directory)
        try await fetch("tokenizer.json", repo: repo, revision: revision, into: directory)
        try await fetch("generation_config.json", repo: repo, revision: revision, into: directory, optional: true)
        let hasIndex = try await fetch("model.safetensors.index.json", repo: repo, revision: revision,
                                       into: directory, optional: true)
        let shards: [String]
        if hasIndex {
            shards = try Self.shardNames(in: directory)
        } else {
            shards = ["model.safetensors"]
        }
        progress?(0)
        for (index, shard) in shards.enumerated() {
            try await fetch(shard, repo: repo, revision: revision, into: directory)
            progress?(Double(index + 1) / Double(shards.count))
        }
        return directory
    }

    /// Names of the weight files in a checkpoint directory, in a stable order.
    static func shardNames(in directory: URL) throws -> [String] {
        let fm = FileManager.default
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        if fm.fileExists(atPath: index.path) {
            guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: index)) as? [String: Any],
                  let map = root["weight_map"] as? [String: String] else {
                throw LLMModelLoadingError.invalidConfig("model.safetensors.index.json has no weight_map")
            }
            let names = Set(map.values).sorted()
            for name in names where !isSafeComponent(name) { throw LLMModelLoadingError.invalidIdentifier(name) }
            guard !names.isEmpty else { throw LLMModelLoadingError.missingFile("model.safetensors") }
            return names
        }
        guard fm.fileExists(atPath: directory.appendingPathComponent("model.safetensors").path) else {
            throw LLMModelLoadingError.missingFile("model.safetensors")
        }
        return ["model.safetensors"]
    }

    @discardableResult
    private func fetch(_ name: String, repo: String, revision: String, into directory: URL,
                       optional: Bool = false) async throws -> Bool {
        let destination = directory.appendingPathComponent(name)
        let size = (try? FileManager.default.attributesOfItem(atPath: destination.path)[.size] as? Int) ?? 0
        if size > 0 { return true }
        let source = endpoint.appendingPathComponent("\(repo)/resolve/\(revision)/\(name)")
        let temporary: URL
        let response: URLResponse
        do {
            (temporary, response) = try await session.download(from: source)
        } catch {
            if optional, (error as? URLError)?.code == .fileDoesNotExist || (error as? URLError)?.code == .resourceUnavailable {
                return false
            }
            throw error
        }
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            try? FileManager.default.removeItem(at: temporary)
            if optional && http.statusCode == 404 { return false }
            throw LLMModelLoadingError.httpStatus(http.statusCode, name)
        }
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: temporary, to: destination)
        return true
    }

    static func isSafeComponent(_ value: String) -> Bool {
        !value.isEmpty && value != "." && value != ".." &&
            value.unicodeScalars.allSatisfy { CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-").contains($0) }
    }

    static func isSafeRepo(_ repo: String) -> Bool {
        let parts = repo.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        return parts.count == 2 && parts.allSatisfy(isSafeComponent)
    }
}
#endif // os(macOS)

