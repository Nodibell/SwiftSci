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
    /// Builds a configuration from a HuggingFace Llama `config.json`.
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
        func number(_ dict: [String: Any], _ key: String) -> Double? {
            guard let n = dict[key] as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else { return nil }
            return n.doubleValue
        }
        func integer(_ dict: [String: Any], _ key: String) -> Int? {
            guard let d = number(dict, key), d.rounded() == d, abs(d) < 1e12 else { return nil }
            return Int(d)
        }
        func required(_ key: String) throws -> Int {
            guard let value = integer(root, key), value > 0 else {
                throw LLMModelLoadingError.invalidConfig("'\(key)' is missing or not a positive integer")
            }
            return value
        }
        let hidden = try required("hidden_size")
        let layers = try required("num_hidden_layers")
        let heads = try required("num_attention_heads")
        let intermediate = try required("intermediate_size")
        let vocab = try required("vocab_size")
        let kvHeads = integer(root, "num_key_value_heads")
        guard hidden % heads == 0 else {
            throw LLMModelLoadingError.invalidConfig("hidden_size must be divisible by num_attention_heads")
        }
        if let kvHeads, kvHeads <= 0 || heads % kvHeads != 0 {
            throw LLMModelLoadingError.invalidConfig("num_attention_heads must be divisible by num_key_value_heads")
        }
        if let type = root["model_type"] as? String, type != "llama" {
            throw LLMModelLoadingError.invalidConfig("unsupported model_type '\(type)'")
        }
        let declaredContext = integer(root, "max_position_embeddings") ?? 2048
        let context = maxSeqLen ?? min(declaredContext, 8192)
        guard context > 0 else { throw LLMModelLoadingError.invalidConfig("maxSeqLen must be positive") }

        let theta = Float(number(root, "rope_theta") ?? 10_000)
        var encoding = PositionalEncodingScheme.rope(base: theta)
        if let scaling = root["rope_scaling"] as? [String: Any] {
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
            } else if let kind, kind != "default" {
                throw LLMModelLoadingError.invalidConfig("unsupported rope_scaling type '\(kind)'")
            }
        }

        var eos = Set<Int>()
        if let single = integer(root, "eos_token_id") {
            eos.insert(single)
        } else if let many = root["eos_token_id"] as? [Any] {
            for case let value as NSNumber in many { eos.insert(value.intValue) }
        }

        self.init(
            vocabSize: vocab, numLayers: layers, hiddenDim: hidden, numHeads: heads,
            numKVHeads: kvHeads, intermediateSize: intermediate, maxSeqLen: context,
            rmsNormEps: Float(number(root, "rms_norm_eps") ?? 1e-5),
            positionalEncoding: encoding,
            tieWordEmbeddings: (root["tie_word_embeddings"] as? Bool) ?? false,
            eosTokenIDs: eos)
    }
}

// MARK: - Directory loading

extension TransformerDecoder {
    /// Loads a complete HuggingFace-style Llama checkpoint directory.
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
        let config = try LLMConfig(huggingFaceConfigJSON: configData, maxSeqLen: maxSeqLen)
        let tokenizer = try BPETokenizer(llama3TokenizerJSON: read("tokenizer.json"))
        let shards = try ModelDownloader.shardNames(in: directory)
        let quantization = AffineQuantizationConfig(configJSON: configData)

        let decoder = TransformerDecoder(config: config, tokenizer: tokenizer)
        var missingAcrossShards: Set<String>?
        for shard in shards {
            let shardWeights = try SafeTensorsParser.parse(url: directory.appendingPathComponent(shard))
            let missing = if let quantization {
                decoder.loadQuantizedWeights(shardWeights, quantization: quantization)
            } else {
                decoder.loadWeights(shardWeights)
            }
            let shardMissing = Set(missing)
            missingAcrossShards = missingAcrossShards.map { $0.intersection(shardMissing) } ?? shardMissing
        }
        let missing = (missingAcrossShards.map { $0.sorted() }) ?? ["model.safetensors"]
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
        let directory = cacheDirectory.appendingPathComponent(repo.replacingOccurrences(of: "/", with: "--"),
                                                              isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        try await fetch("config.json", repo: repo, revision: revision, into: directory)
        try await fetch("tokenizer.json", repo: repo, revision: revision, into: directory)
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

