#if os(macOS)
import Foundation
import MLX
import MLXNN
import SwiftNLP

// MARK: - LLM Configuration

/// Configuration for the TransformerDecoder model architecture.
///
/// Use predefined presets or supply custom values for research purposes.
///
/// ```swift
/// // Llama-3.2-1B compatible config:
/// let config = LLMConfig(
///     vocabSize: 128256,
///     numLayers: 16,
///     hiddenDim: 2048,
///     numHeads: 32,
///     numKVHeads: 8,
///     intermediateSize: 8192,
///     maxSeqLen: 8192
/// )
/// ```
/// Scheme defining how positional information is encoded into tokens.
public enum PositionalEncodingScheme: Sendable, Equatable {
    /// Rotary Positional Embedding (RoPE) applied to Q and K with configurable base frequency.
    case rope(base: Float = 10_000.0)
    /// Llama 3.1/3.2 rotary encoding with explicit checkpoint scaling parameters.
    case llama3RoPE(base: Float = 500_000.0, scaling: Llama3RoPEScaling)
    /// Classical learned additive positional embeddings.
    case learned
}

public struct LLMConfig: Sendable {
    /// Token vocabulary size.
    public var vocabSize: Int
    /// Number of stacked transformer decoder layers.
    public var numLayers: Int
    /// Hidden (embedding) dimensionality.
    public var hiddenDim: Int
    /// Number of query attention heads.
    public var numHeads: Int
    /// Number of key/value heads. `nil` follows `numHeads`, preserving multi-head attention.
    /// A supplied count must be positive and divide `numHeads` exactly.
    public var numKVHeads: Int?
    /// Feed-forward network intermediate dimension (SwiGLU gate/up projection size).
    public var intermediateSize: Int
    /// Configured sequence capacity. Generation counts prompt and output tokens together.
    public var maxSeqLen: Int
    /// RMSNorm epsilon for numerical stability (default `1e-5`).
    public var rmsNormEps: Float
    /// Positional encoding scheme (default `.rope(base: 10_000.0)`).
    public var positionalEncoding: PositionalEncodingScheme
    /// Use the token embedding matrix for output logits instead of a separate head.
    public var tieWordEmbeddings: Bool
    /// Token IDs that end generation without being emitted.
    /// When nonempty, decoded text never acts as a stop signal. An empty set retains
    /// legacy stopping on empty decoded text or the literal `<unk>`.
    public var eosTokenIDs: Set<Int>

    /// Creates an LLM configuration.
    /// - Parameters:
    ///   - vocabSize: Vocabulary size.
    ///   - numLayers: Number of decoder layers (e.g. 16 for 1B, 32 for 7B).
    ///   - hiddenDim: Model hidden dimension (e.g. 2048 for 1B, 4096 for 7B).
    ///   - numHeads: Number of query attention heads.
    ///   - numKVHeads: Key/value head count. `nil` uses the current query-head count.
    ///   - intermediateSize: SwiGLU FFN inner projection size (typically `hiddenDim * 4`).
    ///   - maxSeqLen: Maximum token sequence length (default `2048`).
    ///   - rmsNormEps: RMSNorm epsilon (default `1e-5`).
    ///   - positionalEncoding: Positional encoding scheme (default `.rope(base: 10_000.0)`).
    ///   - tieWordEmbeddings: Share input and output weights. Defaults to `false`.
    ///   - eosTokenIDs: Checkpoint stop-token IDs. Defaults to an empty set.
    public init(
        vocabSize: Int,
        numLayers: Int,
        hiddenDim: Int,
        numHeads: Int,
        numKVHeads: Int? = nil,
        intermediateSize: Int? = nil,
        maxSeqLen: Int = 2048,
        rmsNormEps: Float = 1e-5,
        positionalEncoding: PositionalEncodingScheme = .rope(base: 10_000.0),
        tieWordEmbeddings: Bool = false,
        eosTokenIDs: Set<Int> = []
    ) {
        self.vocabSize          = vocabSize
        self.numLayers          = numLayers
        self.hiddenDim          = hiddenDim
        self.numHeads           = numHeads
        self.numKVHeads         = numKVHeads
        self.intermediateSize   = intermediateSize ?? (hiddenDim * 4)
        self.maxSeqLen          = maxSeqLen
        self.rmsNormEps         = rmsNormEps
        self.positionalEncoding = positionalEncoding
        self.tieWordEmbeddings = tieWordEmbeddings
        self.eosTokenIDs = eosTokenIDs
        _ = validatedKVHeadCount()
    }

    func validatedKVHeadCount() -> Int {
        precondition(numHeads > 0 && hiddenDim > 0 && hiddenDim % numHeads == 0,
                     "hiddenDim must be positive and divisible by a positive numHeads")
        let count = numKVHeads ?? numHeads
        precondition(count > 0 && numHeads % count == 0,
                     "numKVHeads must be positive and divide numHeads")
        return count

    }

    // MARK: - Presets

    /// Minimal debug configuration (fast, not useful for inference).
    public static var debug: LLMConfig {
        LLMConfig(vocabSize: 1024, numLayers: 2, hiddenDim: 128, numHeads: 4, maxSeqLen: 256, positionalEncoding: .rope(base: 10_000.0))
    }

    /// Llama 3.2-1B configuration with an 8,192-token default context capacity.
    /// The checkpoint declares 131,072 positions; 8,192 is the native generation
    /// range validated for this preset, distinct from its factor-32 RoPE scaling.
    /// Set `maxSeqLen` before model construction to experiment with a larger
    /// capacity. Full 131,072-token inference is not validated by this preset.
    public static var llama1B: LLMConfig {
        LLMConfig(vocabSize: 128_256, numLayers: 16, hiddenDim: 2048, numHeads: 32, numKVHeads: 8,
                  intermediateSize: 8192, maxSeqLen: 8192,
                  positionalEncoding: .llama3RoPE(scaling: Llama3RoPEScaling(factor: 32)),
                  tieWordEmbeddings: true, eosTokenIDs: [128001, 128008, 128009])
    }

    /// Approximate Llama 3-8B compatible configuration.
    public static var llama8B: LLMConfig {
        LLMConfig(vocabSize: 128_256, numLayers: 32, hiddenDim: 4096, numHeads: 32, numKVHeads: 8,
                  intermediateSize: 14336, maxSeqLen: 8192, positionalEncoding: .rope(base: 500_000.0))
    }

    /// Alias for Llama 3-8B compatible configuration.
    public static var llama3_8B: LLMConfig { llama8B }
    /// Alias for Llama 3.2-1B compatible configuration.
    public static var llama3_1B: LLMConfig { llama1B }
}

// MARK: - SwiGLU Feed-Forward Network

/// Gated feed-forward network using the SwiGLU activation (used in Llama 2/3).
///
/// ```
/// FFN(x) = down( silu(gate(x)) * up(x) )
/// ```
/// This formulation (Shazeer 2020, "GLU Variants Improve Transformer") trains
/// faster and achieves better perplexity than ReLU FFN at equal FLOP budget.
///
/// - Note: This layer is parameterized by `config.intermediateSize` (inner
///   dimension for `gate` and `up` projections) and `config.hiddenDim`
///   (outer dimension for `down`).
public final class SwiGLUFFN: Module, UnaryLayer {
    /// Gate projection: hidden → intermediate.
    @ModuleInfo public var gate: Linear
    /// Up projection: hidden → intermediate.
    @ModuleInfo public var up: Linear
    /// Down projection: intermediate → hidden.
    @ModuleInfo public var down: Linear

    /// Creates a SwiGLU feed-forward block.
    /// - Parameter config: Model configuration supplying `hiddenDim` and `intermediateSize`.
    public init(config: LLMConfig) {
        self.gate = Linear(config.hiddenDim, config.intermediateSize, bias: false)
        self.up   = Linear(config.hiddenDim, config.intermediateSize, bias: false)
        self.down = Linear(config.intermediateSize, config.hiddenDim, bias: false)
        super.init()
    }

    /// Forward pass: computes `down(silu(gate(x)) * up(x))`.
    /// - Parameter x: Input tensor `[batch, seq, hiddenDim]`.
    /// - Returns: Output tensor `[batch, seq, hiddenDim]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - TransformerBlock

/// A single pre-norm transformer decoder block:
/// ```
/// x = x + Attention( RMSNorm(x) )
/// x = x + FFN( RMSNorm(x) )
/// ```
/// Uses pre-normalization (as in Llama 2/3) for training stability.
public final class TransformerBlock: Module, UnaryLayer {
    @ModuleInfo public var norm1: RMSNorm
    @ModuleInfo public var attention: MultiHeadAttention
    @ModuleInfo public var norm2: RMSNorm
    @ModuleInfo public var ffn: SwiGLUFFN
    @ModuleInfo public var rope: RoPEEmbedding?
    private let numKVHeads: Int

    /// Creates a decoder block.
    /// - Parameter config: Full LLM configuration.
    public init(config: LLMConfig) {
        let kvHeads = config.validatedKVHeadCount()
        self.numKVHeads = kvHeads
        self.norm1     = RMSNorm(dimensions: config.hiddenDim, eps: config.rmsNormEps)
        self.attention = kvHeads == config.numHeads
            ? MultiHeadAttention(dimensions: config.hiddenDim, numHeads: config.numHeads)
            : GroupedQueryAttention(dimensions: config.hiddenDim, numHeads: config.numHeads, numKVHeads: kvHeads)
        self.norm2     = RMSNorm(dimensions: config.hiddenDim, eps: config.rmsNormEps)
        self.ffn       = SwiGLUFFN(config: config)

        switch config.positionalEncoding {
        case .rope(let base):
            let headDim = config.hiddenDim / config.numHeads
            self.rope = RoPEEmbedding(dimensions: headDim, base: base)
        case .llama3RoPE(let base, let scaling):
            let headDim = config.hiddenDim / config.numHeads
            self.rope = RoPEEmbedding(dimensions: headDim, base: base, scaling: scaling)
        case .learned:
            self.rope = nil
        }
        super.init()
    }

    /// Pre-norm forward pass with residual connections.
    /// - Parameter x: Input `[batch, seq, hiddenDim]`.
    /// - Returns: Output `[batch, seq, hiddenDim]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        forward(x, maskMode: x.shape[1] > 1 ? .causal : .none, cache: nil, offset: 0)
    }

    /// Flexible forward pass supporting RoPE position offset and KV-cache accumulation.
    /// - Parameters:
    ///   - x: Input `[batch, seq, hiddenDim]`.
    ///   - mask: Optional attention additive causal mask.
    ///   - cache: Optional KVCache instance for autoregressive accumulation.
    ///   - offset: Token sequence position offset (0 for prefill, accumulated count for decode).
    /// - Returns: Output tensor `[batch, seq, hiddenDim]`.
    public func forward(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil,
        offset: Int = 0
    ) -> MLXArray {
        forward(x, maskMode: mask.map { .array($0) } ?? .none, cache: cache, offset: offset)
    }

    func forward(
        _ x: MLXArray,
        maskMode: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        offset: Int
    ) -> MLXArray {
        let xNorm1 = norm1(x)

        var q = attention.queryProjection(xNorm1)
        var k = attention.keyProjection(xNorm1)
        var v = attention.valueProjection(xNorm1)

        let numHeads = attention.numHeads
        q = unflatten(q, axis: -1, shape: [numHeads, -1]).transposed(0, 2, 1, 3)
        k = unflatten(k, axis: -1, shape: [numKVHeads, -1]).transposed(0, 2, 1, 3)
        v = unflatten(v, axis: -1, shape: [numKVHeads, -1]).transposed(0, 2, 1, 3)

        if let rope = self.rope {
            if q.shape[0] > 1 && q.shape[2] == 1 {
                // Pinned MLX single-token RoPE dispatch omits the batch dimension.
                // Folding batches into heads keeps every independent position in the grid.
                let qShape = q.shape
                let kShape = k.shape
                q = rope(q.reshaped([1, qShape[0] * qShape[1], 1, qShape[3]]), offset: offset)
                    .reshaped(qShape)
                k = rope(k.reshaped([1, kShape[0] * kShape[1], 1, kShape[3]]), offset: offset)
                    .reshaped(kShape)
            } else {
                q = rope(q, offset: offset)
                k = rope(k, offset: offset)
            }
        }

        let finalK: MLXArray
        let finalV: MLXArray
        if let cache = cache {
            let (updatedK, updatedV) = cache.update(keys: k, values: v)
            finalK = updatedK
            finalV = updatedV
        } else {
            finalK = k
            finalV = v
        }

        let scale = sqrt(1 / Float(q.dim(-1)))
        var output = MLXFast.scaledDotProductAttention(
            queries: q, keys: finalK, values: finalV, scale: scale, mask: maskMode)

        output = output.transposed(0, 2, 1, 3).flattened(start: -2, end: -1)
        let attn = attention.outProjection(output)
        let h = x + attn

        return h + ffn(norm2(h))
    }
}

// MARK: - TransformerDecoder (3.5 — N-layer stack)

/// A full N-layer decoder-only Transformer.
///
/// ## Architecture
/// ```
/// Embedding + PositionEmbedding
///     └─► [TransformerBlock × numLayers]
///             └─► Attention(pre-norm) + SwiGLU FFN(pre-norm)
///     └─► Final RMSNorm
///     └─► LM Head (Linear, no bias, tied to embedding by default)
/// ```
///
/// ## Loading weights
/// ```swift
/// let config = LLMConfig.llama1B
/// let decoder = TransformerDecoder(config: config, tokenizer: myTokenizer)
/// let loader  = YOLOWeightLoader(weights: SafeTensorsParser.parse(url: modelURL))
/// try decoder.loadWeights(loader)
/// ```
///
/// ## Inference
/// ```swift
/// let stream = try await decoder.generateStream(prompt: "Hello", options: .init(maxTokens: 50))
/// for try await token in stream { print(token, terminator: "") }
/// ```
///
/// ## Compile caching
/// MLX compiles and caches the forward graph per sequence-length bucket
/// (16 / 32 / 64 / 128 / 256 / 512 / 1024 / 2048) to amortize Metal
/// graph-build cost — analogous to `torch.compile`.
public final class TransformerDecoder: Module, LLMModel, @unchecked Sendable {

    // MARK: Sub-modules

    @ModuleInfo public var embedding: Embedding
    @ModuleInfo public var posEmbedding: Embedding
    /// Stack of N decoder blocks.
    @ModuleInfo public var layers: [TransformerBlock]
    @ModuleInfo public var finalNorm: RMSNorm
    /// Independent output projection, or `nil` when logits use the current token embedding.
    @ModuleInfo public var lmHead: Linear?

    // MARK: Configuration

    /// Full LLM configuration used to build this model.
    public let config: LLMConfig
    /// Tokenizer used to encode/decode text.
    public let tokenizer: any Tokenizer

    // MARK: Compile cache

    private var compiledForwardCache: [Int: (MLXArray) -> MLXArray] = [:]

    // MARK: - Init

    /// Creates an N-layer TransformerDecoder.
    ///
    /// - Parameters:
    ///   - config: Model architecture configuration.
    ///   - tokenizer: Tokenizer for text ↔ token-ID conversion.
    public init(config: LLMConfig, tokenizer: any Tokenizer) {
        _ = config.validatedKVHeadCount()
        self.config    = config
        self.tokenizer = tokenizer

        self.embedding    = Embedding(embeddingCount: config.vocabSize,  dimensions: config.hiddenDim)
        self.posEmbedding = Embedding(embeddingCount: config.maxSeqLen,  dimensions: config.hiddenDim)
        self.layers       = (0..<config.numLayers).map { _ in TransformerBlock(config: config) }
        self.finalNorm    = RMSNorm(dimensions: config.hiddenDim, eps: config.rmsNormEps)
        self.lmHead       = config.tieWordEmbeddings ? nil : Linear(config.hiddenDim, config.vocabSize, bias: false)

        super.init()
    }

    /// Convenience initializer for backwards compatibility.
    public convenience init(
        vocabSize: Int,
        tokenizer: any Tokenizer,
        dimensions: Int = 128,
        numHeads: Int = 4,
        maxSeqLen: Int = 256
    ) {
        let config = LLMConfig(
            vocabSize: vocabSize,
            numLayers: 2,
            hiddenDim: dimensions,
            numHeads: numHeads,
            maxSeqLen: maxSeqLen
        )
        self.init(config: config, tokenizer: tokenizer)
    }

    // MARK: - Forward pass

    /// Executes a forward pass supporting optional layer-wise KV caching and position offsets.
    ///
    /// - Parameters:
    ///   - x: Token IDs, shape `[seq_len]` or `[batch, seq_len]`.
    ///   - caches: Optional array of KVCache instances, one per transformer layer.
    ///   - offset: Token sequence position offset (0 for prefill, cached length for decode).
    /// - Returns: Logits tensor, shape `[batch, seq_len, vocab_size]`.
    public func forward(
        _ x: MLXArray,
        caches: [KVCache]? = nil,
        offset: Int = 0
    ) -> MLXArray {
        var input = x
        if input.ndim == 1 {
            input = input.expandedDimensions(axis: 0)
        }
        let seqLen = input.shape[1]

        var h = embedding(input)
        if case .learned = config.positionalEncoding {
            let positions = MLXArray(offset..<(offset + seqLen))
            h = h + posEmbedding(positions)
        }

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = seqLen > 1 ? .causal : .none

        for (idx, layer) in layers.enumerated() {
            let cache = caches?[idx]
            h = layer.forward(h, maskMode: maskMode, cache: cache, offset: offset)
        }

        h = finalNorm(h)
        if let lmHead { return lmHead(h) }
        return embedding.asLinear(h)
    }

    /// Executes the full N-layer decoder forward pass without KV caching.
    ///
    /// - Parameter x: Token IDs, shape `[seq_len]` or `[batch, seq_len]`.
    /// - Returns: Logits, shape `[batch, seq_len, vocab_size]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        forward(x, caches: nil, offset: 0)
    }

    // MARK: - Compiled forward

    private func compiledForward(seqLen: Int) -> (MLXArray) -> MLXArray {
        let bucket = seqLenBucket(seqLen)
        if let cached = compiledForwardCache[bucket] { return cached }
        let compiled = MLX.compile(self.callAsFunction)
        compiledForwardCache[bucket] = compiled
        return compiled
    }

    private func seqLenBucket(_ seqLen: Int) -> Int {
        let buckets = [16, 32, 64, 128, 256, 512, 1024, 2048]
        return buckets.first(where: { $0 >= seqLen }) ?? seqLen
    }

    // MARK: - Weight loading

    /// Loads weights from a dictionary of tensors into the model's sub-modules.
    ///
    /// Expected key format mirrors HuggingFace Llama naming:
    /// - `model.embed_tokens.weight`
    /// - `model.layers.{i}.self_attn.q_proj.weight`
    /// - `model.layers.{i}.mlp.gate_proj.weight`
    /// - `model.norm.weight`
    /// - `lm_head.weight`
    ///
    /// Loads weights from a dictionary of QuantizedTensors into the model's sub-modules.
    /// - Parameter weights: Weight dictionary of QuantizedTensors.
    /// - Returns: List of expected parameter keys that were missing in the input.
    @discardableResult
    public func loadWeights(_ weights: [String: QuantizedTensor]) -> [String] {
        loadWeights(weights.dequantized())
    }

    /// Loads weights from a dictionary of tensors into the model's sub-modules.
    ///
    /// Expected key format mirrors HuggingFace Llama naming:
    /// - `model.embed_tokens.weight`
    /// - `model.layers.{i}.self_attn.q_proj.weight`
    /// - `model.layers.{i}.mlp.gate_proj.weight`
    /// - `model.norm.weight`
    /// - `lm_head.weight`
    ///
    /// - Parameter weights: Weight dictionary keyed by parameter path.
    /// - Returns: List of expected parameter keys that were missing in the input.
    @discardableResult
    public func loadWeights(_ weights: [String: MLXArray]) -> [String] {
        var params: [String: MLXArray] = [:]
        var missing: [String] = []

        func mapParam(srcKey: String, dstKey: String) {
            if let w = weights[srcKey] {
                params[dstKey] = w
            } else {
                missing.append(srcKey)
            }
        }

        // Embedding
        mapParam(srcKey: "model.embed_tokens.weight", dstKey: "embedding.weight")

        // Positional embeddings may not exist in RoPE-based models (optional)
        if let w = weights["model.pos_embed.weight"] {
            params["posEmbedding.weight"] = w
        }

        // Decoder layers
        for i in 0..<layers.count {
            let srcPfx = "model.layers.\(i)"
            let dstPfx = "layers.\(i)"

            mapParam(srcKey: "\(srcPfx).self_attn.q_proj.weight", dstKey: "\(dstPfx).attention.query_proj.weight")
            mapParam(srcKey: "\(srcPfx).self_attn.k_proj.weight", dstKey: "\(dstPfx).attention.key_proj.weight")
            mapParam(srcKey: "\(srcPfx).self_attn.v_proj.weight", dstKey: "\(dstPfx).attention.value_proj.weight")
            mapParam(srcKey: "\(srcPfx).self_attn.o_proj.weight", dstKey: "\(dstPfx).attention.out_proj.weight")

            mapParam(srcKey: "\(srcPfx).mlp.gate_proj.weight", dstKey: "\(dstPfx).ffn.gate.weight")
            mapParam(srcKey: "\(srcPfx).mlp.up_proj.weight", dstKey: "\(dstPfx).ffn.up.weight")
            mapParam(srcKey: "\(srcPfx).mlp.down_proj.weight", dstKey: "\(dstPfx).ffn.down.weight")

            if let w = weights["\(srcPfx).input_layernorm.weight"] {
                params["\(dstPfx).norm1.weight"] = w
            }
            if let w = weights["\(srcPfx).post_attention_layernorm.weight"] {
                params["\(dstPfx).norm2.weight"] = w
            }
        }

        // Final norm & LM head
        mapParam(srcKey: "model.norm.weight", dstKey: "finalNorm.weight")
        if lmHead != nil {
            mapParam(srcKey: "lm_head.weight", dstKey: "lmHead.weight")
        }

        if !params.isEmpty {
            self.update(parameters: NestedDictionary.unflattened(params))
        }

        return missing
    }

    // MARK: - Generation

    /// Generates decoded text without completion metadata.
    ///
    /// Invalid requests throw before the stream is returned. This legacy stream cannot
    /// report errors during iteration. Use ``generateDetails(prompt:options:)`` when
    /// the caller needs the stop reason and token counts.
    public func generate(prompt: String, options: LLMOptions) async throws -> AsyncStream<String> {
        try Task.checkCancellation()
        let tokens = try generationPrompt(prompt, options: options)
        return AsyncStream { continuation in
            let task = generationTask(tokens: tokens, options: options,
                emit: { event in
                    if let text = event.chunk { continuation.yield(text) }
                }, finish: { continuation.finish() })
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Generates decoded text, reporting invalid requests through the stream.
    ///
    /// This text-only adapter omits completion metadata. Use
    /// ``generateDetails(prompt:options:)`` to distinguish a stop token from a length limit.
    public func generateStream(prompt: String, options: LLMOptions) -> AsyncThrowingStream<String, any Error> {
        generationStream(prompt: prompt, options: options) { $0.chunk }
    }

    /// Generates text chunks followed by one completion record on normal termination.
    ///
    /// Prompts are never truncated. A prompt exceeding `config.maxSeqLen` throws
    /// ``GenerationError/promptTooLong(promptTokens:capacity:)``. A prompt at capacity
    /// produces no output and reports a context limit unless `maxTokens` is zero.
    /// The output budget is a ceiling; reaching context capacity reports a length stop.
    /// Zero output budgets complete without inference. Cancelling the consumer stops
    /// generation but does not guarantee delivery of a final completion record.
    public func generateDetails(prompt: String, options: LLMOptions) -> AsyncThrowingStream<GenerationEvent, any Error> {
        generationStream(prompt: prompt, options: options) { $0 }
    }

    private func generationPrompt(_ prompt: String, options: LLMOptions) throws -> [Int] {
        guard options.maxTokens >= 0 else { throw GenerationError.invalidMaxTokens(options.maxTokens) }
        guard config.maxSeqLen > 0 else { throw GenerationError.invalidContextLimit(config.maxSeqLen) }
        var tokens = tokenizer.encode(text: prompt)
        if tokens.isEmpty { tokens = [0] }
        guard tokens.count <= config.maxSeqLen else {
            throw GenerationError.promptTooLong(promptTokens: tokens.count, capacity: config.maxSeqLen)
        }
        return tokens
    }

    private func generationStream<Element: Sendable>(
        prompt: String, options: LLMOptions,
        transform: @escaping @Sendable (GenerationEvent) -> Element?
    ) -> AsyncThrowingStream<Element, any Error> {
        AsyncThrowingStream { continuation in
            do {
                try Task.checkCancellation()
                let tokens = try generationPrompt(prompt, options: options)
                let task = generationTask(tokens: tokens, options: options,
                    emit: { event in
                        if let value = transform(event) { continuation.yield(value) }
                    }, finish: { continuation.finish() })
                continuation.onTermination = { _ in task.cancel() }
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }

    private func generationTask(
        tokens promptTokens: [Int], options: LLMOptions,
        emit: @escaping @Sendable (GenerationEvent) -> Void,
        finish: @escaping @Sendable () -> Void
    ) -> Task<Void, Never> {
        Task {
            defer { finish() }
            let remaining = config.maxSeqLen - promptTokens.count
            let budget = min(options.maxTokens, remaining)
            var reason: GenerationStopReason = .length(options.maxTokens <= remaining ? .maxTokens : .contextWindow)
            var generated = 0
            var textDecoder = tokenizer.makeStreamDecoder()

            if budget > 0 && !Task.isCancelled {
                var tokens = promptTokens
                let caches = layers.map { _ in KVCache() }
                let prefill = forward(MLXArray(tokens).expandedDimensions(axis: 0), caches: caches, offset: 0)
                var lastLogits = prefill[0, prefill.shape[1] - 1]
                eval(lastLogits)

                while generated < budget {
                    if Task.isCancelled { break }
                    let next = Sampler.sample(logits: lastLogits, options: options, pastTokens: tokens)
                    if config.eosTokenIDs.contains(next) {
                        reason = .stop
                        break
                    }
                    let decoded = textDecoder.append(next)
                    if config.eosTokenIDs.isEmpty, let decoded, decoded.isEmpty || decoded == "<unk>" {
                        reason = .stop
                        break
                    }
                    generated += 1
                    tokens.append(next)
                    if let decoded, !decoded.isEmpty { emit(.chunk(decoded)) }
                    if generated == budget || Task.isCancelled { break }

                    let offset = caches.first?.count ?? (tokens.count - 1)
                    let step = forward(MLXArray([next]).expandedDimensions(axis: 0), caches: caches, offset: offset)
                    lastLogits = step[0, 0]
                    eval(lastLogits)
                }
            }

            if Task.isCancelled {
                reason = .cancelled
            } else {
                let text = textDecoder.finish()
                if !text.isEmpty { emit(.chunk(text)) }
            }
            emit(.info(GenerationCompletionInfo(promptTokenCount: promptTokens.count,
                generationTokenCount: generated, stopReason: reason)))
        }
    }
}

#endif // os(macOS)
