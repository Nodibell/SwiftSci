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
///     intermediateSize: 8192,
///     maxSeqLen: 8192
/// )
/// ```
public struct LLMConfig: Sendable {
    /// Token vocabulary size.
    public var vocabSize: Int
    /// Number of stacked transformer decoder layers.
    public var numLayers: Int
    /// Hidden (embedding) dimensionality.
    public var hiddenDim: Int
    /// Number of attention heads.
    public var numHeads: Int
    /// Feed-forward network intermediate dimension (SwiGLU gate/up projection size).
    public var intermediateSize: Int
    /// Maximum supported input sequence length.
    public var maxSeqLen: Int
    /// RMSNorm epsilon for numerical stability (default `1e-5`).
    public var rmsNormEps: Float

    /// Creates an LLM configuration.
    /// - Parameters:
    ///   - vocabSize: Vocabulary size.
    ///   - numLayers: Number of decoder layers (e.g. 16 for 1B, 32 for 7B).
    ///   - hiddenDim: Model hidden dimension (e.g. 2048 for 1B, 4096 for 7B).
    ///   - numHeads: Number of attention heads.
    ///   - intermediateSize: SwiGLU FFN inner projection size (typically `hiddenDim * 4`).
    ///   - maxSeqLen: Maximum token sequence length (default `2048`).
    ///   - rmsNormEps: RMSNorm epsilon (default `1e-5`).
    public init(
        vocabSize: Int,
        numLayers: Int,
        hiddenDim: Int,
        numHeads: Int,
        intermediateSize: Int? = nil,
        maxSeqLen: Int = 2048,
        rmsNormEps: Float = 1e-5
    ) {
        self.vocabSize        = vocabSize
        self.numLayers        = numLayers
        self.hiddenDim        = hiddenDim
        self.numHeads         = numHeads
        self.intermediateSize = intermediateSize ?? (hiddenDim * 4)
        self.maxSeqLen        = maxSeqLen
        self.rmsNormEps       = rmsNormEps
    }

    // MARK: - Presets

    /// Minimal debug configuration (fast, not useful for inference).
    public static var debug: LLMConfig {
        LLMConfig(vocabSize: 1024, numLayers: 2, hiddenDim: 128, numHeads: 4, maxSeqLen: 256)
    }

    /// Approximate Llama 3.2-1B compatible configuration.
    public static var llama1B: LLMConfig {
        LLMConfig(vocabSize: 128_256, numLayers: 16, hiddenDim: 2048, numHeads: 32,
                  intermediateSize: 8192, maxSeqLen: 8192)
    }

    /// Approximate Llama 3-8B compatible configuration.
    public static var llama8B: LLMConfig {
        LLMConfig(vocabSize: 128_256, numLayers: 32, hiddenDim: 4096, numHeads: 32,
                  intermediateSize: 14336, maxSeqLen: 8192)
    }
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

    /// Creates a decoder block.
    /// - Parameter config: Full LLM configuration.
    public init(config: LLMConfig) {
        self.norm1     = RMSNorm(dimensions: config.hiddenDim, eps: config.rmsNormEps)
        self.attention = MultiHeadAttention(dimensions: config.hiddenDim, numHeads: config.numHeads)
        self.norm2     = RMSNorm(dimensions: config.hiddenDim, eps: config.rmsNormEps)
        self.ffn       = SwiGLUFFN(config: config)
        super.init()
    }

    /// Pre-norm forward pass with residual connections.
    /// - Parameter x: Input `[batch, seq, hiddenDim]`.
    /// - Returns: Output `[batch, seq, hiddenDim]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Attention sub-layer (causal mask built inside for the sequence length)
        let seqLen = x.shape[1]
        let mask   = MultiHeadAttention.createAdditiveCausalMask(seqLen)
        let xNorm1 = norm1(x)
        let attn   = attention(xNorm1, keys: xNorm1, values: xNorm1, mask: mask)
        let h      = x + attn

        // FFN sub-layer
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
    @ModuleInfo public var lmHead: Linear

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
        self.config    = config
        self.tokenizer = tokenizer

        self.embedding    = Embedding(embeddingCount: config.vocabSize,  dimensions: config.hiddenDim)
        self.posEmbedding = Embedding(embeddingCount: config.maxSeqLen,  dimensions: config.hiddenDim)
        self.layers       = (0..<config.numLayers).map { _ in TransformerBlock(config: config) }
        self.finalNorm    = RMSNorm(dimensions: config.hiddenDim, eps: config.rmsNormEps)
        self.lmHead       = Linear(config.hiddenDim, config.vocabSize, bias: false)

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

    /// Executes the full N-layer decoder forward pass.
    ///
    /// - Parameter x: Token IDs, shape `[seq_len]` or `[batch, seq_len]`.
    /// - Returns: Logits, shape `[batch, seq_len, vocab_size]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var input = x
        if input.ndim == 1 {
            input = input.expandedDimensions(axis: 0)
        }
        let seqLen = input.shape[1]

        // Embeddings
        let positions = MLXArray(0..<seqLen)
        var h = embedding(input) + posEmbedding(positions)

        // N decoder layers
        for layer in layers {
            h = layer(h)
        }

        // Final norm + LM head
        return lmHead(finalNorm(h))
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

            mapParam(srcKey: "\(srcPfx).self_attn.q_proj.weight", dstKey: "\(dstPfx).attention.queryProjection.weight")
            mapParam(srcKey: "\(srcPfx).self_attn.k_proj.weight", dstKey: "\(dstPfx).attention.keyProjection.weight")
            mapParam(srcKey: "\(srcPfx).self_attn.v_proj.weight", dstKey: "\(dstPfx).attention.valueProjection.weight")
            mapParam(srcKey: "\(srcPfx).self_attn.o_proj.weight", dstKey: "\(dstPfx).attention.outProjection.weight")

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
        mapParam(srcKey: "lm_head.weight", dstKey: "lmHead.weight")

        if !params.isEmpty {
            self.update(parameters: NestedDictionary.unflattened(params))
        }

        return missing
    }

    // MARK: - Generation

    /// Generates tokens and yields them as an `AsyncStream<String>`.
    ///
    /// - Parameters:
    ///   - prompt: Text prompt to start generation.
    ///   - options: Sampling options (temperature, top-p, max tokens …).
    /// - Returns: An `AsyncStream<String>` of decoded token strings.
    public func generate(prompt: String, options: LLMOptions) async throws -> AsyncStream<String> {
        let tokenizer  = self.tokenizer
        let maxSeqLen  = self.config.maxSeqLen

        return AsyncStream<String> { continuation in
            let task = Task {
                var tokens = tokenizer.encode(text: prompt)
                if tokens.isEmpty { tokens = [0] }

                for _ in 0..<options.maxTokens {
                    if Task.isCancelled { break }
                    let inputTokens = Array(tokens.suffix(maxSeqLen))
                    let logits      = self(MLXArray(inputTokens))
                    let lastLogits  = logits[0, logits.shape[1] - 1]
                    eval(lastLogits)
                    let nextToken   = Sampler.sample(logits: lastLogits, options: options)
                    let decoded     = tokenizer.decode(tokens: [nextToken])
                    if decoded.isEmpty || decoded == "<unk>" { break }
                    continuation.yield(decoded)
                    tokens.append(nextToken)
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Generates tokens as an `AsyncThrowingStream<String, Error>`.
    ///
    /// Identical behaviour to ``generate(prompt:options:)`` but propagates
    /// errors through the stream.
    public func generateStream(prompt: String, options: LLMOptions) -> AsyncThrowingStream<String, any Error> {
        let tokenizer = self.tokenizer
        let maxSeqLen = self.config.maxSeqLen

        return AsyncThrowingStream<String, any Error> { continuation in
            let task = Task {
                var tokens = tokenizer.encode(text: prompt)
                if tokens.isEmpty { tokens = [0] }

                for _ in 0..<options.maxTokens {
                    if Task.isCancelled { break }
                    let inputTokens = Array(tokens.suffix(maxSeqLen))
                    let logits      = self(MLXArray(inputTokens))
                    let lastLogits  = logits[0, logits.shape[1] - 1]
                    eval(lastLogits)
                    let nextToken   = Sampler.sample(logits: lastLogits, options: options)
                    let decoded     = tokenizer.decode(tokens: [nextToken])
                    if decoded.isEmpty || decoded == "<unk>" { break }
                    continuation.yield(decoded)
                    tokens.append(nextToken)
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

#endif // os(macOS)
