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

/// Affine group quantization parameters of an MLX-format checkpoint.
///
/// These match the `quantization` block of an MLX `config.json`
/// (`{"group_size": 64, "bits": 4}`). Quantized tensors are stored as bit-packed
/// `uint32` weights plus per-group `scales` and `biases`.
public struct AffineQuantizationConfig: Sendable, Equatable {
    struct ModuleOverride: Sendable, Equatable {
        var groupSize: Int
        var bits: Int
    }

    /// Number of consecutive input features sharing one scale and bias.
    public var groupSize: Int
    /// Bits per stored weight (for example 2, 3, 4, 6 or 8).
    public var bits: Int
    var moduleOverrides: [String: ModuleOverride] = [:]
    var allowPerModuleInference: Bool = false

    /// Creates quantization parameters.
    public init(groupSize: Int = 64, bits: Int = 4) {
        self.groupSize = groupSize
        self.bits = bits
    }

    public static func == (lhs: AffineQuantizationConfig, rhs: AffineQuantizationConfig) -> Bool {
        lhs.groupSize == rhs.groupSize && lhs.bits == rhs.bits && lhs.moduleOverrides == rhs.moduleOverrides
    }

    /// Reads `quantization` (or `quantization_config`) from a checkpoint `config.json`.
    /// - Parameter data: Raw `config.json` bytes.
    /// - Returns: The parameters, or `nil` when the checkpoint is not quantized.
    public init?(configJSON data: Data) {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let block = (root["quantization"] ?? root["quantization_config"]) as? [String: Any],
              let group = block["group_size"] as? Int, let bits = block["bits"] as? Int,
              group > 0, bits > 0 else { return nil }
        self.init(groupSize: group, bits: bits)
        self.allowPerModuleInference = true
        var overrides: [String: ModuleOverride] = [:]
        for (rawKey, rawValue) in block {
            guard let sub = rawValue as? [String: Any],
                  let subGroup = sub["group_size"] as? Int,
                  let subBits = sub["bits"] as? Int,
                  subGroup > 0, subBits > 0,
                  let normalized = CheckpointKeyNormalizer.normalizeKey(rawKey) else { continue }
            let key = normalized.hasPrefix("layers.") ? "model." + normalized : normalized
            overrides[key] = ModuleOverride(groupSize: subGroup, bits: subBits)
        }
        self.moduleOverrides = overrides
    }
}

/// Checkpoint architectures implemented by the native SwiftSci decoder.
public enum DecoderArchitecture: String, Sendable {
    /// Llama 2/3 style decoder (the legacy default).
    case llama
    /// Dense Mistral decoder; Mixtral and Mistral multimodal models are separate architectures.
    case mistral
    /// Gemma 2 text decoder, including alternating local attention and logit softcaps.
    case gemma2
    /// Gemma 3 text decoder, including QK-normalization, dual RoPE frequencies, and linear long-context scaling.
    case gemma3
    /// Gemma 4 text decoder, including proportional RoPE, KV-sharing, per-layer inputs, and sparse MoE experts.
    case gemma4

    /// Returns the default `ChatTemplate` for this decoder architecture.
    /// - Parameter date: Date text supplied to date-aware templates such as Llama 3.2 Instruct (default `"26 Nov 2026"`).
    /// - Returns: The matching `ChatTemplate` for this architecture.
    public func defaultChatTemplate(date: String = "26 Nov 2026") -> ChatTemplate {
        switch self {
        case .llama:
            return .llama32Instruct(date: date)
        case .mistral:
            return .mistral
        case .gemma2, .gemma3:
            return .gemma
        case .gemma4:
            return .gemma4
        }
    }
}

/// Architectural hyperparameter configuration for a decoder-only Transformer language model.
public struct LLMConfig: Sendable {
    /// Architecture inferred from a Hugging Face config. Existing initializers default to Llama.
    public internal(set) var architecture: DecoderArchitecture = .llama
    var checkpointHeadDim: Int?
    var slidingWindow: Int?
    var slidingWindowPattern: Int = 6
    var attentionSoftcap: Float?
    var finalSoftcap: Float?
    var queryScale: Float?
    var linearRoPEScale: Float?
    var gemma3LocalRopeTheta: Float = 10_000
    var gemma3GlobalRopeTheta: Float = 1_000_000
    var gemma3GlobalRopeScale: Float = 1.0
    var gemma4: Gemma4Config?
    var headDim: Int { checkpointHeadDim ?? hiddenDim / numHeads }

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
    /// Generation rejects nonpositive limits with `GenerationError.invalidContextLimit`.
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

    init(
        vocabSize: Int,
        numLayers: Int,
        hiddenDim: Int,
        numHeads: Int,
        numKVHeads: Int?,
        intermediateSize: Int,
        maxSeqLen: Int,
        rmsNormEps: Float,
        positionalEncoding: PositionalEncodingScheme,
        tieWordEmbeddings: Bool,
        eosTokenIDs: Set<Int>,
        checkpointHeadDim: Int?
    ) {
        self.vocabSize          = vocabSize
        self.numLayers          = numLayers
        self.hiddenDim          = hiddenDim
        self.numHeads           = numHeads
        self.numKVHeads         = numKVHeads
        self.intermediateSize   = intermediateSize
        self.maxSeqLen          = maxSeqLen
        self.rmsNormEps         = rmsNormEps
        self.positionalEncoding = positionalEncoding
        self.tieWordEmbeddings  = tieWordEmbeddings
        self.eosTokenIDs        = eosTokenIDs
        self.checkpointHeadDim  = checkpointHeadDim
        _ = validatedKVHeadCount()
    }

    func validatedKVHeadCount() -> Int {
        precondition(numHeads > 0 && hiddenDim > 0 && (checkpointHeadDim != nil || hiddenDim % numHeads == 0),
                     "hiddenDim must be positive and divisible by a positive numHeads")
        let count = numKVHeads ?? numHeads
        precondition(count > 0 && numHeads % count == 0,
                     "numKVHeads must be positive and divide numHeads")
        return count
    }

    func isSlidingLayer(at layerIndex: Int) -> Bool {
        switch architecture {
        case .llama:
            return false
        case .mistral:
            return slidingWindow != nil
        case .gemma2:
            return layerIndex % 2 == 0
        case .gemma3:
            guard slidingWindowPattern > 1 else { return false }
            return (layerIndex + 1) % slidingWindowPattern != 0
        case .gemma4:
            guard let types = gemma4?.layerTypes, types.indices.contains(layerIndex) else { return false }
            return types[layerIndex] == "sliding_attention"
        }
    }

    func headDim(forLayer layerIndex: Int) -> Int {
        if architecture == .gemma4, !isSlidingLayer(at: layerIndex), let global = gemma4?.globalHeadDim {
            return global
        }
        return headDim
    }

    func kvHeadCount(forLayer layerIndex: Int) -> Int {
        if architecture == .gemma4,
           !isSlidingLayer(at: layerIndex),
           gemma4?.attentionKEqV == true,
           let globalKV = gemma4?.numGlobalKVHeads {
            return globalKV
        }
        return validatedKVHeadCount()
    }

    func hasKV(forLayer layerIndex: Int) -> Bool {
        guard architecture == .gemma4, let g4 = gemma4, g4.numKVSharedLayers > 0 else { return true }
        return layerIndex < numLayers - g4.numKVSharedLayers
    }

    func usesKEqV(forLayer layerIndex: Int) -> Bool {
        architecture == .gemma4 && !isSlidingLayer(at: layerIndex) && (gemma4?.attentionKEqV == true)
    }

    func intermediateSize(forLayer layerIndex: Int) -> Int {
        guard architecture == .gemma4, let g4 = gemma4 else { return intermediateSize }
        let firstShared = numLayers - g4.numKVSharedLayers
        let isDoubleWide = g4.useDoubleWideMLP && layerIndex >= firstShared && firstShared > 0
        return intermediateSize * (isDoubleWide ? 2 : 1)
    }

    func previousKVLayerIndices() -> [Int] {
        var result = Array(0..<numLayers)
        guard architecture == .gemma4, let g4 = gemma4, g4.numKVSharedLayers > 0 else { return result }
        let firstShared = max(0, numLayers - g4.numKVSharedLayers)
        var lastByType: [String: Int] = [:]
        for i in 0..<firstShared {
            lastByType[g4.layerTypes[i]] = i
        }
        for j in firstShared..<numLayers {
            if let source = lastByType[g4.layerTypes[j]] {
                result[j] = source
            }
        }
        return result
    }

    func ropeForLayer(at layerIndex: Int) -> RoPEEmbedding? {
        let dims = headDim(forLayer: layerIndex)
        switch architecture {
        case .gemma3:
            if isSlidingLayer(at: layerIndex) {
                return RoPEEmbedding(dimensions: dims, base: gemma3LocalRopeTheta)
            } else {
                return RoPEEmbedding(dimensions: dims, base: gemma3GlobalRopeTheta, scale: gemma3GlobalRopeScale)
            }
        case .gemma4:
            guard let g4 = gemma4 else { return RoPEEmbedding(dimensions: dims, base: 10_000) }
            let params = isSlidingLayer(at: layerIndex) ? g4.slidingRoPE : g4.fullRoPE
            if params.ropeType == "proportional" {
                let rawRotated = Int((Float(dims) * params.partialRotaryFactor).rounded())
                let rotatedDims = max(2, min(dims, rawRotated - (rawRotated % 2)))
                return RoPEEmbedding(
                    dimensions: dims,
                    rotatedDimensions: rotatedDims,
                    base: params.ropeTheta,
                    scale: params.scale
                )
            }
            return RoPEEmbedding(dimensions: dims, base: params.ropeTheta, scale: params.scale)
        case .llama, .mistral, .gemma2:
            switch positionalEncoding {
            case .rope(let base):
                return RoPEEmbedding(dimensions: dims, base: base, scale: linearRoPEScale ?? 1.0)
            case .llama3RoPE(let base, let scaling):
                return RoPEEmbedding(dimensions: dims, base: base, scaling: scaling)
            case .learned:
                return nil
            }
        }
    }

    // MARK: - Presets

    /// Minimal debug configuration (fast, not useful for inference).
    public static var debug: LLMConfig {
        LLMConfig(vocabSize: 1024, numLayers: 2, hiddenDim: 128, numHeads: 4, maxSeqLen: 256, positionalEncoding: .rope(base: 10_000.0))
    }

    /// Llama 3.2-1B configuration with an 8,192-token default context capacity.
    /// The checkpoint declares 131,072 positions. The smaller default limits
    /// context resource demands and is distinct from its factor-32 RoPE scaling.
    /// Set `maxSeqLen` before model construction to opt into a larger capacity.
    /// Validate memory use, latency, and model quality for the intended workload.
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
    private let usesGELU: Bool

    /// Creates a SwiGLU feed-forward block.
    /// - Parameter config: Model configuration supplying `hiddenDim` and `intermediateSize`.
    public convenience init(config: LLMConfig) {
        let useGELU = config.architecture == .gemma2 || config.architecture == .gemma3 || config.architecture == .gemma4
        self.init(hiddenDim: config.hiddenDim, intermediateSize: config.intermediateSize, usesGELU: useGELU)
    }

    init(hiddenDim: Int, intermediateSize: Int, usesGELU: Bool) {
        self.usesGELU = usesGELU
        self.gate = Linear(hiddenDim, intermediateSize, bias: false)
        self.up   = Linear(hiddenDim, intermediateSize, bias: false)
        self.down = Linear(intermediateSize, hiddenDim, bias: false)
        super.init()
    }

    /// Forward pass: computes `down((usesGELU ? geluApproximate(gate(x)) : silu(gate(x))) * up(x))`.
    /// - Parameter x: Input tensor `[batch, seq, hiddenDim]`.
    /// - Returns: Output tensor `[batch, seq, hiddenDim]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activated = usesGELU ? geluApproximate(gate(x)) : silu(gate(x))
        return down(activated * up(x))
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

    let hasKV: Bool
    let usesKEqV: Bool
    private let numKVHeads: Int
    private let layerHeadDim: Int
    private let architecture: DecoderArchitecture
    private let slidingWindow: Int?
    private let attentionSoftcap: Float?
    private let attentionScale: Float
    private let usesClipResidual: Bool

    @ModuleInfo var qNorm: RMSNorm?
    @ModuleInfo var kNorm: RMSNorm?
    @ModuleInfo var vNorm: ParameterlessRMSNorm?
    @ModuleInfo var postAttentionNorm: RMSNorm?
    @ModuleInfo var postFFNNorm: RMSNorm?
    @ModuleInfo var postFFNNorm1: RMSNorm?
    @ModuleInfo var preFFNNorm2: RMSNorm?
    @ModuleInfo var postFFNNorm2: RMSNorm?
    @ModuleInfo var router: Gemma4Router?
    @ModuleInfo var experts: Gemma4Experts?
    @ModuleInfo var perLayerInputGate: Linear?
    @ModuleInfo var perLayerProjection: Linear?
    @ModuleInfo var postPerLayerInputNorm: RMSNorm?
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray?

    /// Creates a decoder block.
    /// - Parameter config: Full LLM configuration.
    public convenience init(config: LLMConfig) {
        self.init(config: config, layerIndex: 0)
    }

    init(config: LLMConfig, layerIndex: Int) {
        self.architecture = config.architecture
        self.slidingWindow = config.isSlidingLayer(at: layerIndex) ? config.slidingWindow : nil
        self.attentionSoftcap = config.attentionSoftcap
        let headDim = config.headDim(forLayer: layerIndex)
        let kvHeads = config.kvHeadCount(forLayer: layerIndex)
        let hasKV = config.hasKV(forLayer: layerIndex)
        let usesKEqV = config.usesKEqV(forLayer: layerIndex)
        self.layerHeadDim = headDim
        self.numKVHeads = kvHeads
        self.hasKV = hasKV
        self.usesKEqV = usesKEqV
        self.usesClipResidual = config.architecture == .gemma3
        self.attentionScale = config.architecture == .gemma4
            ? 1.0
            : (config.queryScale ?? (1 / Float(headDim).squareRoot()))

        let hasQK = config.architecture == .gemma3 || config.architecture == .gemma4
        self.qNorm = hasQK ? config.makeNorm(dimensions: headDim) : nil
        self.kNorm = hasQK && hasKV ? config.makeNorm(dimensions: headDim) : nil
        self.vNorm = config.architecture == .gemma4 && hasKV ? ParameterlessRMSNorm(eps: config.rmsNormEps) : nil

        let hasPostNorms = config.architecture == .gemma2 || config.architecture == .gemma3 || config.architecture == .gemma4
        self.postAttentionNorm = hasPostNorms ? config.makeNorm() : nil
        self.postFFNNorm = hasPostNorms ? config.makeNorm() : nil

        if config.architecture == .gemma4, let g4 = config.gemma4 {
            if g4.enableMoEBlock {
                self.postFFNNorm1 = config.makeNorm()
                self.preFFNNorm2 = config.makeNorm()
                self.postFFNNorm2 = config.makeNorm()
                self.router = Gemma4Router(
                    hiddenDim: config.hiddenDim,
                    numExperts: g4.numExperts,
                    topKExperts: g4.topKExperts,
                    eps: config.rmsNormEps
                )
                self.experts = Gemma4Experts(
                    hiddenDim: config.hiddenDim,
                    moeIntermediateSize: g4.moeIntermediateSize,
                    numExperts: g4.numExperts
                )
            }
            if g4.hiddenSizePerLayerInput > 0 {
                let pleDim = g4.hiddenSizePerLayerInput
                self.perLayerInputGate = Linear(config.hiddenDim, pleDim, bias: false)
                self.perLayerProjection = Linear(pleDim, config.hiddenDim, bias: false)
                self.postPerLayerInputNorm = config.makeNorm()
            }
            self._layerScalar.wrappedValue = MLXArray.ones([1])
        } else {
            self._layerScalar.wrappedValue = nil
        }

        self.norm1 = config.makeNorm()
        if kvHeads == config.numHeads && config.checkpointHeadDim == nil && headDim == config.headDim && hasKV && !usesKEqV {
            self.attention = MultiHeadAttention(dimensions: config.hiddenDim, numHeads: config.numHeads)
        } else {
            self.attention = GroupedQueryAttention(
                dimensions: config.hiddenDim,
                numHeads: config.numHeads,
                numKVHeads: kvHeads,
                headDim: headDim,
                includeKeyProjection: hasKV,
                includeValueProjection: hasKV && !usesKEqV
            )
        }
        self.norm2 = config.makeNorm()
        let ffnDims = config.intermediateSize(forLayer: layerIndex)
        let useGELU = config.architecture == .gemma2 || config.architecture == .gemma3 || config.architecture == .gemma4
        self.ffn = SwiGLUFFN(hiddenDim: config.hiddenDim, intermediateSize: ffnDims, usesGELU: useGELU)
        self.rope = config.ropeForLayer(at: layerIndex)
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
        forwardStep(x, maskMode: maskMode, cache: cache, offset: offset, sharedKV: nil, perLayerInput: nil).output
    }

    private func applyRoPE(_ tensor: MLXArray, rope: RoPEEmbedding, offset: Int) -> MLXArray {
        if tensor.shape[0] > 1 && tensor.shape[2] == 1 {
            // Pinned MLX single-token RoPE dispatch omits the batch dimension.
            // Folding batches into heads keeps every independent position in the grid.
            let shape = tensor.shape
            return rope(tensor.reshaped([1, shape[0] * shape[1], 1, shape[3]]), offset: offset)
                .reshaped(shape)
        }
        return rope(tensor, offset: offset)
    }

    func forwardStep(
        _ x: MLXArray,
        maskMode: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        offset: Int,
        sharedKV: (keys: MLXArray, values: MLXArray)?,
        perLayerInput: MLXArray?
    ) -> (output: MLXArray, kv: (keys: MLXArray, values: MLXArray)?, offset: Int) {
        let xNorm1 = norm1(x)
        let numHeads = attention.numHeads

        var q = unflatten(attention.queryProjection(xNorm1), axis: -1, shape: [numHeads, -1])
        if let qNorm { q = qNorm(q) }
        q = q.transposed(0, 2, 1, 3)
        if let rope = self.rope { q = applyRoPE(q, rope: rope, offset: offset) }

        let finalK: MLXArray
        let finalV: MLXArray
        if let sharedKV {
            precondition(cache == nil, "KV-shared layers must not update a KV cache")
            finalK = sharedKV.keys
            finalV = sharedKV.values
        } else {
            precondition(hasKV, "Layer without KV projections requires sharedKV")
            let rawK = unflatten(attention.keyProjection(xNorm1), axis: -1, shape: [numKVHeads, -1])
            var k = kNorm.map { $0(rawK) } ?? rawK
            var v = usesKEqV
                ? rawK
                : unflatten(attention.valueProjection(xNorm1), axis: -1, shape: [numKVHeads, -1])
            k = k.transposed(0, 2, 1, 3)
            if let rope = self.rope { k = applyRoPE(k, rope: rope, offset: offset) }
            if let vNorm { v = vNorm(v) }
            v = v.transposed(0, 2, 1, 3)

            if let cache {
                let (updatedK, updatedV) = cache.update(keys: k, values: v)
                finalK = updatedK
                finalV = updatedV
            } else {
                finalK = k
                finalV = v
            }
        }

        var effectiveMask = maskMode
        if let window = slidingWindow {
            let positions = MLXArray(offset..<(offset + q.dim(2))).expandedDimensions(axis: 1)
            let keyPositions = MLXArray(0..<finalK.dim(2)).expandedDimensions(axis: 0)
            let allowed = (keyPositions .<= positions) .&& (keyPositions .> (positions - window))
            var additive = MLX.where(allowed, MLXArray(Float(0)), MLXArray(-Float.infinity))
            if case .array(let external) = maskMode { additive = additive + external }
            effectiveMask = .array(additive.asType(q.dtype))
        }

        let output: MLXArray
        if let cap = attentionSoftcap {
            // Group query heads without materializing repeated K/V arrays.
            let groupedQ = q.reshaped([q.dim(0), numKVHeads, numHeads / numKVHeads, q.dim(2), q.dim(3)])
            let keys = finalK.expandedDimensions(axis: 2)
            let values = finalV.expandedDimensions(axis: 2)
            var scores = matmul(groupedQ, keys.swappedAxes(-1, -2)) * attentionScale
            scores = tanh(scores / cap) * cap
            switch effectiveMask {
            case .causal:
                let positions = MLXArray(offset..<(offset + q.dim(2))).expandedDimensions(axis: 1)
                let keyPositions = MLXArray(0..<finalK.dim(2)).expandedDimensions(axis: 0)
                scores = MLX.where(keyPositions .<= positions, scores, MLXArray(-Float.infinity))
            case .array(let mask):
                if mask.ndim == 4 {
                    let groupedMask = mask.dim(1) == numHeads
                        ? mask.reshaped([mask.dim(0), numKVHeads, numHeads / numKVHeads, mask.dim(2), mask.dim(3)])
                        : mask.expandedDimensions(axis: 2)
                    scores = scores + groupedMask
                } else {
                    scores = scores + mask
                }
            default: break
            }
            output = matmul(softmax(scores.asType(.float32), axis: -1).asType(q.dtype), values)
                .reshaped(q.shape)
        } else {
            output = MLXFast.scaledDotProductAttention(
                queries: q, keys: finalK, values: finalV, scale: attentionScale, mask: effectiveMask
            )
        }

        let attn = attention.outProjection(output.transposed(0, 2, 1, 3).flattened(start: -2, end: -1))
        let attnResidual = postAttentionNorm.map { $0(attn) } ?? attn
        var h = usesClipResidual ? clipResidual(x, attnResidual) : (x + attnResidual)

        if let router, let experts, let postFFNNorm1, let preFFNNorm2, let postFFNNorm2, let postFFNNorm {
            let h1 = postFFNNorm1(ffn(norm2(h)))
            let (topKIndices, topKWeights) = router(h)
            let h2 = postFFNNorm2(experts(preFFNNorm2(h), topKIndices: topKIndices, topKWeights: topKWeights))
            h = h + postFFNNorm(h1 + h2)
        } else {
            let feedForward = ffn(norm2(h))
            let ffResidual = postFFNNorm.map { $0(feedForward) } ?? feedForward
            h = usesClipResidual ? clipResidual(h, ffResidual) : (h + ffResidual)
        }

        if let pleGate = perLayerInputGate,
           let pleProj = perLayerProjection,
           let pleNorm = postPerLayerInputNorm,
           let perLayerInput {
            let gated = pleProj(geluApproximate(pleGate(h)) * perLayerInput)
            h = h + pleNorm(gated)
        }

        if let layerScalar {
            h = h * layerScalar.asType(h.dtype)
        }

        return (h, hasKV ? (finalK, finalV) : sharedKV, offset)
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
public final class TransformerDecoder: Module, CompletionReportingLLMModel, @unchecked Sendable {

    // MARK: Sub-modules

    @ModuleInfo public var embedding: Embedding
    @ModuleInfo public var posEmbedding: Embedding
    /// Stack of N decoder blocks.
    @ModuleInfo public var layers: [TransformerBlock]
    @ModuleInfo public var finalNorm: RMSNorm
    /// Independent output projection, or `nil` when logits use the current token embedding.
    @ModuleInfo public var lmHead: Linear?
    @ModuleInfo var embedTokensPerLayer: Embedding?
    @ModuleInfo var perLayerModelProjection: Linear?
    @ModuleInfo var perLayerProjectionNorm: RMSNorm?

    // MARK: Configuration

    /// Full LLM configuration used to build this model.
    public let config: LLMConfig
    /// Tokenizer used to encode/decode text.
    public let tokenizer: any Tokenizer
    /// Generation configuration loaded from `generation_config.json`, when present.
    public private(set) var generationConfiguration: GenerationConfiguration?

    func setGenerationConfiguration(_ configuration: GenerationConfiguration?) {
        self.generationConfiguration = configuration
    }

    // MARK: Compile cache

    private var compiledForwardCache: [Int: (MLXArray) -> MLXArray] = [:]

    // MARK: Cross-shard quantized staging

    private var pendingQuantizedTensors: [String: MLXArray] = [:]
    private var quantizedModulePaths: Set<String> = []
    private var activeQuantization: AffineQuantizationConfig?

    var hasPendingQuantizedTensors: Bool { !pendingQuantizedTensors.isEmpty }
    var pendingQuantizedKeys: [String] { pendingQuantizedTensors.keys.sorted() }

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
        // Keep invalid limits in config for typed generation errors, without allocating an invalid tensor.
        self.posEmbedding = Embedding(embeddingCount: max(0, config.maxSeqLen), dimensions: config.hiddenDim)
        self.layers       = (0..<config.numLayers).map { TransformerBlock(config: config, layerIndex: $0) }
        self.finalNorm    = config.makeNorm()
        self.lmHead       = config.tieWordEmbeddings ? nil : Linear(config.hiddenDim, config.vocabSize, bias: false)
        if config.architecture == .gemma4, let g4 = config.gemma4, g4.hiddenSizePerLayerInput > 0 {
            let pleDim = g4.hiddenSizePerLayerInput
            self.embedTokensPerLayer = Embedding(
                embeddingCount: g4.vocabSizePerLayerInput,
                dimensions: config.numLayers * pleDim
            )
            self.perLayerModelProjection = Linear(config.hiddenDim, config.numLayers * pleDim, bias: false)
            self.perLayerProjectionNorm = RMSNorm(dimensions: pleDim, eps: config.rmsNormEps)
        }

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
        project(hiddenStates(x, caches: caches, offset: offset))
    }

    private func hiddenStates(_ x: MLXArray, caches: [KVCache]?, offset: Int) -> MLXArray {
        var input = x
        if input.ndim == 1 {
            input = input.expandedDimensions(axis: 0)
        }
        let seqLen = input.shape[1]

        var h = embedding(input)
        switch config.architecture {
        case .gemma2, .gemma4:
            h = h * MLXArray(Float(config.hiddenDim).squareRoot()).asType(h.dtype)
        case .gemma3:
            h = h * MLXArray(Float(config.hiddenDim).squareRoot()).asType(.bfloat16).asType(h.dtype)
        case .llama, .mistral:
            break
        }
        if case .learned = config.positionalEncoding {
            let positions = MLXArray(offset..<(offset + seqLen))
            h = h + posEmbedding(positions)
        }

        var perLayerInputs: [MLXArray]?
        if let pleEmbed = embedTokensPerLayer,
           let pleProj = perLayerModelProjection,
           let pleNorm = perLayerProjectionNorm,
           let g4 = config.gemma4,
           g4.hiddenSizePerLayerInput > 0 {
            let pleDim = g4.hiddenSizePerLayerInput
            var pleEmb = pleEmbed(input) * MLXArray(Float(pleDim).squareRoot()).asType(h.dtype)
            pleEmb = unflatten(pleEmb, axis: -1, shape: [config.numLayers, pleDim])

            var pleModel = pleProj(h) * MLXArray(1.0 / Float(config.hiddenDim).squareRoot()).asType(h.dtype)
            pleModel = unflatten(pleModel, axis: -1, shape: [config.numLayers, pleDim])
            pleModel = pleNorm(pleModel)

            let combined = (pleModel + pleEmb) * MLXArray(1.0 / Float(2.0).squareRoot()).asType(h.dtype)
            perLayerInputs = (0..<config.numLayers).map { combined[0..., 0..., $0, 0...] }
        }

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = seqLen > 1 ? .causal : .none
        let previousKV = config.previousKVLayerIndices()
        var intermediates: [((keys: MLXArray, values: MLXArray)?, Int)] = Array(repeating: (nil, 0), count: layers.count)

        for (idx, layer) in layers.enumerated() {
            let ple = perLayerInputs?[idx]
            if layer.hasKV {
                let cache = caches?[idx]
                let (nextH, kvs, usedOffset) = layer.forwardStep(
                    h, maskMode: maskMode, cache: cache, offset: offset, sharedKV: nil, perLayerInput: ple
                )
                h = nextH
                intermediates[idx] = (kvs, usedOffset)
            } else {
                let (sharedKV, sharedOffset) = intermediates[previousKV[idx]]
                let (nextH, _, _) = layer.forwardStep(
                    h, maskMode: maskMode, cache: nil, offset: sharedOffset, sharedKV: sharedKV, perLayerInput: ple
                )
                h = nextH
            }
        }

        return h
    }

    private func project(_ hidden: MLXArray) -> MLXArray {
        let h = finalNorm(hidden)
        let logits = lmHead.map { $0(h) } ?? embedding.asLinear(h)
        if let cap = config.finalSoftcap { return tanh(logits / cap) * cap }
        return logits
    }

    func prefill(_ tokens: [Int], caches: [KVCache], chunkSize: Int = 2048) -> MLXArray? {
        precondition(!tokens.isEmpty && chunkSize > 0)
        for start in stride(from: 0, to: tokens.count, by: chunkSize) {
            if Task.isCancelled { return nil }
            let end = min(start + chunkSize, tokens.count)
            let input = MLXArray(Array(tokens[start..<end]), [1, end - start])
            let h = hiddenStates(input, caches: caches, offset: start)
            if end == tokens.count {
                let logits = project(h[0, end - start - 1].reshaped([1, 1, config.hiddenDim]))[0, 0]
                eval(logits)
                return logits
            }
            eval(caches.flatMap { [$0.keys, $0.values].compactMap { $0 } })
        }
        return nil
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

    private struct LinearModuleSpec {
        let src: String
        let dst: String
        let inDim: Int
        let required: Bool
    }

    private struct DenseParameterSpec {
        let src: String
        let dst: String
        let required: Bool
    }

    var expectedDenseParameterKeys: [String] {
        denseParameterMappings().filter(\.required).map(\.src)
    }

    private func ensureLMHeadIfPresent(in weights: [String: MLXArray]) {
        guard config.architecture == .gemma2 || config.architecture == .gemma3 || config.architecture == .gemma4,
              lmHead == nil,
              weights["lm_head.weight"] != nil || weights["lm_head.scales"] != nil || weights["lm_head.biases"] != nil
        else { return }
        update(modules: NestedDictionary.unflattened(["lmHead": Linear(config.hiddenDim, config.vocabSize, bias: false)]))
    }

    private func linearModuleSpecs() -> [LinearModuleSpec] {
        var specs: [LinearModuleSpec] = [
            LinearModuleSpec(src: "model.embed_tokens", dst: "embedding", inDim: config.hiddenDim, required: true)
        ]
        if config.architecture == .gemma4, let g4 = config.gemma4, g4.hiddenSizePerLayerInput > 0 {
            let pleTotal = config.numLayers * g4.hiddenSizePerLayerInput
            specs.append(LinearModuleSpec(
                src: "model.embed_tokens_per_layer",
                dst: "embedTokensPerLayer",
                inDim: pleTotal,
                required: true
            ))
            specs.append(LinearModuleSpec(
                src: "model.per_layer_model_projection",
                dst: "perLayerModelProjection",
                inDim: config.hiddenDim,
                required: true
            ))
        }
        for i in 0..<layers.count {
            let s = "model.layers.\(i)", d = "layers.\(i)"
            let layer = layers[i]
            let attnOutInDim = config.numHeads * config.headDim(forLayer: i)
            let ffnIntermediate = config.intermediateSize(forLayer: i)
            specs.append(LinearModuleSpec(src: "\(s).self_attn.q_proj", dst: "\(d).attention.query_proj", inDim: config.hiddenDim, required: true))
            if layer.hasKV {
                specs.append(LinearModuleSpec(src: "\(s).self_attn.k_proj", dst: "\(d).attention.key_proj", inDim: config.hiddenDim, required: true))
                if !layer.usesKEqV {
                    specs.append(LinearModuleSpec(src: "\(s).self_attn.v_proj", dst: "\(d).attention.value_proj", inDim: config.hiddenDim, required: true))
                }
            }
            specs.append(LinearModuleSpec(src: "\(s).self_attn.o_proj", dst: "\(d).attention.out_proj", inDim: attnOutInDim, required: true))
            specs.append(LinearModuleSpec(src: "\(s).mlp.gate_proj", dst: "\(d).ffn.gate", inDim: config.hiddenDim, required: true))
            specs.append(LinearModuleSpec(src: "\(s).mlp.up_proj", dst: "\(d).ffn.up", inDim: config.hiddenDim, required: true))
            specs.append(LinearModuleSpec(src: "\(s).mlp.down_proj", dst: "\(d).ffn.down", inDim: ffnIntermediate, required: true))

            if config.architecture == .gemma4, let g4 = config.gemma4 {
                if g4.enableMoEBlock {
                    specs += [
                        LinearModuleSpec(src: "\(s).router.proj", dst: "\(d).router.proj", inDim: config.hiddenDim, required: true),
                        LinearModuleSpec(src: "\(s).experts.switch_glu.gate_proj", dst: "\(d).experts.switch_glu.gate_proj", inDim: config.hiddenDim, required: true),
                        LinearModuleSpec(src: "\(s).experts.switch_glu.up_proj", dst: "\(d).experts.switch_glu.up_proj", inDim: config.hiddenDim, required: true),
                        LinearModuleSpec(src: "\(s).experts.switch_glu.down_proj", dst: "\(d).experts.switch_glu.down_proj", inDim: g4.moeIntermediateSize, required: true)
                    ]
                }
                if g4.hiddenSizePerLayerInput > 0 {
                    specs += [
                        LinearModuleSpec(src: "\(s).per_layer_input_gate", dst: "\(d).perLayerInputGate", inDim: config.hiddenDim, required: true),
                        LinearModuleSpec(src: "\(s).per_layer_projection", dst: "\(d).perLayerProjection", inDim: g4.hiddenSizePerLayerInput, required: true)
                    ]
                }
            }
        }
        if lmHead != nil {
            specs.append(LinearModuleSpec(src: "lm_head", dst: "lmHead", inDim: config.hiddenDim, required: true))
        }
        return specs
    }

    private func denseParameterMappings() -> [DenseParameterSpec] {
        var mappings: [DenseParameterSpec] = [
            DenseParameterSpec(src: "model.pos_embed.weight", dst: "posEmbedding.weight", required: false),
            DenseParameterSpec(src: "model.norm.weight", dst: "finalNorm.weight", required: true)
        ]
        if perLayerProjectionNorm != nil {
            mappings.append(DenseParameterSpec(
                src: "model.per_layer_projection_norm.weight",
                dst: "perLayerProjectionNorm.weight",
                required: true
            ))
        }
        for i in 0..<layers.count {
            let s = "model.layers.\(i)", d = "layers.\(i)"
            let layer = layers[i]
            mappings.append(DenseParameterSpec(src: "\(s).input_layernorm.weight", dst: "\(d).norm1.weight", required: true))
            if config.architecture == .gemma2 || config.architecture == .gemma3 || config.architecture == .gemma4 {
                mappings += [
                    DenseParameterSpec(src: "\(s).post_attention_layernorm.weight", dst: "\(d).postAttentionNorm.weight", required: true),
                    DenseParameterSpec(src: "\(s).pre_feedforward_layernorm.weight", dst: "\(d).norm2.weight", required: true),
                    DenseParameterSpec(src: "\(s).post_feedforward_layernorm.weight", dst: "\(d).postFFNNorm.weight", required: true)
                ]
            } else {
                mappings.append(DenseParameterSpec(src: "\(s).post_attention_layernorm.weight", dst: "\(d).norm2.weight", required: true))
            }
            if layer.qNorm != nil {
                mappings.append(DenseParameterSpec(src: "\(s).self_attn.q_norm.weight", dst: "\(d).qNorm.weight", required: true))
            }
            if layer.kNorm != nil {
                mappings.append(DenseParameterSpec(src: "\(s).self_attn.k_norm.weight", dst: "\(d).kNorm.weight", required: true))
            }
            if layer.router != nil {
                mappings += [
                    DenseParameterSpec(src: "\(s).router.scale", dst: "\(d).router.scale", required: true),
                    DenseParameterSpec(src: "\(s).router.per_expert_scale", dst: "\(d).router.per_expert_scale", required: true),
                    DenseParameterSpec(src: "\(s).post_feedforward_layernorm_1.weight", dst: "\(d).postFFNNorm1.weight", required: true),
                    DenseParameterSpec(src: "\(s).pre_feedforward_layernorm_2.weight", dst: "\(d).preFFNNorm2.weight", required: true),
                    DenseParameterSpec(src: "\(s).post_feedforward_layernorm_2.weight", dst: "\(d).postFFNNorm2.weight", required: true)
                ]
            }
            if layer.postPerLayerInputNorm != nil {
                mappings.append(DenseParameterSpec(src: "\(s).post_per_layer_input_norm.weight", dst: "\(d).postPerLayerInputNorm.weight", required: true))
            }
            if config.architecture == .gemma4 {
                mappings.append(DenseParameterSpec(src: "\(s).layer_scalar", dst: "\(d).layer_scalar", required: false))
            }
        }
        return mappings
    }

    private func loadDenseParameters(
        _ weights: [String: MLXArray],
        params: inout [String: MLXArray],
        missing: inout [String]
    ) {
        for spec in denseParameterMappings() {
            if let value = weights[spec.src] {
                params[spec.dst] = value
            } else if spec.required {
                missing.append(spec.src)
            }
        }
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
    /// Loads weights from a dictionary of QuantizedTensors into the model's sub-modules.
    ///
    /// - Important: GGUF `QuantizedTensor` inputs are expanded with `dequantized()` before loading, so this overload
    ///   is a legacy fallback that does **not** keep weights packed. For packed 4/8-bit MLX checkpoints use
    ///   `loadWeights(_: [String: MLXArray])` (auto-detects `.scales`/`.biases`) or `load(directory:maxSeqLen:)`.
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
    ///
    /// Checkpoints that contain `<module>.scales` entries (MLX affine quantization)
    /// are detected automatically and loaded through
    /// ``loadQuantizedWeights(_:quantization:)``, which keeps the weights packed in
    /// memory instead of expanding them to dense floating point.
    @discardableResult
    public func loadWeights(_ weights: [String: MLXArray]) -> [String] {
        let normalized = CheckpointKeyNormalizer.normalizeWeights(weights)
        let hasQuantizedHint = !pendingQuantizedTensors.isEmpty ||
            !quantizedModulePaths.isEmpty ||
            normalized.keys.contains(where: { $0.hasSuffix(".scales") || $0.hasSuffix(".biases") }) ||
            normalized.values.contains(where: { $0.dtype == .uint32 })
        if hasQuantizedHint {
            return loadQuantizedWeightsNormalized(normalized, quantization: nil)
        }

        ensureLMHeadIfPresent(in: normalized)
        var params: [String: MLXArray] = [:]
        var missing: [String] = []

        for m in linearModuleSpecs() {
            let weightKey = m.src + ".weight"
            if let weight = normalized[weightKey] {
                params[m.dst + ".weight"] = weight
            } else if m.required {
                missing.append(weightKey)
            }
            if let bias = normalized[m.src + ".bias"] {
                params[m.dst + ".bias"] = bias
            }
        }
        loadDenseParameters(normalized, params: &params, missing: &missing)

        if !params.isEmpty {
            self.update(parameters: NestedDictionary.unflattened(params))
        }
        compiledForwardCache.removeAll()
        return missing
    }

    // MARK: - Native quantized loading

    /// Loads an MLX-format affine-quantized checkpoint without dequantizing it.
    ///
    /// Every projection that has a `<module>.scales` entry is swapped for an MLX
    /// quantized layer that runs `quantized_matmul` directly on the packed weights,
    /// so an 8B model stays near 4.5–5.5 GB in 4-bit instead of expanding to about
    /// 15 GB of Float16. Modules without `scales` (for example norms) load densely.
    /// The embedding table is quantized as well when `model.embed_tokens.scales` exists.
    ///
    /// When a sharded checkpoint splits `.weight`, `.scales`, or `.biases` across
    /// calls, partial module tensors are staged internally until the complete affine tensor set
    /// arrives in a later shard, while still reporting the module's weight as missing
    /// until loading completes.
    ///
    /// - Parameters:
    ///   - weights: Tensors keyed by HuggingFace Llama names, for example the output of
    ///     ``SafeTensorsParser/parse(url:)``.
    ///   - quantization: Group size and bit width. When `nil` they are inferred from the
    ///     tensor shapes, which are unambiguous because the input width of every
    ///     projection is known from the model configuration.
    /// - Returns: Source keys that were missing or whose shapes did not match the
    ///   quantization parameters; an empty array means a complete load.
    @discardableResult
    public func loadQuantizedWeights(
        _ weights: [String: MLXArray],
        quantization: AffineQuantizationConfig? = nil
    ) -> [String] {
        loadQuantizedWeightsNormalized(
            CheckpointKeyNormalizer.normalizeWeights(weights),
            quantization: quantization
        )
    }

    private func loadQuantizedWeightsNormalized(
        _ weights: [String: MLXArray],
        quantization: AffineQuantizationConfig?
    ) -> [String] {
        let combined = pendingQuantizedTensors.isEmpty
            ? weights
            : pendingQuantizedTensors.merging(weights) { _, incoming in incoming }

        ensureLMHeadIfPresent(in: combined)
        let modules = linearModuleSpecs()

        // Infer or validate the quantization parameters from any complete quantized pair.
        var resolved = quantization ?? activeQuantization
        if resolved == nil,
           let first = (modules.first(where: {
               !$0.src.hasSuffix(".router.proj") &&
               combined[$0.src + ".scales"] != nil && combined[$0.src + ".weight"] != nil
           }) ?? modules.first(where: {
               combined[$0.src + ".scales"] != nil && combined[$0.src + ".weight"] != nil
           })),
           let packed = combined[first.src + ".weight"],
           let scales = combined[first.src + ".scales"],
           packed.ndim >= 1, scales.ndim >= 1, packed.dim(-1) > 0, scales.dim(-1) > 0,
           (packed.dim(-1) * 32) % first.inDim == 0, first.inDim % scales.dim(-1) == 0 {
            var inferred = AffineQuantizationConfig(
                groupSize: first.inDim / scales.dim(-1),
                bits: packed.dim(-1) * 32 / first.inDim
            )
            inferred.allowPerModuleInference = true
            resolved = inferred
        }
        let quant = resolved ?? AffineQuantizationConfig()

        var params: [String: MLXArray] = [:]
        var missing: [String] = []
        var newQuantizedSpecs: [String: (groupSize: Int, bits: Int)] = [:]

        func clearPending(_ keys: String...) {
            for key in keys { pendingQuantizedTensors.removeValue(forKey: key) }
        }

        for m in modules {
            let weightKey = m.src + ".weight"
            let scalesKey = m.src + ".scales"
            let biasesKey = m.src + ".biases"
            let biasKey = m.src + ".bias"

            let weight = combined[weightKey]
            let scales = combined[scalesKey]
            let biases = combined[biasesKey]
            let bias = combined[biasKey]

            if let scales, let packed = weight, let biases {
                var effectiveGroup = quant.groupSize
                var effectiveBits = quant.bits
                if let override = quant.moduleOverrides[m.src] {
                    effectiveGroup = override.groupSize
                    effectiveBits = override.bits
                } else if (quantization == nil || quant.allowPerModuleInference),
                          packed.ndim >= 1, scales.ndim >= 1, packed.dim(-1) > 0, scales.dim(-1) > 0 {
                    let defaultPackedOK = packed.dim(-1) * 32 == m.inDim * effectiveBits
                    let defaultScalesOK = scales.dim(-1) * effectiveGroup == m.inDim
                    if !(defaultPackedOK && defaultScalesOK),
                       (packed.dim(-1) * 32) % m.inDim == 0, m.inDim % scales.dim(-1) == 0 {
                        let inferredBits = (packed.dim(-1) * 32) / m.inDim
                        let inferredGroup = m.inDim / scales.dim(-1)
                        if [2, 3, 4, 5, 6, 8].contains(inferredBits), inferredGroup > 0 {
                            effectiveBits = inferredBits
                            effectiveGroup = inferredGroup
                        }
                    }
                }
                let packedOK = packed.ndim >= 1 && packed.dim(-1) * 32 == m.inDim * effectiveBits
                let scalesOK = scales.ndim >= 1 && scales.dim(-1) * effectiveGroup == m.inDim
                clearPending(weightKey, scalesKey, biasesKey, biasKey)
                guard packedOK && scalesOK else {
                    missing.append(weightKey + " (incompatible quantization)")
                    continue
                }
                if !quantizedModulePaths.contains(m.dst) {
                    newQuantizedSpecs[m.dst] = (groupSize: effectiveGroup, bits: effectiveBits)
                }
                quantizedModulePaths.insert(m.dst)
                params[m.dst + ".weight"] = packed
                params[m.dst + ".scales"] = scales
                params[m.dst + ".biases"] = biases
                if let bias { params[m.dst + ".bias"] = bias }
            } else if quantizedModulePaths.contains(m.dst) {
                if let biases {
                    params[m.dst + ".biases"] = biases
                    pendingQuantizedTensors.removeValue(forKey: biasesKey)
                }
                if let bias {
                    params[m.dst + ".bias"] = bias
                    pendingQuantizedTensors.removeValue(forKey: biasKey)
                }
                if m.required && weight == nil {
                    missing.append(weightKey)
                }
            } else if scales != nil || biases != nil || weight?.dtype == .uint32 {
                if let w = weight, w.dtype == .uint32 { pendingQuantizedTensors[weightKey] = w }
                if let scales { pendingQuantizedTensors[scalesKey] = scales }
                if let biases { pendingQuantizedTensors[biasesKey] = biases }
                if let bias { pendingQuantizedTensors[biasKey] = bias }
                if m.required {
                    missing.append(weightKey)
                }
            } else if let denseWeight = weight {
                params[m.dst + ".weight"] = denseWeight
                if let bias { params[m.dst + ".bias"] = bias }
                clearPending(weightKey, scalesKey, biasesKey, biasKey)
            } else if m.required {
                missing.append(weightKey)
            }
        }

        loadDenseParameters(weights, params: &params, missing: &missing)

        if !newQuantizedSpecs.isEmpty {
            activeQuantization = quant
            // Quantize non-array modules on `self` and each touched `layers[i]` directly:
            // upstream MLXNN's `Module.update(modules:)` traps if sparse array holes (`.none`) appear in `layers`.
            var topLevelSpecs: [String: (groupSize: Int, bits: Int)] = [:]
            var layerSubSpecs: [Int: [String: (groupSize: Int, bits: Int)]] = [:]
            for (path, spec) in newQuantizedSpecs {
                if path.hasPrefix("layers.") {
                    let tail = path.dropFirst("layers.".count)
                    if let dot = tail.firstIndex(of: "."),
                       let layerIdx = Int(tail[..<dot]),
                       layers.indices.contains(layerIdx) {
                        let subPath = String(tail[tail.index(after: dot)...])
                        layerSubSpecs[layerIdx, default: [:]][subPath] = spec
                    }
                } else {
                    topLevelSpecs[path] = spec
                }
            }
            if !topLevelSpecs.isEmpty {
                quantize(model: self) { path, _ in
                    guard let spec = topLevelSpecs[path] else { return nil }
                    return (groupSize: spec.groupSize, bits: spec.bits, mode: .affine)
                }
            }
            for (layerIdx, subSpecs) in layerSubSpecs.sorted(by: { $0.key < $1.key }) {
                quantize(model: layers[layerIdx]) { path, _ in
                    guard let spec = subSpecs[path] else { return nil }
                    return (groupSize: spec.groupSize, bits: spec.bits, mode: .affine)
                }
            }
        }
        if !params.isEmpty {
            self.update(parameters: NestedDictionary.unflattened(params))
        }
        compiledForwardCache.removeAll()
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

    /// Generates decoded text for a chat conversation without completion metadata.
    ///
    /// Formats `messages` using `template` (or ``DecoderArchitecture/defaultChatTemplate(date:)``),
    /// merges the template's default stop sequences without duplication, and encodes cleanly without
    /// duplicating leading BOS tokens.
    ///
    /// - Parameters:
    ///   - messages: Conversational messages to render into a prompt.
    ///   - template: Optional chat template override; defaults to `config.architecture.defaultChatTemplate()`.
    ///   - options: Inference options configuring sampling, token budget, and stop signals.
    /// - Returns: An `AsyncStream` of decoded text segments.
    public func generate(
        messages: [ChatMessage],
        template: ChatTemplate? = nil,
        options: LLMOptions = LLMOptions()
    ) async throws -> AsyncStream<String> {
        let (prompt, effectiveOptions) = prepareChatGeneration(messages: messages, template: template, options: options)
        return try await generate(prompt: prompt, options: effectiveOptions)
    }

    /// Generates text chunks followed by one completion record for a chat conversation.
    ///
    /// Formats `messages` using `template` (or ``DecoderArchitecture/defaultChatTemplate(date:)``),
    /// merges the template's default stop sequences without duplication, and encodes cleanly without
    /// duplicating leading BOS tokens.
    ///
    /// - Parameters:
    ///   - messages: Conversational messages to render into a prompt.
    ///   - template: Optional chat template override; defaults to `config.architecture.defaultChatTemplate()`.
    ///   - options: Inference options configuring sampling, token budget, and stop signals.
    /// - Returns: Text chunks followed by one final completion record on normal termination.
    public func generateDetails(
        messages: [ChatMessage],
        template: ChatTemplate? = nil,
        options: LLMOptions = LLMOptions()
    ) -> AsyncThrowingStream<GenerationEvent, any Error> {
        let (prompt, effectiveOptions) = prepareChatGeneration(messages: messages, template: template, options: options)
        return generateDetails(prompt: prompt, options: effectiveOptions)
    }

    private func prepareChatGeneration(
        messages: [ChatMessage],
        template: ChatTemplate?,
        options: LLMOptions
    ) -> (prompt: String, options: LLMOptions) {
        let resolvedTemplate = template ?? config.architecture.defaultChatTemplate()
        let prompt = resolvedTemplate.render(messages: messages, addGenerationPrompt: true)
        var effectiveOptions = options
        for stop in resolvedTemplate.defaultStopSequences where !effectiveOptions.stopSequences.contains(stop) {
            effectiveOptions.stopSequences.append(stop)
        }
        return (prompt, effectiveOptions)
    }

    private func generationPrompt(_ prompt: String, options: LLMOptions) throws -> [Int] {
        guard options.maxTokens >= 0 else { throw GenerationError.invalidMaxTokens(options.maxTokens) }
        guard config.maxSeqLen > 0 else { throw GenerationError.invalidContextLimit(config.maxSeqLen) }
        var tokens = Self.encodePrompt(prompt, using: tokenizer)
        if tokens.isEmpty { tokens = [0] }
        guard tokens.count <= config.maxSeqLen else {
            throw GenerationError.promptTooLong(promptTokens: tokens.count, capacity: config.maxSeqLen)
        }
        return tokens
    }

    private static func encodePrompt(_ prompt: String, using tokenizer: any Tokenizer) -> [Int] {
        if let sp = tokenizer as? SentencePieceBPETokenizer {
            let bosTokens = sp.encode(text: "", addSpecialTokens: true)
            if bosTokens.count == 1 {
                let bosText = sp.decode(tokens: bosTokens)
                if (bosText == "<bos>" || bosText == "<s>"),
                   prompt.hasPrefix(bosText),
                   sp.encode(text: bosText, addSpecialTokens: false) == bosTokens {
                    return sp.encode(text: prompt, addSpecialTokens: false)
                }
            }
        }
        return tokenizer.encode(text: prompt)
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

    private static func earliestStopRange(in text: String, stopSequences: [String]) -> Range<String.Index>? {
        var earliest: Range<String.Index>? = nil
        for sequence in stopSequences {
            if let range = text.range(of: sequence) {
                if earliest == nil || range.lowerBound < earliest!.lowerBound {
                    earliest = range
                }
            }
        }
        return earliest
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
            let effectiveEOS = config.eosTokenIDs.union(options.stopTokenIDs)
            let activeStopSequences: [String] = options.stopSequences.filter { (seq: String) -> Bool in !seq.isEmpty }
            let maxHoldback: Int = activeStopSequences.isEmpty
                ? 0
                : max(1, activeStopSequences.map { (seq: String) -> Int in seq.count }.max() ?? 1) - 1
            var pendingText = ""
            var stoppedBySequence = false

            if budget > 0 && !Task.isCancelled {
                var tokens = promptTokens
                let caches = layers.map { _ in KVCache() }
                if var lastLogits = prefill(tokens, caches: caches) {
                    while generated < budget {
                        if Task.isCancelled { break }
                        let next = Sampler.sample(logits: lastLogits, options: options, pastTokens: tokens)
                        if effectiveEOS.contains(next) {
                            reason = .stop
                            break
                        }
                        let decoded = textDecoder.append(next)
                        if effectiveEOS.isEmpty, let decoded, decoded.isEmpty || decoded == "<unk>" {
                            reason = .stop
                            break
                        }
                        generated += 1
                        tokens.append(next)
                        if let decoded, !decoded.isEmpty {
                            if activeStopSequences.isEmpty {
                                emit(.chunk(decoded))
                            } else {
                                pendingText += decoded
                                if let matchRange = Self.earliestStopRange(in: pendingText, stopSequences: activeStopSequences) {
                                    let prefix = String(pendingText[..<matchRange.lowerBound])
                                    if !prefix.isEmpty { emit(.chunk(prefix)) }
                                    pendingText = ""
                                    reason = .stop
                                    stoppedBySequence = true
                                    break
                                }
                                if pendingText.count > maxHoldback {
                                    let emitCount = pendingText.count - maxHoldback
                                    let splitIdx = pendingText.index(pendingText.startIndex, offsetBy: emitCount)
                                    let safePrefix = String(pendingText[..<splitIdx])
                                    pendingText = String(pendingText[splitIdx...])
                                    if !safePrefix.isEmpty { emit(.chunk(safePrefix)) }
                                }
                            }
                        }
                        if generated == budget || Task.isCancelled { break }

                        let offset = caches.first?.count ?? (tokens.count - 1)
                        let step = forward(MLXArray([next]).expandedDimensions(axis: 0), caches: caches, offset: offset)
                        lastLogits = step[0, 0]
                        eval(lastLogits)
                    }
                }
            }

            if Task.isCancelled {
                reason = .cancelled
            } else if !stoppedBySequence {
                let tail = textDecoder.finish()
                if activeStopSequences.isEmpty {
                    if !tail.isEmpty { emit(.chunk(tail)) }
                } else {
                    if !tail.isEmpty { pendingText += tail }
                    if !pendingText.isEmpty {
                        if let matchRange = Self.earliestStopRange(in: pendingText, stopSequences: activeStopSequences) {
                            let prefix = String(pendingText[..<matchRange.lowerBound])
                            if !prefix.isEmpty { emit(.chunk(prefix)) }
                            reason = .stop
                        } else {
                            emit(.chunk(pendingText))
                        }
                        pendingText = ""
                    }
                }
            }
            emit(.info(GenerationCompletionInfo(promptTokenCount: promptTokens.count,
                generationTokenCount: generated, stopReason: reason)))
        }
    }
}

#endif // os(macOS)
