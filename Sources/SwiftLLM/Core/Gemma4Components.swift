#if os(macOS)
import Foundation
import MLX
import MLXNN

// MARK: - Gemma 4 Configuration

struct Gemma4Config: Sendable, Equatable {
    struct RoPEParameters: Sendable, Equatable {
        var ropeTheta: Float
        var ropeType: String
        var partialRotaryFactor: Float
        var scale: Float
    }

    var layerTypes: [String]
    var globalHeadDim: Int
    var attentionKEqV: Bool
    var numGlobalKVHeads: Int?
    var numKVSharedLayers: Int
    var useDoubleWideMLP: Bool
    var hiddenSizePerLayerInput: Int
    var vocabSizePerLayerInput: Int
    var enableMoEBlock: Bool
    var numExperts: Int
    var topKExperts: Int
    var moeIntermediateSize: Int
    var slidingRoPE: RoPEParameters
    var fullRoPE: RoPEParameters
}

// MARK: - Switch Linear & Quantized Switch Linear (MoE Experts)

private func gatherSort(_ x: MLXArray, _ indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let fanout = indices.dim(-1)
    let flatIndices = indices.flattened()
    let order = argSort(flatIndices, axis: 0)
    let invOrder = argSort(order, axis: 0)
    let flatTokens = x.flattened(start: 0, end: -3)
    return (flatTokens[order / fanout], flatIndices[order], invOrder)
}

private func scatterUnsort(_ x: MLXArray, _ invOrder: MLXArray, shape: [Int]) -> MLXArray {
    unflatten(x[invOrder], axis: 0, shape: shape)
}

/// Expert-batched linear layer supporting dense (`gatherMM`) and packed affine quantized (`gatherQuantizedMM`) execution.
class SwitchLinear: Module, Quantizable {
    @ParameterInfo public var weight: MLXArray

    init(inputDims: Int, outputDims: Int, numExperts: Int) {
        let bound = (1.0 / Float(inputDims)).squareRoot()
        self.weight = MLXRandom.uniform(
            low: -bound,
            high: bound,
            [numExperts, outputDims, inputDims]
        )
        super.init()
    }

    init(weight: MLXArray) {
        self.weight = weight
        super.init()
    }

    var inputDims: Int { weight.dim(2) }
    var outputDims: Int { weight.dim(1) }
    var numExperts: Int { weight.dim(0) }

    func callAsFunction(_ x: MLXArray, indices: MLXArray, sortedIndices: Bool = false) -> MLXArray {
        gatherMM(
            x,
            weight.swappedAxes(-1, -2),
            rhsIndices: indices,
            sortedIndices: sortedIndices
        )
    }

    func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(weight: weight, groupSize: groupSize, bits: bits, mode: mode)
    }
}

final class QuantizedSwitchLinear: SwitchLinear, Quantized {
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode
    @ParameterInfo public var scales: MLXArray
    @ParameterInfo public var biases: MLXArray?

    init(weight: MLXArray, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        let (quantizedWeight, scales, biases) = MLX.quantized(
            weight, groupSize: groupSize, bits: bits, mode: mode
        )
        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases
        super.init(weight: quantizedWeight)
        self.freeze()
    }

    override var inputDims: Int { scales.dim(2) * groupSize }
    override var outputDims: Int { weight.dim(1) }
    override var numExperts: Int { weight.dim(0) }

    override func callAsFunction(_ x: MLXArray, indices: MLXArray, sortedIndices: Bool = false) -> MLXArray {
        gatherQuantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: mode,
            sortedIndices: sortedIndices
        )
    }
}

// MARK: - Gemma 4 MoE Modules

final class Gemma4SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    init(inputDims: Int, hiddenDims: Int, numExperts: Int) {
        self._gateProj.wrappedValue = SwitchLinear(inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts)
        self._upProj.wrappedValue = SwitchLinear(inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts)
        self._downProj.wrappedValue = SwitchLinear(inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        var expanded = expandedDimensions(x, axes: [-3, -2])
        let doSort = indices.size >= 64
        var idx = indices
        var invOrder: MLXArray?
        if doSort {
            (expanded, idx, invOrder) = gatherSort(expanded, indices)
        }
        let xUp = upProj(expanded, indices: idx, sortedIndices: doSort)
        let xGate = gateProj(expanded, indices: idx, sortedIndices: doSort)
        var out = downProj(geluApproximate(xGate) * xUp, indices: idx, sortedIndices: doSort)
        if let invOrder {
            out = scatterUnsort(out, invOrder, shape: indices.shape)
        }
        return squeezed(out, axis: -2)
    }
}

final class Gemma4Router: Module {
    @ModuleInfo var proj: Linear
    @ParameterInfo var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") var perExpertScale: MLXArray
    private let eps: Float
    private let topKExperts: Int
    private let rootSize: Float

    init(hiddenDim: Int, numExperts: Int, topKExperts: Int, eps: Float) {
        self.proj = Linear(hiddenDim, numExperts, bias: false)
        self._scale.wrappedValue = MLXArray.ones([hiddenDim])
        self._perExpertScale.wrappedValue = MLXArray.ones([numExperts])
        self.eps = eps
        self.topKExperts = topKExperts
        self.rootSize = 1.0 / Float(hiddenDim).squareRoot()
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let weight = (scale.asType(.float32) * rootSize).asType(x.dtype)
        let normed = MLXFast.rmsNorm(x, weight: weight, eps: eps)
        let scores = proj(normed)
        let partitioned = argPartition(scores, kth: -topKExperts, axis: -1)
        let numExperts = scores.dim(-1)
        let topKIndices = partitioned[.ellipsis, (numExperts - topKExperts)...]
        var topKWeights = takeAlong(scores, topKIndices, axis: -1)
        topKWeights = softmax(topKWeights, axis: -1)
        topKWeights = topKWeights * perExpertScale[topKIndices].asType(topKWeights.dtype)
        return (topKIndices, topKWeights)
    }
}

final class Gemma4Experts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: Gemma4SwitchGLU

    init(hiddenDim: Int, moeIntermediateSize: Int, numExperts: Int) {
        self._switchGLU.wrappedValue = Gemma4SwitchGLU(
            inputDims: hiddenDim,
            hiddenDims: moeIntermediateSize,
            numExperts: numExperts
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, topKIndices: MLXArray, topKWeights: MLXArray) -> MLXArray {
        let weights = topKWeights.expandedDimensions(axis: -1)
        let y = switchGLU(x, indices: topKIndices)
        return (weights * y).sum(axis: -2)
    }
}

// MARK: - Checkpoint Key Normalization

enum CheckpointKeyNormalizer {
    private static let ignoredPrefixes = [
        "vision_tower.", "multi_modal_projector.", "audio_tower.",
        "embed_audio.", "embed_vision.", "vision_embedder.",
        "model.vision_tower.", "model.multi_modal_projector.", "model.audio_tower.",
        "model.embed_audio.", "model.embed_vision.", "model.vision_embedder."
    ]

    static func normalizeKey(_ raw: String) -> String? {
        for prefix in ignoredPrefixes where raw.hasPrefix(prefix) {
            return nil
        }
        var key = raw
        if key.hasPrefix("model.language_model.model.") {
            key = "model." + key.dropFirst("model.language_model.model.".count)
        } else if key.hasPrefix("model.language_model.") {
            key = "model." + key.dropFirst("model.language_model.".count)
        } else if key.hasPrefix("language_model.model.") {
            key = "model." + key.dropFirst("language_model.model.".count)
        } else if key == "language_model.norm.weight" {
            key = "model.norm.weight"
        } else if key.hasPrefix("language_model.lm_head.") {
            key = String(key.dropFirst("language_model.".count))
        } else if key.hasPrefix("language_model.") {
            key = "model." + key.dropFirst("language_model.".count)
        }
        for prefix in ignoredPrefixes where key.hasPrefix(prefix) {
            return nil
        }
        if key.contains(".rotary_emb") || key.hasSuffix(".input_max") || key.hasSuffix(".input_min")
            || key.hasSuffix(".output_max") || key.hasSuffix(".output_min") {
            return nil
        }
        if key.hasPrefix("model.lm_head.") {
            key = String(key.dropFirst("model.".count))
        }
        return key
    }

    static func normalizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        result.reserveCapacity(weights.count)
        for (rawKey, tensor) in weights {
            guard let key = normalizeKey(rawKey) else { continue }
            if key.hasSuffix(".experts.gate_up_proj") && !key.contains(".switch_glu.") {
                let base = String(key.dropLast(".gate_up_proj".count))
                let parts = split(tensor, parts: 2, axis: -2)
                if parts.count == 2 {
                    result["\(base).switch_glu.gate_proj.weight"] = contiguous(parts[0])
                    result["\(base).switch_glu.up_proj.weight"] = contiguous(parts[1])
                }
            } else if key.hasSuffix(".experts.down_proj") && !key.contains(".switch_glu.") {
                let base = String(key.dropLast(".down_proj".count))
                result["\(base).switch_glu.down_proj.weight"] = tensor
            } else {
                result[key] = tensor
            }
        }
        return result
    }
}
#endif
