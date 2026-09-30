#if os(macOS)
import Foundation
import MLX
import MLXNN

/// Wavelength-dependent frequency scaling used by Llama 3.1 and 3.2 checkpoints.
public struct Llama3RoPEScaling: Sendable, Equatable {
    /// Divisor for the lowest rotary frequencies.
    public let factor: Float
    /// Divisor of the original context length at the low-frequency wavelength boundary.
    public let lowFrequencyFactor: Float
    /// Divisor of the original context length at the high-frequency wavelength boundary.
    public let highFrequencyFactor: Float
    /// Context length before wavelength scaling.
    public let originalContextLength: Int

    /// Creates the three-band scaling configuration from checkpoint metadata.
    public init(
        factor: Float,
        lowFrequencyFactor: Float = 1,
        highFrequencyFactor: Float = 4,
        originalContextLength: Int = 8192
    ) {
        precondition(factor.isFinite && factor >= 1, "factor must be finite and at least one")
        precondition(lowFrequencyFactor.isFinite && lowFrequencyFactor > 0 &&
                     highFrequencyFactor.isFinite && highFrequencyFactor > lowFrequencyFactor,
                     "frequency factors must be finite, positive, and increasing")
        precondition(originalContextLength > 0, "originalContextLength must be positive")
        self.factor = factor
        self.lowFrequencyFactor = lowFrequencyFactor
        self.highFrequencyFactor = highFrequencyFactor
        self.originalContextLength = originalContextLength
    }

    func frequencyDenominators(dimensions: Int, base: Float) -> MLXArray {
        let powers = MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / Float(dimensions)
        let original = MLX.pow(MLXArray(base), powers)
        // Round pi to Float32 as in the checkpoint reference implementation.
        let wavelength = original * (2 * Float(Double.pi))
        let blend = clip(
            (Float(originalContextLength) / wavelength - lowFrequencyFactor) /
                (highFrequencyFactor - lowFrequencyFactor), min: 0, max: 1)
        return original / (blend + (1 - blend) / factor)
    }
}

/// Rotary Positional Embedding (RoPE) applying frequency-based rotations to query and key tensors.
///
/// Ref: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021).
public final class RoPEEmbedding: Module, @unchecked Sendable {
    /// Dimension of each attention head to rotate (must be even).
    public let dimensions: Int
    /// Base frequency (e.g. 10,000 for standard Llama, 500,000 for Llama 3).
    public let base: Float
    /// Scaling factor applied to frequencies.
    public let scale: Float
    /// Whether to use traditional interleaved layout.
    public let traditional: Bool

    /// Optional Llama wavelength-dependent scaling, applied in addition to `scale`.
    public let scaling: Llama3RoPEScaling?

    @ModuleInfo private var rope: RoPE
    // Derived constants are not model weights in MLX's module traversal.
    private let _frequencies: MLXArray?

    /// Initializes a Rotary Positional Embedding module.
    /// - Parameters:
    ///   - dimensions: Head dimension to rotate.
    ///   - base: Angular frequency base (default `10_000.0`).
    ///   - scale: Position scale factor (default `1.0`).
    ///   - traditional: Use traditional rotary formulation (default `false`).
    ///   - scaling: Llama wavelength-dependent scaling. `nil` preserves standard RoPE.
    public init(
        dimensions: Int,
        base: Float = 10_000.0,
        scale: Float = 1.0,
        traditional: Bool = false,
        scaling: Llama3RoPEScaling? = nil
    ) {
        self.dimensions = dimensions
        self.base = base
        self.scale = scale
        self.traditional = traditional
        self.scaling = scaling
        if let scaling {
            precondition(dimensions > 0 && dimensions.isMultiple(of: 2), "rotary dimensions must be positive and even")
            precondition(base.isFinite && base > 0, "base must be finite and positive")
            let frequencies = scaling.frequencyDenominators(dimensions: dimensions, base: base)
            eval(frequencies)
            self._frequencies = frequencies
        } else {
            self._frequencies = nil
        }
        self.rope = RoPE(dimensions: dimensions, traditional: traditional, base: base, scale: scale)
        super.init()
    }

    /// Applies rotary positional encoding with an optional position offset.
    /// - Parameters:
    ///   - x: Input tensor of shape `[batch, numHeads, seqLen, headDim]` or `[batch, seqLen, headDim]`.
    ///   - offset: Starting token position offset in the sequence (default `0`).
    /// - Returns: Rotated tensor with identical shape and type.
    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        guard let frequencies = _frequencies else { return rope(x, offset: offset) }
        // The pinned MLX single-token kernel needs independent rows in its head axis.
        let input = x.dim(-2) == 1 ? x.reshaped([1, -1, 1, x.dim(-1)]) : x
        return MLXFast.RoPE(input, dimensions: dimensions, traditional: traditional,
            base: nil, scale: scale, offset: offset, freqs: frequencies).reshaped(x.shape)
    }

    /// Convenience alias matching `callAsFunction`.
    public func apply(to x: MLXArray, offset: Int = 0) -> MLXArray {
        callAsFunction(x, offset: offset)
    }
}
#endif
