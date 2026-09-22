#if os(macOS)
import Foundation
import MLX
import MLXNN

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

    @ModuleInfo private var rope: RoPE

    /// Initializes a Rotary Positional Embedding module.
    /// - Parameters:
    ///   - dimensions: Head dimension to rotate.
    ///   - base: Angular frequency base (default `10_000.0`).
    ///   - scale: Position scale factor (default `1.0`).
    ///   - traditional: Use traditional rotary formulation (default `false`).
    public init(
        dimensions: Int,
        base: Float = 10_000.0,
        scale: Float = 1.0,
        traditional: Bool = false
    ) {
        self.dimensions = dimensions
        self.base = base
        self.scale = scale
        self.traditional = traditional
        self.rope = RoPE(dimensions: dimensions, traditional: traditional, base: base, scale: scale)
        super.init()
    }

    /// Applies rotary positional encoding with an optional position offset.
    /// - Parameters:
    ///   - x: Input tensor of shape `[batch, numHeads, seqLen, headDim]` or `[batch, seqLen, headDim]`.
    ///   - offset: Starting token position offset in the sequence (default `0`).
    /// - Returns: Rotated tensor with identical shape and type.
    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        rope(x, offset: offset)
    }

    /// Convenience alias matching `callAsFunction`.
    public func apply(to x: MLXArray, offset: Int = 0) -> MLXArray {
        callAsFunction(x, offset: offset)
    }
}
#endif
