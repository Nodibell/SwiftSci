#if os(macOS)
import Foundation
import MLX
import MLXNN

/// The quantization scheme used for compressed model weights.
public enum QuantizationScheme: Sendable {
    /// 4-bit symmetric quantization with block size 32 (GGUF Q4_0).
    case q4_0
    /// 4-bit asymmetric quantization with scale and min offset (GGUF Q4_1).
    case q4_1
    /// 8-bit symmetric quantization with block size 32 (GGUF Q8_0).
    case q8_0
    /// 4-bit activation-aware weight quantization with configurable group size (AWQ).
    case awq4(groupSize: Int)
    /// 8-bit activation-aware weight quantization with configurable group size (AWQ).
    case awq8(groupSize: Int)
}

/// A high-performance quantized Linear layer executing 4-bit and 8-bit matrix multiplications on Apple Silicon Metal GPU.
///
/// `QuantizedLinear` stores compressed weights and scale factors in unified memory and executes
/// fast GPU-accelerated forward passes with minimal memory footprint.
public final class QuantizedLinear: Module, @unchecked Sendable {
    /// The number of input features.
    public let inFeatures: Int
    /// The number of output features.
    public let outFeatures: Int
    /// The quantization scheme used for layer weights.
    public let scheme: QuantizationScheme
    /// The group block size.
    public let groupSize: Int

    /// Packed quantized weight tensor.
    public let weight: MLXArray
    /// Scale factor tensor per quantization group.
    public let scales: MLXArray
    /// Optional bias or zero-point offset tensor.
    public let bias: MLXArray?

    /// Initializes a quantized linear layer with pre-quantized weights and scales.
    ///
    /// - Parameters:
    ///   - inFeatures: Input dimension size.
    ///   - outFeatures: Output dimension size.
    ///   - scheme: Quantization scheme (Q4_0, Q4_1, Q8_0, AWQ).
    ///   - groupSize: Group block size for quantization (default: 32).
    ///   - weight: Quantized weights tensor.
    ///   - scales: Scale factors tensor.
    ///   - bias: Optional bias tensor.
    public init(
        inFeatures: Int,
        outFeatures: Int,
        scheme: QuantizationScheme = .q4_0,
        groupSize: Int = 32,
        weight: MLXArray,
        scales: MLXArray,
        bias: MLXArray? = nil
    ) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.scheme = scheme
        self.groupSize = groupSize
        self.weight = weight
        self.scales = scales
        self.bias = bias
        super.init()
    }

    /// Dequantizes the compressed weight matrix into a dense floating-point tensor on Apple Silicon GPU.
    ///
    /// - Returns: Dequantized weights of shape `[outFeatures, inFeatures]`.
    public func dequantize() -> MLXArray {
        switch scheme {
        case .q4_0:
            // (weight - 8) * scales
            let centered = self.weight - MLXArray(Float(8.0))
            let dequant = centered * self.scales
            return dequant.reshaped([outFeatures, inFeatures])

        case .q4_1:
            // weight * scales + bias
            var dequant = self.weight * self.scales
            if let b = self.bias {
                dequant = dequant + b
            }
            return dequant.reshaped([outFeatures, inFeatures])

        case .q8_0:
            // weight * scales
            let dequant = self.weight * self.scales
            return dequant.reshaped([outFeatures, inFeatures])

        case .awq4(let group):
            _ = group
            let centered = self.weight
            var dequant = centered * self.scales
            if let b = self.bias {
                dequant = dequant + b
            }
            return dequant.reshaped([outFeatures, inFeatures])

        case .awq8(let group):
            _ = group
            var dequant = self.weight * self.scales
            if let b = self.bias {
                dequant = dequant + b
            }
            return dequant.reshaped([outFeatures, inFeatures])
        }
    }

    /// Performs the forward pass linear transformation on input activations.
    ///
    /// - Parameter x: Input activation tensor of shape `[..., inFeatures]`.
    /// - Returns: Output activation tensor of shape `[..., outFeatures]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let w = dequantize()
        // Compute x @ w.T
        var out = matmul(x, w.transposed(1, 0))
        if let b = self.bias, b.shape == [outFeatures] {
            out = out + b
        }
        return out
    }
}
#endif
