#if os(macOS)
import MLX
import MLXNN

// Gemma 2 and Gemma 3 store an offset from unity and multiply in float32 before casting back.
final class GemmaRMSNorm: RMSNorm {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let one = MLXArray(Float(1)).asType(weight.dtype)
        let shifted = one + weight
        let scale = shifted.dtype == x.dtype ? shifted : shifted.asType(x.dtype)
        return MLXFast.rmsNorm(x, weight: scale, eps: eps)
    }
}

/// Gemma 4 `v_norm`: RMS normalization without a learnable weight parameter.
final class ParameterlessRMSNorm: Module, UnaryLayer {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let value = x.asType(.float32)
        return (value * rsqrt(mean(value * value, axis: -1, keepDims: true) + eps)).asType(x.dtype)
    }
}

/// Matches mlx-lm's `clip_residual`: guards float16 residual additions against overflow.
func clipResidual(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
    guard x.dtype == .float16 else { return x + y }
    let bound = Float(Float16.greatestFiniteMagnitude)
    return clip(x.asType(.float32) + y.asType(.float32), min: -bound, max: bound).asType(.float16)
}

extension LLMConfig {
    func makeNorm(dimensions: Int? = nil) -> RMSNorm {
        let dims = dimensions ?? hiddenDim
        switch architecture {
        case .gemma2, .gemma3:
            return GemmaRMSNorm(dimensions: dims, eps: rmsNormEps)
        case .llama, .mistral, .gemma4:
            return RMSNorm(dimensions: dims, eps: rmsNormEps)
        }
    }
}
#endif
