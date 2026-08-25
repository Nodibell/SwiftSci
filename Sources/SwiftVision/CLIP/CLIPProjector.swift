#if os(macOS)
import Foundation
import MLX
import MLXNN

/// Multimodal Vision-Language Feature Projector for CLIP zero-shot classification and cross-modal retrieval.
///
/// `CLIPProjector` maps raw visual and textual embedding representations into a shared latent metric space
/// where cosine distances correlate with semantic alignment.
public final class CLIPProjector: Module, @unchecked Sendable {
    /// Dimension of input visual feature embeddings.
    public let visualDimensions: Int
    /// Dimension of input text feature embeddings.
    public let textDimensions: Int
    /// Dimension of shared multimodal projection space.
    public let projectionDimensions: Int

    /// Visual feature projection layer.
    @ModuleInfo public var visualProjection: Linear
    /// Text feature projection layer.
    @ModuleInfo public var textProjection: Linear
    /// Logit scale parameter (temperature scaling for similarity scores).
    public let logitScale: MLXArray

    /// Initializes a multimodal CLIP projector.
    ///
    /// - Parameters:
    ///   - visualDimensions: Visual encoder hidden size (default: 768).
    ///   - textDimensions: Text encoder hidden size (default: 512).
    ///   - projectionDimensions: Shared latent embedding size (default: 512).
    ///   - initialTemperature: Initial softmax temperature parameter (default: 0.07).
    public init(
        visualDimensions: Int = 768,
        textDimensions: Int = 512,
        projectionDimensions: Int = 512,
        initialTemperature: Float = 0.07
    ) {
        self.visualDimensions = visualDimensions
        self.textDimensions = textDimensions
        self.projectionDimensions = projectionDimensions

        self.visualProjection = Linear(visualDimensions, projectionDimensions, bias: false)
        self.textProjection = Linear(textDimensions, projectionDimensions, bias: false)
        let logScaleVal = log(1.0 / initialTemperature)
        self.logitScale = MLXArray(Float(logScaleVal))

        super.init()
    }

    /// Projects and L2-normalizes visual feature vectors into the shared latent space.
    ///
    /// - Parameter visualFeatures: Visual encoder activations `[..., visualDimensions]`.
    /// - Returns: Normalized image embeddings `[..., projectionDimensions]`.
    public func projectVision(_ visualFeatures: MLXArray) -> MLXArray {
        let projected = visualProjection(visualFeatures)
        let norm = sqrt(sum(projected * projected, axis: -1, keepDims: true) + Float(1e-12))
        return projected / norm
    }

    /// Projects and L2-normalizes text feature vectors into the shared latent space.
    ///
    /// - Parameter textFeatures: Text encoder activations `[..., textDimensions]`.
    /// - Returns: Normalized text embeddings `[..., projectionDimensions]`.
    public func projectText(_ textFeatures: MLXArray) -> MLXArray {
        let projected = textProjection(textFeatures)
        let norm = sqrt(sum(projected * projected, axis: -1, keepDims: true) + Float(1e-12))
        return projected / norm
    }

    /// Computes temperature-scaled cosine similarity logits between images and texts.
    ///
    /// - Parameters:
    ///   - visualFeatures: Visual encoder activations `[numImages, visualDimensions]`.
    ///   - textFeatures: Text encoder activations `[numTexts, textDimensions]`.
    /// - Returns: Logits matrix `[numImages, numTexts]`.
    public func similarity(visualFeatures: MLXArray, textFeatures: MLXArray) -> MLXArray {
        let imgEmb = projectVision(visualFeatures)
        let txtEmb = projectText(textFeatures)

        // Dot product: [numImages, projectionDimensions] @ [projectionDimensions, numTexts]
        let logits = matmul(imgEmb, txtEmb.transposed(1, 0))
        return logits * exp(logitScale)
    }

    /// Predicts zero-shot classification probabilities for candidate text labels given image representations.
    ///
    /// - Parameters:
    ///   - visualFeatures: Visual activations for query images `[numImages, visualDimensions]`.
    ///   - candidateTextFeatures: Text activations for label prompts `[numClasses, textDimensions]`.
    /// - Returns: Classification probability matrix `[numImages, numClasses]`.
    public func zeroShotClassify(visualFeatures: MLXArray, candidateTextFeatures: MLXArray) -> MLXArray {
        let sim = similarity(visualFeatures: visualFeatures, textFeatures: candidateTextFeatures)
        return softmax(sim, axis: -1)
    }
}
#endif
