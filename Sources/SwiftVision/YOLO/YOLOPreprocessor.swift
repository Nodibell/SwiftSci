import Foundation
import MLX

/// Letterbox image preprocessor for YOLOv8 neural network inference.
/// Preserves aspect ratio, resizes to target 640x640 resolution, pads with gray `(114, 114, 114)` pixels,
/// and normalizes to `[0, 1]` tensor shape `[1, 640, 640, 3]`.
public struct YOLOPreprocessor {
    /// Target image tensor width in pixels (default 640).
    public let targetWidth: Int
    /// Target image tensor height in pixels (default 640).
    public let targetHeight: Int
    /// Normalized padding background color value (default 114/255).
    public let paddingColor: Double

    /// Creates a YOLOPreprocessor.
    /// - Parameters:
    ///   - targetWidth: Target width (default 640).
    ///   - targetHeight: Target height (default 640).
    ///   - paddingColor: Normalized gray padding background value (default 114/255).
    public init(targetWidth: Int = 640, targetHeight: Int = 640, paddingColor: Double = 114.0 / 255.0) {
        self.targetWidth = targetWidth
        self.targetHeight = targetHeight
        self.paddingColor = paddingColor
    }

    /// Preprocesses input `ImageDataset` into a normalized `MLXArray` tensor [1, 640, 640, 3].
    /// - Parameter image: The input image dataset.
    /// - Returns: `MLXArray` of shape `[1, 640, 640, 3]` in `NHWC` layout.
    public func preprocess(image: ImageDataset) -> MLXArray {
        let w = Double(image.width)
        let h = Double(image.height)
        let scale = min(Double(targetWidth) / max(1.0, w), Double(targetHeight) / max(1.0, h))

        let newW = min(targetWidth, max(1, Int(round(w * scale))))
        let newH = min(targetHeight, max(1, Int(round(h * scale))))

        let padX = (targetWidth - newW) / 2
        let padY = (targetHeight - newH) / 2

        var buffer = [Float](repeating: Float(paddingColor), count: targetWidth * targetHeight * 3)

        // Bilinear or nearest sampling from input image
        for ty in 0..<newH {
            let sy = Int(min(h - 1, max(0.0, Double(ty) / scale)))
            let outY = padY + ty

            for tx in 0..<newW {
                let sx = Int(min(w - 1, max(0.0, Double(tx) / scale)))
                let outX = padX + tx

                let inIdx = sy * image.width + sx
                let val = Float(inIdx < image.data.count ? image.data[inIdx] : 0.0)

                let outIdx = (outY * targetWidth + outX) * 3
                buffer[outIdx + 0] = val
                buffer[outIdx + 1] = val
                buffer[outIdx + 2] = val
            }
        }

        let array = MLXArray(buffer, [1, targetHeight, targetWidth, 3])
        return array
    }
}
