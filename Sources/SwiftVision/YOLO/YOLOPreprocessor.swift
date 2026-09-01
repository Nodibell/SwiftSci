import Foundation
import Accelerate
import Accelerate.vImage
import MLX

/// Letterbox image preprocessor for YOLOv8 neural network inference.
/// Preserves aspect ratio, resizes using Apple Accelerate `vImageScale_PlanarF` high-performance
/// bilinear resampling, pads with gray `(114, 114, 114)` pixels, and normalizes to `[0, 1]` tensor shape `[1, 640, 640, 3]`.
public struct YOLOPreprocessor: Sendable {
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

    /// Preprocesses input `ImageDataset` into a normalized `MLXArray` tensor [1, targetHeight, targetWidth, 3].
    /// - Parameter image: The input image dataset.
    /// - Returns: `MLXArray` of shape `[1, targetHeight, targetWidth, 3]` in `NHWC` layout.
    public func preprocess(image: ImageDataset) -> MLXArray {
        guard image.width > 0, image.height > 0 else {
            let buffer = [Float](repeating: Float(paddingColor), count: targetWidth * targetHeight * 3)
            return MLXArray(buffer, [1, targetHeight, targetWidth, 3])
        }

        let w = Double(image.width)
        let h = Double(image.height)
        let scale = min(Double(targetWidth) / max(1.0, w), Double(targetHeight) / max(1.0, h))

        let newW = min(targetWidth, max(1, Int(round(w * scale))))
        let newH = min(targetHeight, max(1, Int(round(h * scale))))

        let padX = (targetWidth - newW) / 2
        let padY = (targetHeight - newH) / 2

        var buffer = [Float](repeating: Float(paddingColor), count: targetWidth * targetHeight * 3)
        let srcPixelCount = image.width * image.height
        let isRGB = image.channels >= 3 && image.data.count >= srcPixelCount * 3

        if isRGB {
            // Planar scale each of R, G, B channels with vImage
            var rSrc = [Float](repeating: 0, count: srcPixelCount)
            var gSrc = [Float](repeating: 0, count: srcPixelCount)
            var bSrc = [Float](repeating: 0, count: srcPixelCount)
            
            for p in 0..<srcPixelCount {
                rSrc[p] = Float(image.data[p])
                gSrc[p] = Float(image.data[srcPixelCount + p])
                bSrc[p] = Float(image.data[srcPixelCount * 2 + p])
            }
            
            var rDst = [Float](repeating: 0, count: newW * newH)
            var gDst = [Float](repeating: 0, count: newW * newH)
            var bDst = [Float](repeating: 0, count: newW * newH)
            
            scalePlanar(src: &rSrc, srcW: image.width, srcH: image.height, dst: &rDst, dstW: newW, dstH: newH)
            scalePlanar(src: &gSrc, srcW: image.width, srcH: image.height, dst: &gDst, dstW: newW, dstH: newH)
            scalePlanar(src: &bSrc, srcW: image.width, srcH: image.height, dst: &bDst, dstW: newW, dstH: newH)
            
            // Scatter into letterboxed NHWC buffer
            for ty in 0..<newH {
                let outY = padY + ty
                for tx in 0..<newW {
                    let outX = padX + tx
                    let dstIdx = (outY * targetWidth + outX) * 3
                    let sIdx = ty * newW + tx
                    buffer[dstIdx + 0] = rDst[sIdx]
                    buffer[dstIdx + 1] = gDst[sIdx]
                    buffer[dstIdx + 2] = bDst[sIdx]
                }
            }
        } else {
            // Grayscale planar scaling
            var srcPlanar = [Float](repeating: 0, count: srcPixelCount)
            for p in 0..<srcPixelCount {
                srcPlanar[p] = p < image.data.count ? Float(image.data[p]) : 0.0
            }
            
            var dstPlanar = [Float](repeating: 0, count: newW * newH)
            scalePlanar(src: &srcPlanar, srcW: image.width, srcH: image.height, dst: &dstPlanar, dstW: newW, dstH: newH)
            
            // Scatter grayscale into 3-channel letterbox buffer
            for ty in 0..<newH {
                let outY = padY + ty
                for tx in 0..<newW {
                    let outX = padX + tx
                    let dstIdx = (outY * targetWidth + outX) * 3
                    let val = dstPlanar[ty * newW + tx]
                    buffer[dstIdx + 0] = val
                    buffer[dstIdx + 1] = val
                    buffer[dstIdx + 2] = val
                }
            }
        }

        return MLXArray(buffer, [1, targetHeight, targetWidth, 3])
    }
    
    private func scalePlanar(src: inout [Float], srcW: Int, srcH: Int, dst: inout [Float], dstW: Int, dstH: Int) {
        src.withUnsafeMutableBufferPointer { srcPtr in
            dst.withUnsafeMutableBufferPointer { dstPtr in
                var srcBuffer = vImage_Buffer(
                    data: srcPtr.baseAddress,
                    height: vImagePixelCount(srcH),
                    width: vImagePixelCount(srcW),
                    rowBytes: srcW * MemoryLayout<Float>.stride
                )
                var dstBuffer = vImage_Buffer(
                    data: dstPtr.baseAddress,
                    height: vImagePixelCount(dstH),
                    width: vImagePixelCount(dstW),
                    rowBytes: dstW * MemoryLayout<Float>.stride
                )
                _ = vImageScale_PlanarF(&srcBuffer, &dstBuffer, nil, vImage_Flags(kvImageHighQualityResampling))
            }
        }
    }
}

