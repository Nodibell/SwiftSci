import Foundation
#if canImport(CoreGraphics)
import CoreGraphics
#endif
#if canImport(ImageIO)
import ImageIO
#endif

extension ImageDataset {
#if canImport(CoreGraphics)
    /// Converts this in-memory `ImageDataset` into a CoreGraphics `CGImage`.
    ///
    /// Values are scaled automatically depending on whether they are in range `[0, 1]` or `[0, 255]`.
    /// - Returns: A rendered `CGImage`, or `nil` if dimensions are invalid or memory allocation fails.
    public func toCGImage() -> CGImage? {
        guard width > 0, height > 0, !data.isEmpty else { return nil }
        let pixelCount = width * height
        let c = max(1, channels)

        let maxVal = data.max() ?? 1.0
        let scale: Double = maxVal <= 1.0 ? 255.0 : 1.0

        if c == 1 {
            var rawBytes = [UInt8](repeating: 0, count: pixelCount)
            for i in 0..<pixelCount {
                let v = i < data.count ? data[i] * scale : 0
                rawBytes[i] = UInt8(clamping: Int(round(v)))
            }
            let colorSpace = CGColorSpaceCreateDeviceGray()
            guard let context = CGContext(
                data: &rawBytes,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: 0).rawValue
            ) else { return nil }
            return context.makeImage()
        } else {
            var rawBytes = [UInt8](repeating: 255, count: pixelCount * 4)
            for p in 0..<pixelCount {
                let rIdx = p
                let gIdx = c > 1 ? pixelCount + p : p
                let bIdx = c > 2 ? 2 * pixelCount + p : p
                let aIdx = c > 3 ? 3 * pixelCount + p : -1

                let r = rIdx < data.count ? data[rIdx] * scale : 0
                let g = gIdx < data.count ? data[gIdx] * scale : 0
                let b = bIdx < data.count ? data[bIdx] * scale : 0
                let a = (aIdx >= 0 && aIdx < data.count) ? data[aIdx] * scale : 255.0

                rawBytes[p * 4 + 0] = UInt8(clamping: Int(round(r)))
                rawBytes[p * 4 + 1] = UInt8(clamping: Int(round(g)))
                rawBytes[p * 4 + 2] = UInt8(clamping: Int(round(b)))
                rawBytes[p * 4 + 3] = UInt8(clamping: Int(round(a)))
            }

            let colorSpace = CGColorSpaceCreateDeviceRGB()
            guard let context = CGContext(
                data: &rawBytes,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return nil }
            return context.makeImage()
        }
    }
#endif

#if canImport(CoreGraphics) && canImport(ImageIO)
    /// Serializes the image dataset to encoded PNG `Data`.
    /// - Returns: Encoded PNG byte data, or `nil` on failure.
    public func toPNGData() -> Data? {
        guard let cgImage = toCGImage() else { return nil }
        let mutableData = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(mutableData as CFMutableData, "public.png" as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(dest, cgImage, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return mutableData as Data
    }

    /// Loads and decodes an image file from a local filesystem URL.
    ///
    /// - Parameter url: Local file URL pointing to an image (.png, .jpg, .heic, etc.).
    /// - Throws: `VisionError.invalidInput` if the image cannot be read or decoded.
    /// - Returns: An `ImageDataset` with normalized RGB channels in `[0, 1]`.
    public static func load(from url: URL) throws -> ImageDataset {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw VisionError.invalidInput("Could not decode image at URL: \(url.path)")
        }
        return try from(cgImage: cgImage)
    }

    /// Loads and decodes an image from raw memory Data.
    ///
    /// - Parameter data: Encoded image byte data.
    /// - Throws: `VisionError.invalidInput` if the image cannot be decoded.
    /// - Returns: An `ImageDataset` with normalized RGB channels in `[0, 1]`.
    public static func load(from data: Data) throws -> ImageDataset {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw VisionError.invalidInput("Could not decode image from byte buffer")
        }
        return try from(cgImage: cgImage)
    }

    /// Converts a CoreGraphics `CGImage` into an `ImageDataset`.
    ///
    /// - Parameter cgImage: Source CGImage reference.
    /// - Throws: `VisionError.invalidInput` if dimensions are invalid or memory allocation fails.
    /// - Returns: An `ImageDataset` with normalized RGB channels in `[0, 1]`.
    public static func from(cgImage: CGImage) throws -> ImageDataset {
        let w = cgImage.width
        let h = cgImage.height
        let pixelCount = w * h
        guard pixelCount > 0 else {
            throw VisionError.invalidInput("Image dimensions must be non-zero")
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var rawBytes = [UInt8](repeating: 0, count: pixelCount * 4)
        guard let context = CGContext(
            data: &rawBytes,
            width: w,
            height: h,
            bitsPerComponent: 8,
            bytesPerRow: w * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw VisionError.invalidInput("Failed to allocate bitmap rendering context")
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))

        var doubleData = [Double](repeating: 0.0, count: pixelCount * 3)
        for p in 0..<pixelCount {
            doubleData[p] = Double(rawBytes[p * 4 + 0]) / 255.0
            doubleData[pixelCount + p] = Double(rawBytes[p * 4 + 1]) / 255.0
            doubleData[2 * pixelCount + p] = Double(rawBytes[p * 4 + 2]) / 255.0
        }

        return ImageDataset(width: w, height: h, channels: 3, data: doubleData)
    }
#endif
}
