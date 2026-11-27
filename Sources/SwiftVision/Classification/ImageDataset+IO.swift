import Foundation
#if canImport(CoreGraphics)
import CoreGraphics
#endif
#if canImport(ImageIO)
import ImageIO
#endif

/// Labeled multi-image classification dataset loaded from a directory of class subdirectories.
public struct ImageFolderDataset: Sendable {
    /// Decoded and resized image pixel buffers in loading order.
    public let images: [ImageDataset]
    /// Integer class label indices corresponding one-to-one with `images` (`0..<classes.count`).
    public let labels: [Int]
    /// Ordered class names mapped by integer label index.
    public let classes: [String]
    /// Source file URLs corresponding one-to-one with `images`.
    public let imageURLs: [URL]

    /// Creates a new image folder dataset container.
    ///
    /// - Parameters:
    ///   - images: Decoded image datasets.
    ///   - labels: Integer class label indices (`0..<classes.count`).
    ///   - classes: Ordered class label names.
    ///   - imageURLs: Source image file URLs.
    public init(images: [ImageDataset], labels: [Int], classes: [String], imageURLs: [URL]) {
        self.images = images
        self.labels = labels
        self.classes = classes
        self.imageURLs = imageURLs
    }
}

/// Split-aware image classification dataset container preserving globally aligned class indices across splits.
public struct SplitImageDataset: Sendable {
    /// Training split dataset (or the full dataset when loaded from a flat `<class>/<image>` folder).
    public let train: ImageFolderDataset
    /// Optional validation split dataset (`val`, `valid`, or `validation`).
    public let validation: ImageFolderDataset?
    /// Optional test split dataset (`test` or `testing`).
    public let test: ImageFolderDataset?
    /// Globally sorted union of class names shared across all splits.
    public let classes: [String]

    /// Creates a new split-aware image dataset container.
    ///
    /// - Parameters:
    ///   - train: Training split dataset.
    ///   - validation: Optional validation split dataset.
    ///   - test: Optional test split dataset.
    ///   - classes: Globally consistent ordered list of class names.
    public init(
        train: ImageFolderDataset,
        validation: ImageFolderDataset? = nil,
        test: ImageFolderDataset? = nil,
        classes: [String]
    ) {
        self.train = train
        self.validation = validation
        self.test = test
        self.classes = classes
    }
}

extension ImageDataset {
#if canImport(CoreGraphics)
    /// Crops a `CGImage` to a normalized `[0, 1]` `BoundingBox` with optional proportional padding.
    ///
    /// Coordinates are interpreted in normalized top-left image space `[0, 1] x [0, 1]`, expanded
    /// proportionally by `paddingFraction` on each side, and clamped to image pixel bounds.
    ///
    /// - Parameters:
    ///   - image: Source `CGImage` to crop.
    ///   - box: Normalized `[0, 1]` bounding box (`xMin`, `yMin`, `xMax`, `yMax`).
    ///   - paddingFraction: Proportional padding fraction relative to box dimensions (default: `0.0`).
    /// - Returns: Cropped `CGImage`, or `nil` if the box is degenerate or the clamped crop is smaller than 1 pixel.
    public static func crop(
        image: CGImage,
        to box: BoundingBox,
        paddingFraction: Double = 0.0
    ) -> CGImage? {
        guard box.xMin.isFinite, box.yMin.isFinite, box.xMax.isFinite, box.yMax.isFinite,
              box.xMax > box.xMin, box.yMax > box.yMin else {
            return nil
        }
        let normalizedRect = CGRect(
            x: box.xMin,
            y: box.yMin,
            width: box.xMax - box.xMin,
            height: box.yMax - box.yMin
        )
        return crop(image: image, to: normalizedRect, paddingFraction: paddingFraction)
    }

    /// Crops a `CGImage` to a normalized `[0, 1]` `CGRect` with optional proportional padding.
    ///
    /// - Parameters:
    ///   - image: Source `CGImage` to crop.
    ///   - normalizedRect: Normalized `[0, 1]` rectangle in top-left image coordinates.
    ///   - paddingFraction: Proportional padding fraction relative to rectangle width and height (default: `0.0`).
    /// - Returns: Cropped `CGImage`, or `nil` if the rectangle is invalid or the clamped crop is smaller than 1 pixel in width or height.
    public static func crop(
        image: CGImage,
        to normalizedRect: CGRect,
        paddingFraction: Double = 0.0
    ) -> CGImage? {
        guard image.width > 0, image.height > 0 else { return nil }
        guard !normalizedRect.isNull, !normalizedRect.isInfinite,
              normalizedRect.origin.x.isFinite, normalizedRect.origin.y.isFinite,
              normalizedRect.size.width.isFinite, normalizedRect.size.height.isFinite,
              normalizedRect.width > 0, normalizedRect.height > 0 else {
            return nil
        }

        let imgW = Double(image.width)
        let imgH = Double(image.height)
        let padFactor = paddingFraction.isFinite ? max(0.0, paddingFraction) : 0.0

        let rawXMin = Double(normalizedRect.minX) * imgW
        let rawYMin = Double(normalizedRect.minY) * imgH
        let rawXMax = Double(normalizedRect.maxX) * imgW
        let rawYMax = Double(normalizedRect.maxY) * imgH

        let boxW = rawXMax - rawXMin
        let boxH = rawYMax - rawYMin
        let padX = boxW * padFactor
        let padY = boxH * padFactor

        let clampedXMin = max(0.0, min(imgW, rawXMin - padX))
        let clampedYMin = max(0.0, min(imgH, rawYMin - padY))
        let clampedXMax = max(0.0, min(imgW, rawXMax + padX))
        let clampedYMax = max(0.0, min(imgH, rawYMax + padY))

        guard (clampedXMax - clampedXMin) >= 1.0, (clampedYMax - clampedYMin) >= 1.0 else {
            return nil
        }

        let cropX = Int(floor(clampedXMin))
        let cropY = Int(floor(clampedYMin))
        let cropMaxX = min(image.width, Int(ceil(clampedXMax)))
        let cropMaxY = min(image.height, Int(ceil(clampedYMax)))
        let cropW = cropMaxX - cropX
        let cropH = cropMaxY - cropY

        guard cropW >= 1, cropH >= 1 else {
            return nil
        }

        return image.cropping(to: CGRect(x: cropX, y: cropY, width: cropW, height: cropH))
    }

    /// Crops all valid normalized bounding boxes from a `CGImage`.
    ///
    /// - Parameters:
    ///   - image: Source `CGImage`.
    ///   - boxes: Array of normalized `[0, 1]` `BoundingBox` annotations.
    ///   - paddingFraction: Proportional padding fraction applied to each box (default: `0.0`).
    /// - Returns: Array of `(box, image)` tuples for every bounding box yielding a valid crop.
    public static func cropAll(
        image: CGImage,
        boxes: [BoundingBox],
        paddingFraction: Double = 0.0
    ) -> [(box: BoundingBox, image: CGImage)] {
        var croppedResults: [(box: BoundingBox, image: CGImage)] = []
        croppedResults.reserveCapacity(boxes.count)
        for box in boxes {
            if let croppedImage = crop(image: image, to: box, paddingFraction: paddingFraction) {
                croppedResults.append((box: box, image: croppedImage))
            }
        }
        return croppedResults
    }

    /// Samples a deterministic, non-overlapping background crop from a `CGImage` avoiding annotated object boxes.
    ///
    /// Useful for constructing negative/background training examples from detection datasets when
    /// annotation `.txt` files are empty or leave background regions unoccupied.
    ///
    /// - Parameters:
    ///   - image: Source `CGImage`.
    ///   - boxes: Normalized `[0, 1]` object bounding boxes to avoid overlapping.
    ///   - size: Optional pixel dimensions of the background crop (defaults to one-quarter of the smaller image edge).
    ///   - maxAttempts: Maximum random placement attempts before returning `nil` (default: `16`).
    ///   - seed: Deterministic PRNG seed for reproducible sampling (default: `42`).
    /// - Returns: A non-overlapping background `CGImage` crop, or `nil` if no valid region is found.
    public static func sampleBackgroundCrop(
        from image: CGImage,
        avoiding boxes: [BoundingBox],
        size: CGSize? = nil,
        maxAttempts: Int = 16,
        seed: UInt64 = 42
    ) -> CGImage? {
        guard image.width >= 1, image.height >= 1, maxAttempts >= 1 else {
            return nil
        }

        let cropW: Int
        let cropH: Int
        if let requestedSize = size {
            guard requestedSize.width.isFinite, requestedSize.height.isFinite,
                  requestedSize.width >= 1.0, requestedSize.height >= 1.0 else {
                return nil
            }
            cropW = Int(round(requestedSize.width))
            cropH = Int(round(requestedSize.height))
        } else {
            let defaultEdge = max(1, min(image.width, image.height) / 4)
            cropW = defaultEdge
            cropH = defaultEdge
        }

        guard cropW >= 1, cropH >= 1, cropW <= image.width, cropH <= image.height else {
            return nil
        }

        let maxPixelX = image.width - cropW
        let maxPixelY = image.height - cropH
        let imgW = Double(image.width)
        let imgH = Double(image.height)

        var state: UInt64 = seed == 0 ? 0x9E3779B97F4A7C15 : seed
        func nextUnitDouble() -> Double {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            z = z ^ (z >> 31)
            return Double(z >> 11) / Double(1 << 53)
        }

        for _ in 0..<maxAttempts {
            let px = maxPixelX > 0 ? min(maxPixelX, Int(floor(nextUnitDouble() * Double(maxPixelX + 1)))) : 0
            let py = maxPixelY > 0 ? min(maxPixelY, Int(floor(nextUnitDouble() * Double(maxPixelY + 1)))) : 0

            let candidateBox = BoundingBox(
                xMin: Double(px) / imgW,
                yMin: Double(py) / imgH,
                xMax: Double(px + cropW) / imgW,
                yMax: Double(py + cropH) / imgH,
                confidence: 1.0,
                classLabel: "background"
            )

            var overlaps = false
            for box in boxes {
                if candidateBox.iou(with: box) > 0.0 {
                    overlaps = true
                    break
                }
            }

            if !overlaps {
                return image.cropping(to: CGRect(x: px, y: py, width: cropW, height: cropH))
            }
        }

        return nil
    }
#endif
}

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

    /// Loads a labeled multi-image dataset from a directory organized as `<class>/<image>`.
    ///
    /// Subdirectories are treated as class labels and images within each class directory are
    /// decoded, resized to `targetSize`, and ordered deterministically by class and file path.
    ///
    /// - Parameters:
    ///   - folderURL: Root folder containing class subdirectories.
    ///   - targetSize: Target pixel dimensions for all decoded images (default: `224x224`).
    ///   - classes: Optional pre-specified class order for globally aligned label IDs across splits.
    ///     When `nil`, class names are discovered from immediate subdirectories and sorted lexicographically.
    /// - Throws: `VisionError.invalidInput` if the folder is missing, `targetSize` is invalid, or an image fails to decode.
    /// - Returns: A populated ``ImageFolderDataset``.
    public static func loadFolder(
        at folderURL: URL,
        targetSize: CGSize = CGSize(width: 224, height: 224),
        classes: [String]? = nil
    ) throws -> ImageFolderDataset {
        let resolvedRoot = folderURL.resolvingSymlinksInPath()
        let fileManager = FileManager.default
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: resolvedRoot.path, isDirectory: &isDir), isDir.boolValue else {
            throw VisionError.invalidInput("Directory not found: \(folderURL.path)")
        }

        guard targetSize.width.isFinite, targetSize.height.isFinite,
              targetSize.width >= 1.0, targetSize.height >= 1.0 else {
            throw VisionError.invalidInput("targetSize dimensions must be positive: \(targetSize)")
        }

        let targetW = Int(round(targetSize.width))
        let targetH = Int(round(targetSize.height))
        guard targetW >= 1, targetH >= 1 else {
            throw VisionError.invalidInput("targetSize pixel dimensions must be at least 1x1")
        }

        let subdirs = try immediateSubdirectories(in: resolvedRoot)
        var subdirByName: [String: URL] = [:]
        for dirURL in subdirs {
            subdirByName[dirURL.lastPathComponent] = dirURL
        }

        let resolvedClasses: [String]
        if let explicitClasses = classes {
            resolvedClasses = explicitClasses
        } else {
            resolvedClasses = subdirByName.keys.sorted()
        }

        var images: [ImageDataset] = []
        var labels: [Int] = []
        var imageURLs: [URL] = []

        for (labelIndex, className) in resolvedClasses.enumerated() {
            guard let classDirURL = subdirByName[className] else { continue }
            let urls = try collectSortedImageURLs(in: classDirURL)
            for imgURL in urls {
                guard let source = CGImageSourceCreateWithURL(imgURL as CFURL, nil),
                      let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                    throw VisionError.invalidInput("Could not decode image at URL: \(imgURL.path)")
                }
                let resized = try resizedDataset(from: cgImage, targetWidth: targetW, targetHeight: targetH)
                images.append(resized)
                labels.append(labelIndex)
                imageURLs.append(imgURL)
            }
        }

        return ImageFolderDataset(
            images: images,
            labels: labels,
            classes: resolvedClasses,
            imageURLs: imageURLs
        )
    }

    /// Loads a split-aware image classification dataset from either a split directory layout or a flat class layout.
    ///
    /// Recognizes case-insensitive split subdirectories (`train`/`training`, `val`/`valid`/`validation`,
    /// `test`/`testing`), computes the sorted union of class names across all present splits so integer
    /// label IDs `0..<classes.count` remain globally consistent, and falls back to loading a flat
    /// `<class>/<image>` directory into `train` with `validation: nil` and `test: nil`.
    ///
    /// - Parameters:
    ///   - folderURL: Root dataset directory URL.
    ///   - targetSize: Target pixel dimensions for all decoded images (default: `224x224`).
    /// - Throws: `VisionError.invalidInput` if the root directory does not exist, a split layout is missing
    ///   a `train`/`training` split, `targetSize` is invalid, or any image fails to decode.
    /// - Returns: A ``SplitImageDataset`` with aligned class indices across splits.
    public static func loadSplitFolder(
        at folderURL: URL,
        targetSize: CGSize = CGSize(width: 224, height: 224)
    ) throws -> SplitImageDataset {
        let resolvedRoot = folderURL.resolvingSymlinksInPath()
        let fileManager = FileManager.default
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: resolvedRoot.path, isDirectory: &isDir), isDir.boolValue else {
            throw VisionError.invalidInput("Directory not found: \(folderURL.path)")
        }

        guard targetSize.width.isFinite, targetSize.height.isFinite,
              targetSize.width >= 1.0, targetSize.height >= 1.0 else {
            throw VisionError.invalidInput("targetSize dimensions must be positive: \(targetSize)")
        }

        let topSubdirs = try immediateSubdirectories(in: resolvedRoot)
        let trainAliases = ["train", "training"]
        let valAliases = ["val", "valid", "validation"]
        let testAliases = ["test", "testing"]

        let trainDir = resolveSplitDirectory(from: topSubdirs, aliases: trainAliases)
        let valDir = resolveSplitDirectory(from: topSubdirs, aliases: valAliases)
        let testDir = resolveSplitDirectory(from: topSubdirs, aliases: testAliases)

        let hasAnySplitDir = (trainDir != nil || valDir != nil || testDir != nil)
        guard hasAnySplitDir else {
            let flatTrain = try loadFolder(at: resolvedRoot, targetSize: targetSize, classes: nil)
            return SplitImageDataset(
                train: flatTrain,
                validation: nil,
                test: nil,
                classes: flatTrain.classes
            )
        }

        guard let resolvedTrainDir = trainDir else {
            throw VisionError.invalidInput("Split dataset folder is missing a 'train' or 'training' subdirectory: \(folderURL.path)")
        }

        var classUnion: Set<String> = []
        for splitDir in [resolvedTrainDir, valDir, testDir].compactMap({ (dir: URL?) -> URL? in dir }) {
            let classDirs = try immediateSubdirectories(in: splitDir)
            for classDir in classDirs {
                classUnion.insert(classDir.lastPathComponent)
            }
        }

        let globalClasses = classUnion.sorted()
        let trainDataset = try loadFolder(at: resolvedTrainDir, targetSize: targetSize, classes: globalClasses)
        let validationDataset: ImageFolderDataset?
        if let resolvedValDir = valDir {
            validationDataset = try loadFolder(at: resolvedValDir, targetSize: targetSize, classes: globalClasses)
        } else {
            validationDataset = nil
        }

        let testDataset: ImageFolderDataset?
        if let resolvedTestDir = testDir {
            testDataset = try loadFolder(at: resolvedTestDir, targetSize: targetSize, classes: globalClasses)
        } else {
            testDataset = nil
        }

        return SplitImageDataset(
            train: trainDataset,
            validation: validationDataset,
            test: testDataset,
            classes: globalClasses
        )
    }

    // MARK: - Private Folder & Resize Helpers

    private static func resizedDataset(from cgImage: CGImage, targetWidth: Int, targetHeight: Int) throws -> ImageDataset {
        let pixelCount = targetWidth * targetHeight
        guard pixelCount > 0 else {
            throw VisionError.invalidInput("Image dimensions must be non-zero")
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var rawBytes = [UInt8](repeating: 0, count: pixelCount * 4)
        guard let context = CGContext(
            data: &rawBytes,
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: targetWidth * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw VisionError.invalidInput("Failed to allocate bitmap rendering context")
        }

        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))

        var doubleData = [Double](repeating: 0.0, count: pixelCount * 3)
        for p in 0..<pixelCount {
            doubleData[p] = Double(rawBytes[p * 4 + 0]) / 255.0
            doubleData[pixelCount + p] = Double(rawBytes[p * 4 + 1]) / 255.0
            doubleData[2 * pixelCount + p] = Double(rawBytes[p * 4 + 2]) / 255.0
        }

        return ImageDataset(width: targetWidth, height: targetHeight, channels: 3, data: doubleData)
    }

    private static func immediateSubdirectories(in directoryURL: URL) throws -> [URL] {
        let contents = try FileManager.default.contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )
        var dirs: [URL] = []
        for item in contents {
            let values = try item.resourceValues(forKeys: [.isDirectoryKey])
            if values.isDirectory == true {
                dirs.append(item)
            }
        }
        dirs.sort { (lhs: URL, rhs: URL) -> Bool in
            lhs.lastPathComponent < rhs.lastPathComponent
        }
        return dirs
    }

    private static func resolveSplitDirectory(from subdirectories: [URL], aliases: [String]) -> URL? {
        for alias in aliases {
            if let match = subdirectories.first(where: { (dir: URL) -> Bool in
                dir.lastPathComponent.lowercased() == alias
            }) {
                return match
            }
        }
        return nil
    }

    private static func collectSortedImageURLs(in directoryURL: URL) throws -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: directoryURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw VisionError.invalidInput("Failed to enumerate directory at: \(directoryURL.path)")
        }

        var urls: [URL] = []
        while let fileURL = enumerator.nextObject() as? URL {
            let ext = fileURL.pathExtension.lowercased()
            if VisionImageClassifier.supportedExtensions.contains(ext) {
                urls.append(fileURL)
            }
        }

        urls.sort { (lhs: URL, rhs: URL) -> Bool in
            lhs.standardizedFileURL.path < rhs.standardizedFileURL.path
        }
        return urls
    }
#endif
}
