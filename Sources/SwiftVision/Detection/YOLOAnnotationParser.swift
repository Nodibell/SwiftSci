import Foundation

/// Parser for standard YOLO-format bounding box annotation text and directory structures.
public enum YOLOAnnotationParser {

    /// Parses a YOLO-format annotation string into an array of `BoundingBox` instances.
    ///
    /// Expected format per line:
    /// `<class_id> <x_center> <y_center> <width> <height> [optional_confidence]`
    /// Coordinates are normalized in [0.0, 1.0].
    ///
    /// - Parameters:
    ///   - text: Raw multiline text content of a YOLO annotation file.
    ///   - imageWidth: Optional image width to un-normalize coordinates to pixel scale (default: 1.0).
    ///   - imageHeight: Optional image height to un-normalize coordinates to pixel scale (default: 1.0).
    ///   - classLabels: Optional array mapping integer class indices to human-readable labels.
    /// - Returns: Array of parsed `BoundingBox` records.
    public static func parse(
        text: String,
        imageWidth: Double = 1.0,
        imageHeight: Double = 1.0,
        classLabels: [String] = []
    ) -> [BoundingBox] {
        var boxes: [BoundingBox] = []
        let lines = text.components(separatedBy: .newlines)

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, !trimmed.hasPrefix("#") else { continue }

            let parts = trimmed.split(separator: " ").map(String.init)
            guard parts.count >= 5 else { continue }

            guard let classId = Int(parts[0]),
                  let xCenterNorm = Double(parts[1]),
                  let yCenterNorm = Double(parts[2]),
                  let widthNorm = Double(parts[3]),
                  let heightNorm = Double(parts[4]) else {
                continue
            }

            let confidence = parts.count >= 6 ? (Double(parts[5]) ?? 1.0) : 1.0

            let xCenter = xCenterNorm * imageWidth
            let yCenter = yCenterNorm * imageHeight
            let w = widthNorm * imageWidth
            let h = heightNorm * imageHeight

            let xMin = max(0.0, xCenter - (w / 2.0))
            let yMin = max(0.0, yCenter - (h / 2.0))
            let xMax = xMin + w
            let yMax = yMin + h

            let label = (classId >= 0 && classId < classLabels.count) ? classLabels[classId] : "class_\(classId)"

            boxes.append(BoundingBox(
                xMin: xMin,
                yMin: yMin,
                xMax: xMax,
                yMax: yMax,
                confidence: confidence,
                classLabel: label
            ))
        }

        return boxes
    }

    /// Reads and parses a YOLO annotation file from a filesystem URL.
    ///
    /// - Parameters:
    ///   - fileURL: Local file URL to a `.txt` annotation file.
    ///   - imageWidth: Image width scaling factor (default: 1.0).
    ///   - imageHeight: Image height scaling factor (default: 1.0).
    ///   - classLabels: Class name dictionary or list.
    /// - Throws: `VisionError.invalidInput` if the file cannot be read.
    /// - Returns: Array of parsed `BoundingBox` instances.
    public static func parse(
        fileURL: URL,
        imageWidth: Double = 1.0,
        imageHeight: Double = 1.0,
        classLabels: [String] = []
    ) throws -> [BoundingBox] {
        guard let text = try? String(contentsOf: fileURL, encoding: .utf8) else {
            throw VisionError.invalidInput("Failed to read annotation file at: \(fileURL.path)")
        }
        return parse(text: text, imageWidth: imageWidth, imageHeight: imageHeight, classLabels: classLabels)
    }

    /// Discovers and pairs image files with their corresponding YOLO annotation text files across a directory.
    ///
    /// Supports standard layouts:
    /// 1. Co-located: `dataset/image_01.jpg` and `dataset/image_01.txt`
    /// 2. Separate folders: `dataset/images/image_01.jpg` and `dataset/labels/image_01.txt`
    ///
    /// - Parameters:
    ///   - imagesDirectory: Root folder containing image files.
    ///   - labelsDirectory: Optional custom directory containing annotation `.txt` files.
    ///   - classLabels: Class names mapping.
    /// - Throws: `VisionError.invalidInput` if directory cannot be read.
    /// - Returns: List of paired tuples containing the image URL and its ground-truth bounding boxes.
    public static func parseDataset(
        imagesDirectory: URL,
        labelsDirectory: URL? = nil,
        classLabels: [String] = []
    ) throws -> [(imageURL: URL, groundTruth: [BoundingBox])] {
        let fileManager = FileManager.default
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: imagesDirectory.path, isDirectory: &isDir), isDir.boolValue else {
            throw VisionError.invalidInput("Images directory not found: \(imagesDirectory.path)")
        }

        let supportedImageExts: Set<String> = ["jpg", "jpeg", "png", "bmp", "heic", "webp"]
        var results: [(imageURL: URL, groundTruth: [BoundingBox])] = []

        guard let enumerator = fileManager.enumerator(
            at: imagesDirectory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var imageURLs: [URL] = []
        while let fileURL = enumerator.nextObject() as? URL {
            let ext = fileURL.pathExtension.lowercased()
            if supportedImageExts.contains(ext) {
                imageURLs.append(fileURL)
            }
        }

        imageURLs.sort { $0.lastPathComponent < $1.lastPathComponent }

        for imgURL in imageURLs {
            let baseName = imgURL.deletingPathExtension().lastPathComponent

            // Candidate paths for annotation file
            var candidateLabelURLs: [URL] = []

            if let explicitLabelsDir = labelsDirectory {
                candidateLabelURLs.append(explicitLabelsDir.appendingPathComponent("\(baseName).txt"))
            }

            // Same directory
            candidateLabelURLs.append(imgURL.deletingLastPathComponent().appendingPathComponent("\(baseName).txt"))

            // Sibling labels directory (replace /images/ with /labels/)
            let imgPath = imgURL.path
            if imgPath.contains("/images/") {
                let labelPath = imgPath.replacingOccurrences(of: "/images/", with: "/labels/")
                let candidateURL = URL(fileURLWithPath: (labelPath as NSString).deletingPathExtension + ".txt")
                candidateLabelURLs.append(candidateURL)
            }

            var groundTruthBoxes: [BoundingBox] = []
            for labelURL in candidateLabelURLs {
                if fileManager.fileExists(atPath: labelURL.path) {
                    if let boxes = try? parse(fileURL: labelURL, classLabels: classLabels) {
                        groundTruthBoxes = boxes
                        break
                    }
                }
            }

            results.append((imageURL: imgURL, groundTruth: groundTruthBoxes))
        }

        return results
    }
}
