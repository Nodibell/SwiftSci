import Foundation

/// Metadata parsed from an Ultralytics YOLO dataset configuration file (`dataset.yaml` or `data.yaml`).
public struct YOLODatasetMetadata: Sendable, Equatable, Codable {
    /// Ordered list of class label names indexed by integer class ID (`0..<classNames.count`).
    public let classNames: [String]
    /// Optional training split path declared in the YAML configuration.
    public let trainPath: String?
    /// Optional validation split path declared in the YAML configuration (`val` or `valid`).
    public let valPath: String?
    /// Optional test split path declared in the YAML configuration.
    public let testPath: String?
    /// Optional declared class count (`nc`) from the YAML configuration.
    public let nc: Int?

    /// Creates a new YOLO dataset metadata record.
    ///
    /// - Parameters:
    ///   - classNames: Ordered class names indexed from `0`.
    ///   - trainPath: Optional training split path.
    ///   - valPath: Optional validation split path.
    ///   - testPath: Optional test split path.
    ///   - nc: Optional declared number of classes (`nc`).
    public init(
        classNames: [String],
        trainPath: String? = nil,
        valPath: String? = nil,
        testPath: String? = nil,
        nc: Int? = nil
    ) {
        self.classNames = classNames
        self.trainPath = trainPath
        self.valPath = valPath
        self.testPath = testPath
        self.nc = nc
    }

    /// Searches `directoryURL` and up to two parent directories for `dataset.yaml` or `data.yaml` and parses it.
    ///
    /// Supports both root-level layouts (`dataset/dataset.yaml`) and split layouts (`dataset/images/train`).
    ///
    /// - Parameter directoryURL: Directory URL to search from (such as an images split directory or dataset root).
    /// - Returns: Parsed `YOLODatasetMetadata`, or `nil` if no readable dataset YAML file is found.
    public static func parseYAML(at directoryURL: URL) -> YOLODatasetMetadata? {
        let baseDir = directoryURL.standardizedFileURL
        let parentDir = baseDir.deletingLastPathComponent()
        let grandParentDir = parentDir.deletingLastPathComponent()

        var searchDirs: [URL] = []
        var seenPaths: Set<String> = []
        for candidateDir in [baseDir, parentDir, grandParentDir] {
            if seenPaths.insert(candidateDir.path).inserted {
                searchDirs.append(candidateDir)
            }
        }

        let fileManager = FileManager.default
        let yamlFilenames = ["dataset.yaml", "data.yaml"]

        for dir in searchDirs {
            for filename in yamlFilenames {
                let yamlURL = dir.appendingPathComponent(filename)
                if fileManager.fileExists(atPath: yamlURL.path),
                   let content = try? String(contentsOf: yamlURL, encoding: .utf8),
                   let metadata = parseYAMLContent(content) {
                    return metadata
                }
            }
        }

        return nil
    }

    /// Parses the raw text content of an Ultralytics YOLO YAML configuration file.
    ///
    /// Supports `names` in dictionary (`0: car`), block list (`- car`), and inline list (`['car', 'truck']`)
    /// formats, strips inline `#` comments and outer quotes, extracts `train`, `val`/`valid`, `test`, and `nc`,
    /// and sorts dictionary indices numerically (`0, 1, 2, ...`).
    ///
    /// - Parameter content: Raw UTF-8 text content of a `dataset.yaml` or `data.yaml` file.
    /// - Returns: Parsed `YOLODatasetMetadata`, or `nil` if no recognized YOLO metadata fields are present.
    public static func parseYAMLContent(_ content: String) -> YOLODatasetMetadata? {
        let rawLines = content.components(separatedBy: .newlines)

        var trainPath: String?
        var valPath: String?
        var testPath: String?
        var nc: Int?
        var listNames: [String] = []
        var dictNames: [(index: Int, name: String)] = []
        var foundRecognizedField = false

        var inNamesBlock = false
        var namesBlockIndent = 0
        var collectingBracketList = false
        var bracketBuffer = ""

        for rawLine in rawLines {
            let uncommented = stripYAMLComment(rawLine)
            let trimmed = uncommented.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }

            if collectingBracketList {
                bracketBuffer += " " + trimmed
                if let closeIdx = findClosingBracketIndex(in: bracketBuffer, openChar: "[", closeChar: "]") {
                    let inner = String(bracketBuffer[bracketBuffer.index(after: bracketBuffer.startIndex)..<closeIdx])
                    listNames = parseInlineListItems(inner)
                    collectingBracketList = false
                    bracketBuffer = ""
                }
                continue
            }

            let indent = leadingSpaceCount(uncommented)

            if inNamesBlock {
                let isListItem = trimmed == "-" || trimmed.hasPrefix("- ") || trimmed.hasPrefix("-\t")
                let stillInBlock = indent > namesBlockIndent || (indent == namesBlockIndent && isListItem)
                if stillInBlock {
                    if isListItem {
                        let afterDash = String(trimmed.dropFirst()).trimmingCharacters(in: .whitespaces)
                        if !afterDash.isEmpty {
                            listNames.append(unquoteYAMLValue(afterDash))
                        }
                        continue
                    } else if let colonIdx = trimmed.firstIndex(of: ":") {
                        let rawKey = String(trimmed[..<colonIdx]).trimmingCharacters(in: .whitespaces)
                        let rawVal = String(trimmed[trimmed.index(after: colonIdx)...]).trimmingCharacters(in: .whitespaces)
                        let keyStr = unquoteYAMLValue(rawKey)
                        if let classIdx = Int(keyStr) {
                            dictNames.append((index: classIdx, name: unquoteYAMLValue(rawVal)))
                            continue
                        }
                    }
                }
                inNamesBlock = false
            }

            guard let colonIdx = trimmed.firstIndex(of: ":") else {
                continue
            }
            let key = unquoteYAMLValue(String(trimmed[..<colonIdx])).lowercased()
            let valuePart = String(trimmed[trimmed.index(after: colonIdx)...]).trimmingCharacters(in: .whitespaces)

            switch key {
            case "train":
                foundRecognizedField = true
                trainPath = normalizedOptionalPath(valuePart)
            case "val":
                foundRecognizedField = true
                valPath = normalizedOptionalPath(valuePart)
            case "valid":
                foundRecognizedField = true
                if valPath == nil {
                    valPath = normalizedOptionalPath(valuePart)
                }
            case "test":
                foundRecognizedField = true
                testPath = normalizedOptionalPath(valuePart)
            case "nc":
                foundRecognizedField = true
                let cleanNum = unquoteYAMLValue(valuePart)
                nc = Int(cleanNum)
            case "names":
                foundRecognizedField = true
                listNames = []
                dictNames = []
                if valuePart.isEmpty {
                    inNamesBlock = true
                    namesBlockIndent = indent
                } else if valuePart.hasPrefix("[") {
                    if let closeIdx = findClosingBracketIndex(in: valuePart, openChar: "[", closeChar: "]") {
                        let inner = String(valuePart[valuePart.index(after: valuePart.startIndex)..<closeIdx])
                        listNames = parseInlineListItems(inner)
                    } else {
                        collectingBracketList = true
                        bracketBuffer = valuePart
                    }
                } else if valuePart.hasPrefix("{") {
                    if let closeIdx = findClosingBracketIndex(in: valuePart, openChar: "{", closeChar: "}") {
                        let inner = String(valuePart[valuePart.index(after: valuePart.startIndex)..<closeIdx])
                        dictNames = parseInlineDictItems(inner)
                    }
                }
            default:
                break
            }
        }

        guard foundRecognizedField else {
            return nil
        }

        let resolvedNames: [String]
        if !dictNames.isEmpty {
            let sortedEntries = dictNames.sorted { (lhs: (index: Int, name: String), rhs: (index: Int, name: String)) -> Bool in
                lhs.index < rhs.index
            }
            resolvedNames = sortedEntries.map { (entry: (index: Int, name: String)) -> String in
                entry.name
            }
        } else {
            resolvedNames = listNames
        }

        return YOLODatasetMetadata(
            classNames: resolvedNames,
            trainPath: trainPath,
            valPath: valPath,
            testPath: testPath,
            nc: nc
        )
    }

    // MARK: - Private YAML Helpers

    private static func leadingSpaceCount(_ line: String) -> Int {
        var count = 0
        for ch in line {
            if ch == " " {
                count += 1
            } else if ch == "\t" {
                count += 2
            } else {
                break
            }
        }
        return count
    }

    private static func stripYAMLComment(_ line: String) -> String {
        var inSingleQuote = false
        var inDoubleQuote = false
        var isEscaped = false
        var previousChar: Character?

        for idx in line.indices {
            let ch = line[idx]
            if isEscaped {
                isEscaped = false
                previousChar = ch
                continue
            }
            if ch == "\\" && inDoubleQuote {
                isEscaped = true
                previousChar = ch
                continue
            }
            if ch == "'" && !inDoubleQuote {
                inSingleQuote.toggle()
            } else if ch == "\"" && !inSingleQuote {
                inDoubleQuote.toggle()
            } else if ch == "#" && !inSingleQuote && !inDoubleQuote {
                if previousChar == nil || previousChar?.isWhitespace == true {
                    return String(line[..<idx])
                }
            }
            previousChar = ch
        }
        return line
    }

    private static func unquoteYAMLValue(_ value: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 2 else { return trimmed }

        if trimmed.hasPrefix("'") && trimmed.hasSuffix("'") {
            let inner = String(trimmed.dropFirst().dropLast())
            return inner.replacingOccurrences(of: "''", with: "'")
        }
        if trimmed.hasPrefix("\"") && trimmed.hasSuffix("\"") {
            let inner = String(trimmed.dropFirst().dropLast())
            return inner
                .replacingOccurrences(of: "\\\"", with: "\"")
                .replacingOccurrences(of: "\\\\", with: "\\")
        }
        return trimmed
    }

    private static func normalizedOptionalPath(_ rawValue: String) -> String? {
        let cleaned = unquoteYAMLValue(rawValue)
        if cleaned.isEmpty || cleaned == "~" || cleaned.lowercased() == "null" {
            return nil
        }
        return cleaned
    }

    private static func findClosingBracketIndex(in text: String, openChar: Character, closeChar: Character) -> String.Index? {
        var inSingleQuote = false
        var inDoubleQuote = false
        var isEscaped = false
        var depth = 0

        for idx in text.indices {
            let ch = text[idx]
            if isEscaped {
                isEscaped = false
                continue
            }
            if ch == "\\" && inDoubleQuote {
                isEscaped = true
                continue
            }
            if ch == "'" && !inDoubleQuote {
                inSingleQuote.toggle()
            } else if ch == "\"" && !inSingleQuote {
                inDoubleQuote.toggle()
            } else if !inSingleQuote && !inDoubleQuote {
                if ch == openChar {
                    depth += 1
                } else if ch == closeChar {
                    depth -= 1
                    if depth == 0 {
                        return idx
                    }
                }
            }
        }
        return nil
    }

    private static func splitTopLevelCommaItems(_ text: String) -> [String] {
        var items: [String] = []
        var current = ""
        var inSingleQuote = false
        var inDoubleQuote = false
        var isEscaped = false

        for ch in text {
            if isEscaped {
                current.append(ch)
                isEscaped = false
                continue
            }
            if ch == "\\" && inDoubleQuote {
                current.append(ch)
                isEscaped = true
                continue
            }
            if ch == "'" && !inDoubleQuote {
                inSingleQuote.toggle()
                current.append(ch)
            } else if ch == "\"" && !inSingleQuote {
                inDoubleQuote.toggle()
                current.append(ch)
            } else if ch == "," && !inSingleQuote && !inDoubleQuote {
                let item = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if !item.isEmpty {
                    items.append(item)
                }
                current = ""
            } else {
                current.append(ch)
            }
        }

        let trailing = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trailing.isEmpty {
            items.append(trailing)
        }
        return items
    }

    private static func parseInlineListItems(_ inner: String) -> [String] {
        let rawItems = splitTopLevelCommaItems(inner)
        var parsed: [String] = []
        parsed.reserveCapacity(rawItems.count)
        for item in rawItems {
            let unquoted = unquoteYAMLValue(item)
            if !unquoted.isEmpty {
                parsed.append(unquoted)
            }
        }
        return parsed
    }

    private static func parseInlineDictItems(_ inner: String) -> [(index: Int, name: String)] {
        let rawEntries = splitTopLevelCommaItems(inner)
        var entries: [(index: Int, name: String)] = []
        for entry in rawEntries {
            guard let colonIdx = entry.firstIndex(of: ":") else { continue }
            let rawKey = String(entry[..<colonIdx])
            let rawVal = String(entry[entry.index(after: colonIdx)...])
            let keyStr = unquoteYAMLValue(rawKey)
            if let idx = Int(keyStr) {
                entries.append((index: idx, name: unquoteYAMLValue(rawVal)))
            }
        }
        return entries
    }
}

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

            let xMin = min(imageWidth, max(0.0, xCenter - (w / 2.0)))
            let yMin = min(imageHeight, max(0.0, yCenter - (h / 2.0)))
            let xMax = min(imageWidth, max(0.0, xCenter + (w / 2.0)))
            let yMax = min(imageHeight, max(0.0, yCenter + (h / 2.0)))

            guard xMax > xMin, yMax > yMin else { continue }

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
    /// 3. Split subfolders: `dataset/images/train/image_01.jpg` and `dataset/labels/train/image_01.txt`
    ///
    /// When `classLabels` is empty, automatically searches `imagesDirectory` and up to two parent
    /// directories for `dataset.yaml` or `data.yaml` via ``YOLODatasetMetadata/parseYAML(at:)`` before
    /// falling back to `"class_<id>"`.
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

        let resolvedClassLabels: [String]
        if classLabels.isEmpty {
            resolvedClassLabels = YOLODatasetMetadata.parseYAML(at: imagesDirectory)?.classNames ?? []
        } else {
            resolvedClassLabels = classLabels
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

        imageURLs.sort { (lhs: URL, rhs: URL) -> Bool in
            if lhs.lastPathComponent != rhs.lastPathComponent {
                return lhs.lastPathComponent < rhs.lastPathComponent
            }
            return lhs.path < rhs.path
        }

        let standardizedImagesDir = imagesDirectory.standardizedFileURL
        let imagesRootPrefix = standardizedImagesDir.path.hasSuffix("/")
            ? standardizedImagesDir.path
            : standardizedImagesDir.path + "/"

        var autoLabelDirs: [URL] = []
        if labelsDirectory == nil {
            if standardizedImagesDir.path.hasSuffix("/images") || standardizedImagesDir.lastPathComponent == "images" {
                autoLabelDirs.append(standardizedImagesDir.deletingLastPathComponent().appendingPathComponent("labels"))
            } else if standardizedImagesDir.deletingLastPathComponent().lastPathComponent == "images" {
                let labelsRoot = standardizedImagesDir
                    .deletingLastPathComponent()
                    .deletingLastPathComponent()
                    .appendingPathComponent("labels")
                autoLabelDirs.append(labelsRoot.appendingPathComponent(standardizedImagesDir.lastPathComponent))
                autoLabelDirs.append(labelsRoot)
            }
        }

        for imgURL in imageURLs {
            let baseName = imgURL.deletingPathExtension().lastPathComponent
            let standardizedImgPath = imgURL.standardizedFileURL.path
            var relativeTxtSubpath: String?
            if standardizedImgPath.hasPrefix(imagesRootPrefix) {
                let relative = String(standardizedImgPath.dropFirst(imagesRootPrefix.count))
                relativeTxtSubpath = (relative as NSString).deletingPathExtension + ".txt"
            }

            // Candidate paths for annotation file
            var candidateLabelURLs: [URL] = []
            var seenCandidates: Set<String> = []
            func appendCandidate(_ url: URL) {
                let key = url.standardizedFileURL.path
                if seenCandidates.insert(key).inserted {
                    candidateLabelURLs.append(url)
                }
            }

            if let explicitLabelsDir = labelsDirectory {
                if let relTxt = relativeTxtSubpath {
                    appendCandidate(explicitLabelsDir.appendingPathComponent(relTxt))
                }
                appendCandidate(explicitLabelsDir.appendingPathComponent("\(baseName).txt"))
            }

            // Same directory
            appendCandidate(imgURL.deletingLastPathComponent().appendingPathComponent("\(baseName).txt"))

            if labelsDirectory == nil {
                for autoDir in autoLabelDirs {
                    if let relTxt = relativeTxtSubpath {
                        appendCandidate(autoDir.appendingPathComponent(relTxt))
                    }
                    appendCandidate(autoDir.appendingPathComponent("\(baseName).txt"))
                }

                // Sibling labels directory (replace last /images/ with /labels/)
                if let lastRange = standardizedImgPath.range(of: "/images/", options: .backwards) {
                    var replaced = standardizedImgPath
                    replaced.replaceSubrange(lastRange, with: "/labels/")
                    let txtPath = (replaced as NSString).deletingPathExtension + ".txt"
                    appendCandidate(URL(fileURLWithPath: txtPath))
                } else if standardizedImgPath.hasSuffix("/images") {
                    let replaced = String(standardizedImgPath.dropLast("/images".count)) + "/labels"
                    appendCandidate(URL(fileURLWithPath: replaced).appendingPathComponent("\(baseName).txt"))
                }

                let imgPath = imgURL.path
                if let lastRange = imgPath.range(of: "/images/", options: .backwards) {
                    var replaced = imgPath
                    replaced.replaceSubrange(lastRange, with: "/labels/")
                    let txtPath = (replaced as NSString).deletingPathExtension + ".txt"
                    appendCandidate(URL(fileURLWithPath: txtPath))
                }
            }

            var groundTruthBoxes: [BoundingBox] = []
            for labelURL in candidateLabelURLs {
                if fileManager.fileExists(atPath: labelURL.path) {
                    if let boxes = try? parse(fileURL: labelURL, classLabels: resolvedClassLabels) {
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
