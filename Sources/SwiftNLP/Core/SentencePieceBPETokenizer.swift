import Foundation

/// Native tokenizer for SentencePiece-style BPE exports used by Gemma 2, Gemma 3, Gemma 4, and Mistral v0.3.
///
/// Reads `tokenizer.json`; no SentencePiece runtime, Python, or external tokenizer is used.
/// Only the explicitly supported normalization, byte fallback, and BOS templates are accepted.
public struct SentencePieceBPETokenizer: Tokenizer, Sendable {
    private struct MergePair: Hashable, Sendable {
        let left: Int
        let right: Int
    }

    private final class AddedTokenTrie: @unchecked Sendable {
        var terminalID: Int?
        var children: [UInt8: AddedTokenTrie] = [:]

        func insert(_ bytes: [UInt8], id: Int) {
            var node = self
            for byte in bytes {
                if let next = node.children[byte] {
                    node = next
                } else {
                    let next = AddedTokenTrie()
                    node.children[byte] = next
                    node = next
                }
            }
            node.terminalID = id
        }

        func longestMatch(in utf8: String.UTF8View, from start: String.UTF8View.Index) -> (id: Int, end: String.UTF8View.Index)? {
            var node = self
            var idx = start
            var match: (Int, String.UTF8View.Index)?
            while idx < utf8.endIndex, let next = node.children[utf8[idx]] {
                node = next
                idx = utf8.index(after: idx)
                if let terminalID = node.terminalID {
                    match = (terminalID, idx)
                }
            }
            return match
        }
    }

    private static let metaspaceUTF8: [UInt8] = [0xE2, 0x96, 0x81]

    private let vocabBytes: [[UInt8]: Int]
    private let decoderBytes: [Int: [UInt8]]
    private let mergeRanks: [MergePair: (rank: Int, mergedID: Int)]
    private let byteFallbackIDs: [Int]
    private let addedTrie: AddedTokenTrie
    private let postProcessorBOS: Int?
    private let prependSpace: Bool

    /// Loads a validated SentencePiece BPE tokenizer export (Mistral v0.3, Gemma 2, Gemma 3, or Gemma 4).
    /// - Parameter tokenizerJSON: Contents of the checkpoint's `tokenizer.json`.
    /// - Throws: `BPETokenizerConfigurationError` for unsupported tokenizer stages.
    public init(tokenizerJSON: Data) throws {
        func fail(_ message: String) -> BPETokenizerConfigurationError { .unsupported(message) }
        let root = try Self.parseUTF8JSON(tokenizerJSON)
        guard let model = root.field("model"),
              model.field("type")?.utf8String == "BPE",
              model.field("byte_fallback")?.boolValue == true,
              model.field("ignore_merges")?.boolValue != true,
              model.field("dropout")?.isNull ?? true,
              model.field("continuing_subword_prefix")?.isNull ?? true,
              model.field("end_of_word_suffix")?.isNull ?? true,
              let vocabEntries = model.field("vocab")?.objectEntries,
              let merges = model.field("merges")?.arrayValues,
              let pre = root.field("pre_tokenizer"),
              let post = root.field("post_processor"),
              let decode = root.field("decoder"),
              root.field("truncation")?.isNull ?? true,
              root.field("padding")?.isNull ?? true else { throw fail("SentencePiece BPE configuration") }
        // Key by raw UTF-8 bytes directly from our RFC 8259 parser so neither Swift `String`
        // nor Foundation `JSONSerialization` collapses canonically equivalent or U+FEFF keys.
        var vocabBytes: [[UInt8]: Int] = [:]
        vocabBytes.reserveCapacity(vocabEntries.count)
        for (tokenBytes, rawValue) in vocabEntries {
            guard let id = rawValue.intValue,
                  vocabBytes.updateValue(id, forKey: tokenBytes) == nil else {
                throw fail("SentencePiece BPE configuration")
            }
        }
        func same(_ lhs: JSONValue?, _ rhs: JSONPattern) -> Bool {
            Self.matches(lhs, rhs)
        }
        let metaspace = pre.field("type")?.utf8String == "Metaspace"
        if metaspace {
            guard root.field("normalizer")?.isNull ?? true,
                  same(pre, ["type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": false])
            else { throw fail("Mistral metaspace stage") }
        } else {
            guard same(root.field("normalizer"), ["type": "Replace", "pattern": ["String": " "], "content": "▁"]),
                  same(pre, ["type": "Split", "pattern": ["String": " "], "behavior": "MergedWithPrevious", "invert": false])
            else { throw fail("Gemma normalization stage") }
        }
        var stages: [JSONPattern] = [["type": "Replace", "pattern": ["String": "▁"], "content": " "],
                                      ["type": "ByteFallback"], ["type": "Fuse"]]
        if metaspace { stages.append(["type": "Strip", "content": " ", "start": 1, "stop": 0]) }
        guard same(decode, ["type": "Sequence", "decoders": .array(stages)]) else { throw fail("SentencePiece decoder") }
        let bosToken = metaspace ? "<s>" : "<bos>"
        guard post.field("type")?.utf8String == "TemplateProcessing",
              let single = post.field("single")?.arrayValues else { throw fail("single-sequence BOS template") }
        let postProcessorBOS: Int?
        if single.count == 2,
           let bosID = vocabBytes[Array(bosToken.utf8)],
           same(single[0], ["SpecialToken": ["id": .text(bosToken), "type_id": 0]]),
           same(single[1], ["Sequence": ["id": "A", "type_id": 0]]) {
            postProcessorBOS = bosID
        } else if single.count == 1,
                  same(single[0], ["Sequence": ["id": "A", "type_id": 0]]) {
            postProcessorBOS = nil
        } else {
            throw fail("single-sequence BOS template")
        }
        var decoderBytes: [Int: [UInt8]] = [:]
        decoderBytes.reserveCapacity(vocabBytes.count)
        for (utf8, id) in vocabBytes {
            guard id >= 0, decoderBytes.updateValue(utf8, forKey: id) == nil else {
                throw fail("duplicate or negative vocabulary IDs")
            }
        }
        var mergeRanks: [MergePair: (rank: Int, mergedID: Int)] = [:]
        mergeRanks.reserveCapacity(merges.count)
        for (rank, merge) in merges.enumerated() {
            let leftBytes: [UInt8]
            let rightBytes: [UInt8]
            if let array = merge.arrayValues {
                guard array.count == 2,
                      let first = array[0].stringBytes,
                      let second = array[1].stringBytes else { throw fail("BPE merge vocabulary") }
                leftBytes = first
                rightBytes = second
            } else if let mergedBytes = merge.stringBytes {
                guard let space = mergedBytes.firstIndex(of: 0x20) else { throw fail("BPE merge vocabulary") }
                let after = mergedBytes.index(after: space)
                guard mergedBytes[after...].firstIndex(of: 0x20) == nil else { throw fail("BPE merge vocabulary") }
                leftBytes = Array(mergedBytes[..<space])
                rightBytes = Array(mergedBytes[after...])
            } else {
                throw fail("BPE merge encoding")
            }
            guard let leftID = vocabBytes[leftBytes],
                  let rightID = vocabBytes[rightBytes],
                  let mergedID = vocabBytes[leftBytes + rightBytes] else {
                throw fail("BPE merge vocabulary")
            }
            let pair = MergePair(left: leftID, right: rightID)
            if mergeRanks[pair] == nil {
                mergeRanks[pair] = (rank, mergedID)
            }
        }
        let trie = AddedTokenTrie()
        for entry in root.field("added_tokens")?.arrayValues ?? [] {
            guard let tokenBytes = entry.field("content")?.stringBytes, !tokenBytes.isEmpty,
                  let id = entry.field("id")?.intValue, id >= 0,
                  entry.field("single_word")?.boolValue != true,
                  entry.field("lstrip")?.boolValue != true,
                  entry.field("rstrip")?.boolValue != true,
                  entry.field("normalized")?.boolValue != true else { throw fail("added token behavior") }
            if let existing = vocabBytes[tokenBytes] {
                guard existing == id else { throw fail("added token behavior") }
            } else {
                guard decoderBytes.updateValue(tokenBytes, forKey: id) == nil else {
                    throw fail("duplicate or negative vocabulary IDs")
                }
                vocabBytes[tokenBytes] = id
            }
            trie.insert(tokenBytes, id: id)
        }
        var byteFallbackIDs = [Int](repeating: 0, count: 256)
        for byte in 0...255 {
            let hexBytes = Array(String(format: "<0x%02X>", byte).utf8)
            if let hexID = vocabBytes[hexBytes] {
                byteFallbackIDs[byte] = hexID
            } else if byte < 128, let asciiID = vocabBytes[[UInt8(byte)]] {
                // Gemma 2 exports omit `<0x09>` because literal `"\t"` already exists in `model.vocab`.
                byteFallbackIDs[byte] = asciiID
            } else {
                throw fail("complete byte fallback vocabulary")
            }
        }
        self.vocabBytes = vocabBytes
        self.decoderBytes = decoderBytes
        self.mergeRanks = mergeRanks
        self.byteFallbackIDs = byteFallbackIDs
        self.addedTrie = trie
        self.postProcessorBOS = postProcessorBOS
        self.prependSpace = metaspace
    }

    /// Tokenizes without adding the BOS template. Whitespace and Unicode scalars are preserved.
    public func tokenize(text: String) -> [String] {
        encodeWithoutBOS(text: text).compactMap { id in
            decoderBytes[id].map { String(decoding: $0, as: UTF8.self) }
        }
    }

    /// Encodes text with the checkpoint's BOS template.
    public func encode(text: String) -> [Int] { encode(text: text, addSpecialTokens: true) }

    /// Encodes text, optionally adding the single-sequence BOS token when configured.
    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let prefix = addSpecialTokens ? (postProcessorBOS.map { [$0] } ?? []) : []
        return prefix + encodeWithoutBOS(text: text)
    }

    /// Decodes token IDs, preserving special-token spellings and combining byte-fallback tokens.
    public func decode(tokens: [Int]) -> String {
        var allBytes = tokens.flatMap(bytes)
        if prependSpace, allBytes.first == 0x20 { allBytes.removeFirst() }
        return String(decoding: allBytes, as: UTF8.self)
    }

    /// Preserves spaces and incomplete UTF-8 bytes while streaming continuation tokens.
    public func makeStreamDecoder() -> TokenStreamDecoder { TokenStreamDecoder(decodeBytes: bytes) }

    private func encodeWithoutBOS(text: String) -> [Int] {
        var result: [Int] = []
        var plainBytes: [UInt8] = []
        let utf8 = text.utf8
        var cursor = utf8.startIndex
        var first = true
        func flush() {
            guard !plainBytes.isEmpty else { return }
            if prependSpace && first && !plainBytes.starts(with: Self.metaspaceUTF8) {
                plainBytes = Self.metaspaceUTF8 + plainBytes
            }
            result += bpeIDs(plainBytes)
            plainBytes.removeAll(keepingCapacity: true)
            first = false
        }
        while cursor < utf8.endIndex {
            if let match = addedTrie.longestMatch(in: utf8, from: cursor) {
                flush()
                result.append(match.id)
                cursor = match.end
                first = false
            } else {
                let next = text.unicodeScalars.index(after: cursor)
                if utf8.distance(from: cursor, to: next) == 1, utf8[cursor] == 0x20 {
                    plainBytes.append(contentsOf: Self.metaspaceUTF8)
                } else {
                    plainBytes.append(contentsOf: utf8[cursor..<next])
                }
                cursor = next
            }
        }
        flush()
        return result
    }

    private func bytes(_ id: Int) -> [UInt8] {
        guard let tokenBytes = decoderBytes[id] else { return [] }
        return Self.decodeTokenBytes(tokenBytes)
    }

    private func bpeIDs(_ utf8: [UInt8]) -> [Int] {
        var ids: [Int] = []
        var idx = utf8.startIndex
        while idx < utf8.endIndex {
            let b0 = utf8[idx]
            let scalarLen: Int
            if b0 < 0x80 { scalarLen = 1 }
            else if b0 < 0xE0 { scalarLen = 2 }
            else if b0 < 0xF0 { scalarLen = 3 }
            else { scalarLen = 4 }
            let end = min(idx + scalarLen, utf8.endIndex)
            let scalarBytes = Array(utf8[idx..<end])
            if let id = vocabBytes[scalarBytes] {
                ids.append(id)
            } else {
                for byte in scalarBytes {
                    ids.append(byteFallbackIDs[Int(byte)])
                }
            }
            idx = end
        }
        while ids.count > 1 {
            var bestIndex = -1
            var bestRank = Int.max
            var bestMergedID = 0
            for i in 0..<(ids.count - 1) {
                if let merge = mergeRanks[MergePair(left: ids[i], right: ids[i + 1])],
                   merge.rank < bestRank {
                    bestRank = merge.rank
                    bestIndex = i
                    bestMergedID = merge.mergedID
                }
            }
            guard bestIndex >= 0 else { break }
            ids[bestIndex] = bestMergedID
            ids.remove(at: bestIndex + 1)
        }
        return ids
    }

    private static func decodeTokenBytes(_ tokenBytes: [UInt8]) -> [UInt8] {
        if tokenBytes.count == 6,
           tokenBytes[0] == 0x3C, tokenBytes[1] == 0x30, tokenBytes[2] == 0x78, tokenBytes[5] == 0x3E,
           let high = hexNibble(tokenBytes[3]), let low = hexNibble(tokenBytes[4]) {
            return [(high << 4) | low]
        }
        var out: [UInt8] = []
        out.reserveCapacity(tokenBytes.count)
        var i = 0
        while i < tokenBytes.count {
            if i + 2 < tokenBytes.count,
               tokenBytes[i] == 0xE2, tokenBytes[i + 1] == 0x96, tokenBytes[i + 2] == 0x81 {
                out.append(0x20)
                i += 3
            } else {
                out.append(tokenBytes[i])
                i += 1
            }
        }
        return out
    }

    private static func hexNibble(_ byte: UInt8) -> UInt8? {
        switch byte {
        case 0x30...0x39: return byte - 0x30
        case 0x41...0x46: return byte - 0x41 + 10
        case 0x61...0x66: return byte - 0x61 + 10
        default: return nil
        }
    }

    private enum JSONValue {
        case object([([UInt8], JSONValue)])
        case array([JSONValue])
        case string([UInt8])
        case int(Int)
        case number
        case bool(Bool)
        case null

        var objectEntries: [([UInt8], JSONValue)]? {
            if case .object(let entries) = self { return entries }
            return nil
        }

        func field(_ key: String) -> JSONValue? {
            guard case .object(let entries) = self else { return nil }
            let target = Array(key.utf8)
            var found: JSONValue?
            for (entryKey, value) in entries where entryKey == target {
                found = value
            }
            return found
        }

        var stringBytes: [UInt8]? {
            if case .string(let bytes) = self { return bytes }
            return nil
        }

        var utf8String: String? {
            stringBytes.map { String(decoding: $0, as: UTF8.self) }
        }

        var intValue: Int? {
            if case .int(let value) = self { return value }
            return nil
        }

        var boolValue: Bool? {
            if case .bool(let value) = self { return value }
            return nil
        }

        var isNull: Bool {
            if case .null = self { return true }
            return false
        }

        var arrayValues: [JSONValue]? {
            if case .array(let values) = self { return values }
            return nil
        }
    }

    private enum JSONPattern: ExpressibleByStringLiteral, ExpressibleByIntegerLiteral,
                              ExpressibleByBooleanLiteral, ExpressibleByArrayLiteral,
                              ExpressibleByDictionaryLiteral {
        case string([UInt8])
        case int(Int)
        case bool(Bool)
        case array([JSONPattern])
        case object([([UInt8], JSONPattern)])

        init(stringLiteral value: String) { self = .string(Array(value.utf8)) }
        init(integerLiteral value: Int) { self = .int(value) }
        init(booleanLiteral value: Bool) { self = .bool(value) }
        init(arrayLiteral elements: JSONPattern...) { self = .array(elements) }
        init(dictionaryLiteral elements: (String, JSONPattern)...) {
            self = .object(elements.map { (Array($0.0.utf8), $0.1) })
        }

        static func text(_ value: String) -> JSONPattern { .string(Array(value.utf8)) }
    }

    private static func matches(_ value: JSONValue?, _ pattern: JSONPattern) -> Bool {
        switch pattern {
        case .string(let expected):
            return value?.stringBytes == expected
        case .int(let expected):
            return value?.intValue == expected
        case .bool(let expected):
            return value?.boolValue == expected
        case .array(let expected):
            guard let actual = value?.arrayValues, actual.count == expected.count else { return false }
            return zip(actual, expected).allSatisfy { matches($0, $1) }
        case .object(let expected):
            guard let entries = value?.objectEntries else { return false }
            var actual: [[UInt8]: JSONValue] = [:]
            actual.reserveCapacity(entries.count)
            for (key, item) in entries {
                actual[key] = item
            }
            guard actual.count == expected.count else { return false }
            return expected.allSatisfy { key, subPattern in
                guard let found = actual[key] else { return false }
                return matches(found, subPattern)
            }
        }
    }

    private static func parseUTF8JSON(_ data: Data) throws -> JSONValue {
        guard !data.isEmpty else { throw BPETokenizerConfigurationError.unsupported("SentencePiece BPE configuration") }
        return try data.withUnsafeBytes { rawBuffer in
            let bytes = rawBuffer.bindMemory(to: UInt8.self)
            var parser = UTF8JSONParser(bytes: bytes)
            return try parser.parseDocument()
        }
    }

    private struct UTF8JSONParser {
        let bytes: UnsafeBufferPointer<UInt8>
        var index: Int = 0

        private func badJSON() -> BPETokenizerConfigurationError {
            .unsupported("SentencePiece BPE configuration")
        }

        mutating func parseDocument() throws -> JSONValue {
            skipWhitespace()
            let root = try parseValue(depth: 0)
            skipWhitespace()
            guard index == bytes.count else { throw badJSON() }
            return root
        }

        private mutating func skipWhitespace() {
            while index < bytes.count {
                switch bytes[index] {
                case 0x20, 0x09, 0x0A, 0x0D:
                    index += 1
                default:
                    return
                }
            }
        }

        private mutating func parseValue(depth: Int) throws -> JSONValue {
            guard depth <= 64, index < bytes.count else { throw badJSON() }
            switch bytes[index] {
            case 0x7B:
                return try parseObject(depth: depth + 1)
            case 0x5B:
                return try parseArray(depth: depth + 1)
            case 0x22:
                return .string(try parseStringBytes())
            case 0x74:
                try expectASCII([0x74, 0x72, 0x75, 0x65])
                return .bool(true)
            case 0x66:
                try expectASCII([0x66, 0x61, 0x6C, 0x73, 0x65])
                return .bool(false)
            case 0x6E:
                try expectASCII([0x6E, 0x75, 0x6C, 0x6C])
                return .null
            case 0x2D, 0x30...0x39:
                return try parseNumber()
            default:
                throw badJSON()
            }
        }

        private mutating func expectASCII(_ expected: [UInt8]) throws {
            guard index + expected.count <= bytes.count else { throw badJSON() }
            for offset in 0..<expected.count where bytes[index + offset] != expected[offset] {
                throw badJSON()
            }
            index += expected.count
        }

        private mutating func parseObject(depth: Int) throws -> JSONValue {
            index += 1
            skipWhitespace()
            var entries: [([UInt8], JSONValue)] = []
            if index < bytes.count, bytes[index] == 0x7D {
                index += 1
                return .object(entries)
            }
            while true {
                guard index < bytes.count, bytes[index] == 0x22 else { throw badJSON() }
                let key = try parseStringBytes()
                skipWhitespace()
                guard index < bytes.count, bytes[index] == 0x3A else { throw badJSON() }
                index += 1
                skipWhitespace()
                let value = try parseValue(depth: depth)
                entries.append((key, value))
                skipWhitespace()
                guard index < bytes.count else { throw badJSON() }
                if bytes[index] == 0x7D {
                    index += 1
                    return .object(entries)
                }
                guard bytes[index] == 0x2C else { throw badJSON() }
                index += 1
                skipWhitespace()
            }
        }

        private mutating func parseArray(depth: Int) throws -> JSONValue {
            index += 1
            skipWhitespace()
            var items: [JSONValue] = []
            if index < bytes.count, bytes[index] == 0x5D {
                index += 1
                return .array(items)
            }
            while true {
                let item = try parseValue(depth: depth)
                items.append(item)
                skipWhitespace()
                guard index < bytes.count else { throw badJSON() }
                if bytes[index] == 0x5D {
                    index += 1
                    return .array(items)
                }
                guard bytes[index] == 0x2C else { throw badJSON() }
                index += 1
                skipWhitespace()
            }
        }

        private mutating func parseStringBytes() throws -> [UInt8] {
            guard index < bytes.count, bytes[index] == 0x22 else { throw badJSON() }
            index += 1
            var segmentStart = index
            var out: [UInt8]?
            var hasNonASCII = false
            while index < bytes.count {
                let b = bytes[index]
                if b == 0x22 {
                    let tail = bytes[segmentStart..<index]
                    index += 1
                    if hasNonASCII, !Self.isValidUTF8(tail) { throw badJSON() }
                    if var buffer = out {
                        buffer.append(contentsOf: tail)
                        return buffer
                    }
                    return Array(tail)
                }
                if b == 0x5C {
                    var buffer = out ?? []
                    let segment = bytes[segmentStart..<index]
                    if hasNonASCII, !Self.isValidUTF8(segment) { throw badJSON() }
                    buffer.append(contentsOf: segment)
                    hasNonASCII = false
                    index += 1
                    guard index < bytes.count else { throw badJSON() }
                    let esc = bytes[index]
                    index += 1
                    switch esc {
                    case 0x22: buffer.append(0x22)
                    case 0x5C: buffer.append(0x5C)
                    case 0x2F: buffer.append(0x2F)
                    case 0x62: buffer.append(0x08)
                    case 0x66: buffer.append(0x0C)
                    case 0x6E: buffer.append(0x0A)
                    case 0x72: buffer.append(0x0D)
                    case 0x74: buffer.append(0x09)
                    case 0x75:
                        let unit1 = try parseHex4()
                        if (0xD800...0xDBFF).contains(unit1) {
                            guard index + 2 <= bytes.count, bytes[index] == 0x5C, bytes[index + 1] == 0x75 else {
                                throw badJSON()
                            }
                            index += 2
                            let unit2 = try parseHex4()
                            guard (0xDC00...0xDFFF).contains(unit2),
                                  let scalar = UnicodeScalar(0x10000 + ((unit1 - 0xD800) << 10) + (unit2 - 0xDC00)) else {
                                throw badJSON()
                            }
                            buffer.append(contentsOf: scalar.utf8)
                        } else {
                            guard !(0xDC00...0xDFFF).contains(unit1),
                                  let scalar = UnicodeScalar(unit1) else {
                                throw badJSON()
                            }
                            buffer.append(contentsOf: scalar.utf8)
                        }
                    default:
                        throw badJSON()
                    }
                    out = buffer
                    segmentStart = index
                    continue
                }
                guard b >= 0x20 else { throw badJSON() }
                if b >= 0x80 { hasNonASCII = true }
                index += 1
            }
            throw badJSON()
        }

        private mutating func parseHex4() throws -> UInt32 {
            guard index + 4 <= bytes.count else { throw badJSON() }
            var value: UInt32 = 0
            for offset in 0..<4 {
                guard let nibble = SentencePieceBPETokenizer.hexNibble(bytes[index + offset]) else {
                    throw badJSON()
                }
                value = (value << 4) | UInt32(nibble)
            }
            index += 4
            return value
        }

        private mutating func parseNumber() throws -> JSONValue {
            var negative = false
            if bytes[index] == 0x2D {
                negative = true
                index += 1
            }
            guard index < bytes.count else { throw badJSON() }
            let digitStart = index
            if bytes[index] == 0x30 {
                index += 1
                if index < bytes.count, (0x30...0x39).contains(bytes[index]) { throw badJSON() }
            } else if (0x31...0x39).contains(bytes[index]) {
                while index < bytes.count, (0x30...0x39).contains(bytes[index]) {
                    index += 1
                }
            } else {
                throw badJSON()
            }
            let intEnd = index
            var isFloat = false
            if index < bytes.count, bytes[index] == 0x2E {
                isFloat = true
                index += 1
                guard index < bytes.count, (0x30...0x39).contains(bytes[index]) else { throw badJSON() }
                while index < bytes.count, (0x30...0x39).contains(bytes[index]) {
                    index += 1
                }
            }
            if index < bytes.count, bytes[index] == 0x65 || bytes[index] == 0x45 {
                isFloat = true
                index += 1
                if index < bytes.count, bytes[index] == 0x2B || bytes[index] == 0x2D {
                    index += 1
                }
                guard index < bytes.count, (0x30...0x39).contains(bytes[index]) else { throw badJSON() }
                while index < bytes.count, (0x30...0x39).contains(bytes[index]) {
                    index += 1
                }
            }
            if isFloat { return .number }
            var value = 0
            for pos in digitStart..<intEnd {
                let digit = Int(bytes[pos] - 0x30)
                let mul = value.multipliedReportingOverflow(by: 10)
                let step = negative ? mul.partialValue.subtractingReportingOverflow(digit)
                                    : mul.partialValue.addingReportingOverflow(digit)
                if mul.overflow || step.overflow { return .number }
                value = step.partialValue
            }
            return .int(value)
        }

        private static func isValidUTF8(_ slice: UnsafeBufferPointer<UInt8>.SubSequence) -> Bool {
            var codec = UTF8()
            var iter = slice.makeIterator()
            while true {
                switch codec.decode(&iter) {
                case .scalarValue: continue
                case .emptyInput: return true
                case .error: return false
                }
            }
        }
    }
}
