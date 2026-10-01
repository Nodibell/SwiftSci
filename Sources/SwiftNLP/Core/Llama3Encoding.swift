import Foundation

/// A checkpoint tokenizer uses a configuration SwiftSci cannot interpret safely.
public enum BPETokenizerConfigurationError: Error, Sendable, Equatable {
    /// The named tokenizer component is invalid or unsupported.
    case unsupported(String)
}

/// The supported Llama 3 byte-level pipeline, distinct from legacy word-ending BPE.
struct Llama3Encoding: Sendable {
    static let pattern = #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    let splitter: NSRegularExpression
    let specialMatcher: NSRegularExpression
    let beginningOfText: Int
    let specialTokens: Set<String>

    init(specialTokens: [String: Int], beginningOfText: Int) throws {
        splitter = try NSRegularExpression(pattern: Self.pattern)
        let alternatives = specialTokens.keys.sorted {
            $0.count == $1.count ? $0 < $1 : $0.count > $1.count
        }.map(NSRegularExpression.escapedPattern(for:)).joined(separator: "|")
        specialMatcher = try NSRegularExpression(pattern: alternatives)
        self.beginningOfText = beginningOfText
        self.specialTokens = Set(specialTokens.keys)
    }

    func pieces(_ text: String) -> [String] {
        let input = text as NSString
        var result: [String] = []
        var start = 0
        func appendOrdinary(_ range: NSRange) {
            // Run on the substring so lookarounds stop at special-token boundaries.
            let ordinary = input.substring(with: range) as NSString
            for match in splitter.matches(in: ordinary as String, range: NSRange(location: 0, length: ordinary.length)) {
                result.append(ordinary.substring(with: match.range))
            }
        }
        for match in specialMatcher.matches(in: text, range: NSRange(location: 0, length: input.length)) {
            appendOrdinary(NSRange(location: start, length: match.range.location - start))
            result.append(input.substring(with: match.range))
            start = NSMaxRange(match.range)
        }
        appendOrdinary(NSRange(location: start, length: input.length - start))
        return result
    }
}

extension BPETokenizer {
    /// Loads the Llama 3 byte-level BPE pipeline from a Hugging Face `tokenizer.json`.
    ///
    /// Supports the Llama 3 regex, byte alphabet, special tokens, vocabulary-first
    /// BPE and single-sequence BOS template. Other processing rules throw rather
    /// than silently changing the checkpoint's token IDs. This does not interpret
    /// `tokenizer_config.json` chat templates, padding, truncation or paired input.
    ///
    /// `encode(text:)` encodes exactly the supplied text, including literal special
    /// tokens. Use `encode(text:addSpecialTokens:)` to prepend the configured BOS.
    /// Decoding preserves whitespace and includes special-token text.
    /// - Parameter data: Contents of the checkpoint's `tokenizer.json` file.
    /// - Throws: A decoding error or `BPETokenizerConfigurationError`.
    public init(llama3TokenizerJSON data: Data) throws {
        func require(_ condition: Bool, _ component: String) throws {
            if !condition { throw BPETokenizerConfigurationError.unsupported(component) }
        }
        func absent(_ value: Any?) -> Bool { value == nil || value is NSNull }
        func matches(_ value: Any?, _ expected: [String: Any]) -> Bool {
            guard let actual = value as? [String: Any] else { return false }
            return NSDictionary(dictionary: actual).isEqual(to: expected)
        }
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any],
              var vocab = model["vocab"] as? [String: Int],
              let rawMerges = model["merges"] as? [Any],
              let added = root["added_tokens"] as? [[String: Any]] else {
            throw BPETokenizerConfigurationError.unsupported("BPE document")
        }
        try require(root["version"] as? String == "1.0", "version")
        for name in ["normalizer", "truncation", "padding"] {
            try require(absent(root[name]), name)
        }
        try require(model["type"] as? String == "BPE" && model["ignore_merges"] as? Bool == true,
                    "vocabulary-first BPE")
        for name in ["dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix"] {
            try require(absent(model[name]), name)
        }
        for name in ["fuse_unk", "byte_fallback"] {
            try require(model[name] as? Bool == false, name)
        }
        try require(matches(root["pre_tokenizer"], ["type": "Sequence", "pretokenizers": [
            ["type": "Split", "pattern": ["Regex": Llama3Encoding.pattern], "behavior": "Isolated", "invert": false],
            ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false]
        ]]), "Llama 3 pre-tokenizer")
        try require(matches(root["decoder"], ["type": "ByteLevel", "add_prefix_space": true,
                                             "trim_offsets": true, "use_regex": true]), "byte decoder")
        try require(vocab.values.allSatisfy { $0 >= 0 } && Set(vocab.values).count == vocab.count,
                    "unique nonnegative vocabulary IDs")
        try require(Self.byteEncoder.values.allSatisfy { vocab[String($0)] != nil }, "complete byte alphabet")
        let alphabet = Set(Self.byteEncoder.values.map { String($0).unicodeScalars.first! })
        try require(vocab.keys.allSatisfy { !$0.isEmpty && $0.unicodeScalars.allSatisfy(alphabet.contains) },
                    "byte-encoded vocabulary")
        var special: [String: Int] = [:]
        var ids = Set(vocab.values)
        for token in added {
            guard let text = token["content"] as? String, !text.isEmpty,
                  let id = token["id"] as? Int, id >= 0 else {
                throw BPETokenizerConfigurationError.unsupported("added token")
            }
            try require(token["special"] as? Bool == true, "non-special added token")
            for flag in ["single_word", "lstrip", "rstrip", "normalized"] {
                try require(token[flag] as? Bool == false, "added token \(flag)")
            }
            try require(vocab[text] == nil && special[text] == nil && ids.insert(id).inserted,
                        "unique special tokens and IDs")
            special[text] = id
        }
        guard let bos = special["<|begin_of_text|>"] else {
            throw BPETokenizerConfigurationError.unsupported("beginning-of-text token")
        }
        guard let post = root["post_processor"] as? [String: Any], post["type"] as? String == "Sequence",
              let processors = post["processors"] as? [[String: Any]], processors.count == 2 else {
            throw BPETokenizerConfigurationError.unsupported("post-processor sequence")
        }
        try require(matches(processors[0], ["type": "ByteLevel", "add_prefix_space": true,
                                           "trim_offsets": false, "use_regex": true]), "post-processor byte offsets")
        var template = processors[1]
        // Paired inputs are outside this API; only the single-sequence template is executed.
        template.removeValue(forKey: "pair")
        try require(matches(template, ["type": "TemplateProcessing",
            "single": [["SpecialToken": ["id": "<|begin_of_text|>", "type_id": 0]], ["Sequence": ["id": "A", "type_id": 0]]],
            "special_tokens": ["<|begin_of_text|>": ["id": "<|begin_of_text|>", "ids": [bos], "tokens": ["<|begin_of_text|>"]]]
        ]), "single-sequence BOS template")
        let merges = try rawMerges.map { value -> String in
            let parts: [String]
            if let string = value as? String {
                parts = string.components(separatedBy: " ")
            } else if let pair = value as? [String] {
                parts = pair
            } else {
                throw BPETokenizerConfigurationError.unsupported("merge pair")
            }
            try require(parts.count == 2 && parts.allSatisfy { !$0.isEmpty && vocab[$0] != nil }, "merge operands")
            try require(vocab[parts.joined()] != nil, "merge result")
            return parts.joined(separator: " ")
        }
        for (text, id) in special { vocab[text] = id }
        self.init(vocab: vocab, merges: merges)
        llama3 = try Llama3Encoding(specialTokens: special, beginningOfText: bos)
    }

    /// Encodes text with optional checkpoint post-processing.
    ///
    /// For the Llama 3 loader, `true` prepends BOS even when the text already
    /// contains a literal BOS token, matching the checkpoint's single-input template.
    /// Legacy word-ending tokenizers have no post-processor and are unchanged.
    /// - Parameters:
    ///   - text: Text or an already rendered chat prompt.
    ///   - addSpecialTokens: Whether to apply the checkpoint's BOS template.
    /// - Returns: Token IDs in input order.
    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let tokens = encode(text: text)
        if addSpecialTokens, let llama3 { return [llama3.beginningOfText] + tokens }
        return tokens
    }
}
