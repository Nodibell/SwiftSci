import Foundation
import Testing
@testable import SwiftNLP

@Suite("Llama byte-level tokenizer")
struct LlamaTokenizerTests {
    private func fixture() throws -> Data {
        // GPT-2 byte alphabet: ASCII plus remapped whitespace and control bytes.
        let visible = Array(33...126) + Array(161...172) + Array(174...255)
        var vocab: [String: Int] = [:]
        var next = 256
        for byte in 0...255 {
            let scalar = visible.contains(byte) ? byte : next
            if !visible.contains(byte) { next += 1 }
            vocab[String(UnicodeScalar(scalar)!)] = byte
        }
        vocab["hello"] = 256
        vocab["12"] = 257
        vocab["123"] = 258
        vocab["1234"] = 259
        vocab["a!"] = 260
        vocab["'s"] = 261
        let special: [[String: Any]] = [
            ["id": 1000, "content": "<|begin_of_text|>", "special": true,
             "single_word": false, "lstrip": false, "rstrip": false, "normalized": false],
            ["id": 1001, "content": "<|eot_id|>", "special": true,
             "single_word": false, "lstrip": false, "rstrip": false, "normalized": false],
        ]
        let root: [String: Any] = [
            "version": "1.0", "normalizer": NSNull(), "truncation": NSNull(), "padding": NSNull(),
            "added_tokens": special,
            "model": ["type": "BPE", "vocab": vocab, "merges": [],
                      "ignore_merges": true, "dropout": NSNull(), "unk_token": NSNull(),
                      "continuing_subword_prefix": NSNull(), "end_of_word_suffix": NSNull(),
                      "fuse_unk": false, "byte_fallback": false],
            "pre_tokenizer": ["type": "Sequence", "pretokenizers": [
                ["type": "Split", "pattern": ["Regex": #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#], "behavior": "Isolated", "invert": false],
                ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false]
            ]],
            "decoder": ["type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true],
            "post_processor": ["type": "Sequence", "processors": [
                ["type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true],
                ["type": "TemplateProcessing",
                 "single": [["SpecialToken": ["id": "<|begin_of_text|>", "type_id": 0]], ["Sequence": ["id": "A", "type_id": 0]]],
                 "pair": [["SpecialToken": ["id": "<|begin_of_text|>", "type_id": 0]], ["Sequence": ["id": "A", "type_id": 0]], ["SpecialToken": ["id": "<|begin_of_text|>", "type_id": 1]], ["Sequence": ["id": "B", "type_id": 1]]],
                 "special_tokens": ["<|begin_of_text|>": ["id": "<|begin_of_text|>", "ids": [1000], "tokens": ["<|begin_of_text|>"]]]]
            ]]
        ]
        return try JSONSerialization.data(withJSONObject: root)
    }

    @Test("Streaming preserves split Unicode scalars, whitespace, and literal replacement characters")
    func streamingUnicode() throws {
        let tokenizer = try BPETokenizer(llama3TokenizerJSON: fixture())
        for text in [" café e\u{301} 世界 👩‍🔬 🇺🇦\r\n", "\u{0}\u{7} � ", "<|begin_of_text|>hello<|eot_id|>"] {
            var stream = tokenizer.makeStreamDecoder()
            var actual = ""
            for token in tokenizer.encode(text: text) { actual += stream.append(token) ?? "" }
            actual += stream.finish()
            #expect(Array(actual.utf8) == Array(text.utf8))
        }
    }

    private func changed(_ mutate: (inout [String: Any]) -> Void) throws -> Data {
        var root = try #require(JSONSerialization.jsonObject(with: fixture()) as? [String: Any])
        mutate(&root)
        return try JSONSerialization.data(withJSONObject: root)
    }

    @Test("Byte mode keeps whitespace and uses whole-vocabulary tokens without merges")
    func preservesBytes() throws {
        let tokenizer = try BPETokenizer(llama3TokenizerJSON: fixture())
        #expect(tokenizer.encode(text: "hello") == [256])
        #expect(tokenizer.encode(text: " hello\t\n ") == [32, 104, 101, 108, 108, 111, 9, 10, 32])
        #expect(tokenizer.decode(tokens: [32, 256, 9, 10, 32]) == " hello\t\n ")
        #expect(tokenizer.encode(text: "") == [])
    }

    @Test("Llama pre-tokenization prevents merges across punctuation and digit boundaries")
    func boundaries() throws {
        let tokenizer = try BPETokenizer(llama3TokenizerJSON: fixture())
        #expect(tokenizer.encode(text: "1234") == [258, 52])
        #expect(tokenizer.encode(text: "a!") == [97, 33])
        #expect(tokenizer.encode(text: "a's") == [97, 261])
    }

    @Test("Special tokens are atomic and BOS insertion is explicit")
    func specialTokens() throws {
        let tokenizer = try BPETokenizer(llama3TokenizerJSON: fixture())
        #expect(tokenizer.encode(text: "<|begin_of_text|>hello<|eot_id|>") == [1000, 256, 1001])
        #expect(tokenizer.encode(text: "hello", addSpecialTokens: true) == [1000, 256])
        #expect(tokenizer.encode(text: "", addSpecialTokens: true) == [1000])
        #expect(tokenizer.decode(tokens: [1000, 32, 256, 1001]) == "<|begin_of_text|> hello<|eot_id|>")
    }

    @Test("Unicode and literal legacy word endings round-trip without normalization")
    func roundTrip() throws {
        let tokenizer = try BPETokenizer(llama3TokenizerJSON: fixture())
        for text in ["\t café e\u{301}\r\n", "привіт 世界 🚀 🇺🇦 👩‍🔬", "\u{0}\u{7}", " </w> ", "\r\n\r\n", "\u{00a0}\u{2028}"] {
            #expect(Array(tokenizer.decode(tokens: tokenizer.encode(text: text)).utf8) == Array(text.utf8))
        }
    }

    @Test("Unsupported checkpoint processing fails at initialization")
    func unsupportedConfiguration() throws {
        let data = try changed { $0["normalizer"] = ["type": "NFC"] }
        #expect(throws: (any Error).self) { try BPETokenizer(llama3TokenizerJSON: data) }
    }

    @Test("Incomplete byte vocabularies and duplicate token IDs fail at initialization")
    func invalidVocabulary() throws {
        for duplicate in [false, true] {
            let data = try changed { root in
                var model = root["model"] as! [String: Any]
                var vocab = model["vocab"] as! [String: Int]
                if duplicate { vocab["extra"] = 0 } else { vocab.removeValue(forKey: "!") }
                model["vocab"] = vocab
                root["model"] = model
            }
            #expect(throws: (any Error).self) { try BPETokenizer(llama3TokenizerJSON: data) }
        }
    }

    @Test("Merge rank wins and both tokenizer JSON merge representations work")
    func mergeRanks() throws {
        for arrays in [false, true] {
            let data = try changed { root in
                var model = root["model"] as! [String: Any]
                var vocab = model["vocab"] as! [String: Int]
                vocab["ab"] = 300
                vocab["bc"] = 301
                model["vocab"] = vocab
                model["merges"] = arrays ? [["b", "c"], ["a", "b"]] as Any : ["b c", "a b"] as Any
                root["model"] = model
            }
            let tokenizer = try BPETokenizer(llama3TokenizerJSON: data)
            #expect(tokenizer.encode(text: "abc") == [97, 301])
        }
    }

    @Test("Invalid merges and unsupported tokenizer stages are rejected")
    func invalidStages() throws {
        let mutations: [(inout [String: Any]) -> Void] = [
            { $0["pre_tokenizer"] = ["type": "Whitespace"] },
            { $0["decoder"] = ["type": "WordPiece"] },
            { $0["post_processor"] = NSNull() },
            { $0["padding"] = ["length": 32] },
            { $0["truncation"] = ["max_length": 32] },
            { $0["version"] = "2.0" },
            { root in
                var model = root["model"] as! [String: Any]
                model["merges"] = ["a"]
                root["model"] = model
            },
            { root in
                var model = root["model"] as! [String: Any]
                model["merges"] = ["a b"] // The joined token is absent from the vocabulary.
                root["model"] = model
            },
            { root in
                var model = root["model"] as! [String: Any]
                model["ignore_merges"] = false
                root["model"] = model
            },
            { root in
                var tokens = root["added_tokens"] as! [[String: Any]]
                tokens[0]["lstrip"] = true
                root["added_tokens"] = tokens
            },
            { root in
                var tokens = root["added_tokens"] as! [[String: Any]]
                tokens[0]["id"] = 1001
                root["added_tokens"] = tokens
            },
        ]
        for mutation in mutations {
            #expect(throws: (any Error).self) {
                try BPETokenizer(llama3TokenizerJSON: changed(mutation))
            }
        }
    }

    @Test("Malformed documents identify the unsupported tokenizer component")
    func malformedDocuments() throws {
        let mutations: [(String, (inout [String: Any]) -> Void)] = [
            ("BPE document", { $0.removeValue(forKey: "model") }),
            ("Llama 3 pre-tokenizer", { $0["pre_tokenizer"] = NSNull() }),
            ("added token", { root in
                var tokens = root["added_tokens"] as! [[String: Any]]
                tokens[0]["content"] = ""
                root["added_tokens"] = tokens
            }),
            ("beginning-of-text token", { root in
                var tokens = root["added_tokens"] as! [[String: Any]]
                tokens.removeFirst()
                root["added_tokens"] = tokens
            }),
            ("merge pair", { root in
                var model = root["model"] as! [String: Any]
                model["merges"] = [42]
                root["model"] = model
            })
        ]
        for (component, mutation) in mutations {
            let data = try changed(mutation)
            #expect(throws: BPETokenizerConfigurationError.unsupported(component)) {
                try BPETokenizer(llama3TokenizerJSON: data)
            }
        }
    }

    @Test("Unknown IDs contribute no bytes and do not discard pending Unicode")
    func missingTokenID() throws {
        let tokenizer = try BPETokenizer(llama3TokenizerJSON: fixture())
        #expect(tokenizer.decode(tokens: [256, -1, 9999]) == "hello")
        var stream = tokenizer.makeStreamDecoder()
        #expect(stream.append(0xe4) == nil)
        #expect(stream.append(9999) == "")
        #expect(stream.append(0xb8) == nil)
        #expect(stream.append(0x96) == "世")
        #expect(stream.finish().isEmpty)
    }

    @Test("Legacy BPE streaming retains per-token decoding without a BOS template")
    func legacyStream() {
        let tokenizer = BPETokenizer(vocab: ["hello</w>": 1, "world</w>": 2], merges: [])
        var stream = tokenizer.makeStreamDecoder()
        #expect(stream.append(1) == "hello")
        #expect(stream.append(2) == "world")
        #expect(stream.append(9999) == "")
        #expect(stream.finish().isEmpty)
        #expect(tokenizer.encode(text: "hello", addSpecialTokens: true) == tokenizer.encode(text: "hello"))
    }

}
