import Foundation
import Testing
@testable import SwiftNLP

@Suite("SentencePiece BPE Tokenizer & Gemma Chat Templates")
struct SentencePieceBPETokenizerTests {
    private static func makeTokenizerJSON(
        vocab: [String: Int],
        merges: [String] = [],
        addedTokens: [[String: Any]] = [],
        metaspace: Bool = true,
        singleTemplate: [[String: Any]]? = [
            ["SpecialToken": ["id": "<s>", "type_id": 0]],
            ["Sequence": ["id": "A", "type_id": 0]],
        ],
        specialTokens: [String: Any] = [
            "<s>": ["id": "<s>", "ids": [1], "tokens": ["<s>"]],
        ]
    ) throws -> Data {
        let normalizer: Any = metaspace
            ? NSNull()
            : ["type": "Replace", "pattern": ["String": " "], "content": "▁"]
        let preTokenizer: [String: Any] = metaspace
            ? [
                "type": "Metaspace",
                "replacement": "▁",
                "prepend_scheme": "first",
                "split": false,
            ]
            : [
                "type": "Split",
                "pattern": ["String": " "],
                "behavior": "MergedWithPrevious",
                "invert": false,
            ]
        var decoders: [[String: Any]] = [
            ["type": "Replace", "pattern": ["String": "▁"], "content": " "],
            ["type": "ByteFallback"],
            ["type": "Fuse"],
        ]
        if metaspace {
            decoders.append(["type": "Strip", "content": " ", "start": 1, "stop": 0])
        }
        let postProcessor: Any = singleTemplate.map {
            [
                "type": "TemplateProcessing",
                "single": $0,
                "special_tokens": specialTokens,
            ] as [String: Any]
        } ?? NSNull()

        let root: [String: Any] = [
            "version": "1.0",
            "added_tokens": addedTokens,
            "normalizer": normalizer,
            "pre_tokenizer": preTokenizer,
            "post_processor": postProcessor,
            "decoder": [
                "type": "Sequence",
                "decoders": decoders,
            ],
            "model": [
                "type": "BPE",
                "dropout": NSNull(),
                "unk_token": "<unk>",
                "continuing_subword_prefix": NSNull(),
                "end_of_word_suffix": NSNull(),
                "fuse_unk": false,
                "byte_fallback": true,
                "vocab": vocab,
                "merges": merges,
            ],
        ]
        return try JSONSerialization.data(withJSONObject: root)
    }

    private static func addAllByteTokens(to vocab: inout [String: Int], omitting: Set<Int> = []) {
        var nextID = (vocab.values.max() ?? 9) + 1
        for byte in 0..<256 where !omitting.contains(byte) {
            vocab[String(format: "<0x%02X>", byte)] = nextID
            nextID += 1
        }
    }

    @Test("Mistral metaspace normalization, BPE merges, UTF-8 byte fallback, and leading space strip")
    func mistralMetaspaceAndByteFallback() throws {
        var vocab: [String: Int] = [
            "<unk>": 0,
            "<s>": 1,
            "</s>": 2,
            "▁": 3,
            "h": 4,
            "i": 5,
            "▁h": 6,
            "▁hi": 7,
        ]
        Self.addAllByteTokens(to: &vocab)
        let e2 = vocab["<0xE2>"]!
        let b98 = vocab["<0x98>"]!
        let b83 = vocab["<0x83>"]!
        let data = try Self.makeTokenizerJSON(
            vocab: vocab,
            merges: ["▁ h", "▁h i"],
            addedTokens: [
                ["id": 0, "content": "<unk>", "special": true],
                ["id": 1, "content": "<s>", "special": true],
                ["id": 2, "content": "</s>", "special": true],
            ],
            metaspace: true
        )
        let tokenizer = try SentencePieceBPETokenizer(tokenizerJSON: data)

        // "hi ☃" -> normalizes to "▁hi▁☃", byte-falls back for '☃' (E2 98 83)
        let rawIDs = tokenizer.encode(text: "hi ☃", addSpecialTokens: false)
        #expect(rawIDs == [7, 3, e2, b98, b83])
        #expect(tokenizer.decode(tokens: rawIDs) == "hi ☃")

        // addSpecialTokens prepends <s> from 2-element TemplateProcessing
        let withBOS = tokenizer.encode(text: "hi", addSpecialTokens: true)
        #expect(withBOS == [1, 7])
        #expect(tokenizer.decode(tokens: withBOS) == "<s> hi")

        // Streaming decoder handles split UTF-8 byte fallback tokens cleanly
        var stream = tokenizer.makeStreamDecoder()
        let parts = rawIDs.map { stream.append($0) ?? "" } + [stream.finish()]
        #expect(parts.joined() == " hi ☃")
    }

    @Test("Gemma 2 ASCII tab fallback works when <0x09> byte token is absent from vocab")
    func gemma2ASCIITabFallbackWithoutHexByte() throws {
        // Deliberately omit <0x09> and provide raw "\t" like Gemma 2 tokenizers
        var vocab: [String: Int] = [
            "<unk>": 0,
            "<bos>": 2,
            "▁": 3,
            "a": 4,
            "b": 5,
            "\t": 6,
        ]
        Self.addAllByteTokens(to: &vocab, omitting: [0x09])
        let data = try Self.makeTokenizerJSON(
            vocab: vocab,
            merges: [],
            addedTokens: [
                ["id": 0, "content": "<unk>", "special": true],
                ["id": 2, "content": "<bos>", "special": true],
            ],
            metaspace: false,
            singleTemplate: [
                ["SpecialToken": ["id": "<bos>", "type_id": 0]],
                ["Sequence": ["id": "A", "type_id": 0]],
            ],
            specialTokens: [
                "<bos>": ["id": "<bos>", "ids": [2], "tokens": ["<bos>"]],
            ]
        )
        let tokenizer = try SentencePieceBPETokenizer(tokenizerJSON: data)
        let ids = tokenizer.encode(text: "a\tb", addSpecialTokens: false)
        #expect(ids == [4, 6, 5])
        #expect(tokenizer.decode(tokens: ids) == "a\tb")
    }

    @Test("Gemma 3 out-of-vocab added tokens use trie longest-match efficiently")
    func gemma3AddedTokensOutOfModelVocabAndTrieLongestMatch() throws {
        var addedTokens: [[String: Any]] = [
            ["id": 0, "content": "<unk>", "special": true],
            ["id": 2, "content": "<bos>", "special": true],
            ["id": 262144, "content": "<image_soft_token>", "special": true],
            ["id": 262145, "content": "<start_of_image>", "special": true],
            ["id": 262146, "content": "<start_of>", "special": false],
        ]
        // Add 2,000 distractor added tokens to exercise the Trie fast path
        for i in 0..<2_000 {
            addedTokens.append([
                "id": 270000 + i,
                "content": "<extra_tok_\(i)>",
                "special": true,
            ])
        }
        var vocab: [String: Int] = [
            "<unk>": 0,
            "<bos>": 2,
            "▁": 3,
            "h": 4,
            "i": 5,
            "hi": 6,
        ]
        Self.addAllByteTokens(to: &vocab)
        let data = try Self.makeTokenizerJSON(
            vocab: vocab,
            merges: ["h i"],
            addedTokens: addedTokens,
            metaspace: false,
            singleTemplate: [
                ["SpecialToken": ["id": "<bos>", "type_id": 0]],
                ["Sequence": ["id": "A", "type_id": 0]],
            ],
            specialTokens: [
                "<bos>": ["id": "<bos>", "ids": [2], "tokens": ["<bos>"]],
            ]
        )
        let tokenizer = try SentencePieceBPETokenizer(tokenizerJSON: data)
        let text = "<start_of_image><image_soft_token>hi<start_of><extra_tok_1999>"
        let ids = tokenizer.encode(text: text, addSpecialTokens: false)
        #expect(ids == [262145, 262144, 6, 262146, 271999])
        #expect(tokenizer.decode(tokens: ids) == text)
    }

    @Test("Gemma 4 single-element TemplateProcessing omits automatic BOS")
    func gemma4SingleElementTemplateOmitsAutoBOS() throws {
        var vocab: [String: Int] = [
            "<unk>": 0,
            "<bos>": 2,
            "▁": 3,
            "h": 4,
            "i": 5,
            "hi": 6,
        ]
        Self.addAllByteTokens(to: &vocab)
        let data = try Self.makeTokenizerJSON(
            vocab: vocab,
            merges: ["h i"],
            addedTokens: [
                ["id": 0, "content": "<unk>", "special": true],
                ["id": 2, "content": "<bos>", "special": true],
            ],
            metaspace: false,
            singleTemplate: [
                ["Sequence": ["id": "A", "type_id": 0]],
            ],
            specialTokens: [:]
        )
        let tokenizer = try SentencePieceBPETokenizer(tokenizerJSON: data)
        #expect(tokenizer.encode(text: "hi", addSpecialTokens: true) == [6])
        #expect(tokenizer.encode(text: "<bos>hi", addSpecialTokens: true) == [2, 6])
    }

    @Test("Gemma 2/3 ChatTemplate folds system message into first user turn")
    func gemmaChatTemplate() {
        let messages: [ChatMessage] = [
            .system("Be concise."),
            .user("Hi"),
            .assistant("Hello!"),
            .user("Next"),
        ]
        let rendered = ChatTemplate.gemma.render(messages: messages, addGenerationPrompt: true)
        let expected =
            "<start_of_turn>user\nBe concise.\n\nHi<end_of_turn>\n" +
            "<start_of_turn>model\nHello!<end_of_turn>\n" +
            "<start_of_turn>user\nNext<end_of_turn>\n" +
            "<start_of_turn>model\n"
        #expect(rendered == expected)
    }

    @Test("Gemma 4 ChatTemplate renders <bos><|turn>role\\n...<turn|>\\n framing")
    func gemma4ChatTemplate() {
        let messages: [ChatMessage] = [
            .system("Act as a scientist."),
            .user("Hi"),
            .assistant("Hello!"),
            .user("Next"),
        ]
        let rendered = ChatTemplate.gemma4.render(messages: messages, addGenerationPrompt: true)
        let expected =
            "<bos><|turn>system\nAct as a scientist.<turn|>\n" +
            "<|turn>user\nHi<turn|>\n" +
            "<|turn>model\nHello!<turn|>\n" +
            "<|turn>user\nNext<turn|>\n" +
            "<|turn>model\n"
        #expect(rendered == expected)
    }

    @Test("Preserves canonically equivalent UTF-8 tokens and U+FEFF sequences without collapsing")
    func preservesCanonicalAndIgnorableByteTokens() throws {
        var byteEntries: [String] = []
        for byte in 0...255 {
            byteEntries.append(String(format: "\"<0x%02X>\": %d", byte, 100 + byte))
        }
        let json = """
        {
          "version": "1.0",
          "truncation": null,
          "padding": null,
          "added_tokens": [
            {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 2, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
          ],
          "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
          "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false},
          "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}]
          },
          "decoder": {
            "type": "Sequence",
            "decoders": [
              {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
              {"type": "ByteFallback"},
              {"type": "Fuse"}
            ]
          },
          "model": {
            "type": "BPE",
            "dropout": null,
            "byte_fallback": true,
            "vocab": {
              "<unk>": 0,
              "<bos>": 2,
              "▁": 3,
              "a": 4,
              "\\u0300": 5,
              "à": 6,
              "a\\u0300": 7,
              "▁à": 8,
              "▁a\\u0300": 9,
              "\\ufeff": 10,
              "\\ufeff\\ufeff": 11,
              "য়": 12,
              "য়": 13,
              "\\u09af": 14,
              "\\u09bc": 15,
              \(byteEntries.joined(separator: ",\n      "))
            },
            "merges": [
              ["a", "\\u0300"],
              ["▁", "à"],
              ["▁", "a\\u0300"],
              ["\\ufeff", "\\ufeff"],
              ["\\u09af", "\\u09bc"]
            ]
          }
        }
        """
        let tokenizer = try SentencePieceBPETokenizer(tokenizerJSON: Data(json.utf8))
        #expect(tokenizer.encode(text: " \u{00E0}", addSpecialTokens: false) == [8])
        #expect(tokenizer.encode(text: " a\u{0300}", addSpecialTokens: false) == [9])
        #expect(tokenizer.encode(text: "\u{FEFF}\u{FEFF}", addSpecialTokens: false) == [11])
        #expect(tokenizer.encode(text: "\u{09AF}\u{09BC}", addSpecialTokens: false) == [12])
        #expect(tokenizer.encode(text: "\u{09DF}", addSpecialTokens: false) == [13])
        #expect(Array(tokenizer.decode(tokens: [8]).utf8) == Array(" \u{00E0}".utf8))
        #expect(Array(tokenizer.decode(tokens: [9]).utf8) == Array(" a\u{0300}".utf8))
        #expect(Array(tokenizer.decode(tokens: [11]).utf8) == Array("\u{FEFF}\u{FEFF}".utf8))
    }
}
