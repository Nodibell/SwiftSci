#if os(macOS)
import Testing
import Foundation
import MLX
import MLXNN
@testable import SwiftNLP
@testable import SwiftLLM

@Suite("GenerationConfiguration, stopSequences, stopTokenIDs & chat generation 3.14.0", .serialized)
struct GenerationConfigStopSeq3140Tests {

    // MARK: - GenerationConfiguration Parsing & Options

    @Test("GenerationConfiguration parses scalar and array eos_token_id plus sampling & stop fields")
    func testGenerationConfigurationJSONParsing() throws {
        let scalarJSON = """
        {
          "eos_token_id": 2,
          "bos_token_id": 1,
          "pad_token_id": 0,
          "temperature": 0.4,
          "top_k": 50,
          "top_p": 0.95,
          "repetition_penalty": 1.15,
          "max_new_tokens": 256,
          "stop_strings": ["<|end|>", "\\nObservation:"]
        }
        """
        let scalarCfg = try GenerationConfiguration(huggingFaceGenerationConfigJSON: Data(scalarJSON.utf8))
        #expect(scalarCfg.eosTokenIDs == [2])
        #expect(scalarCfg.bosTokenID == 1)
        #expect(scalarCfg.padTokenID == 0)
        #expect(scalarCfg.temperature == 0.4)
        #expect(scalarCfg.topK == 50)
        #expect(scalarCfg.topP == 0.95)
        #expect(scalarCfg.repetitionPenalty == 1.15)
        #expect(scalarCfg.maxNewTokens == 256)
        #expect(scalarCfg.stopStrings == ["<|end|>", "\nObservation:"])

        let arrayJSON = """
        {
          "eos_token_id": [1, 106, 107],
          "bos_token_id": true,
          "temperature": false
        }
        """
        let arrayCfg = try GenerationConfiguration(huggingFaceGenerationConfigJSON: Data(arrayJSON.utf8))
        #expect(arrayCfg.eosTokenIDs == [1, 106, 107])
        #expect(arrayCfg.bosTokenID == nil)
        #expect(arrayCfg.temperature == nil)
        #expect(arrayCfg.stopStrings.isEmpty)

        #expect(throws: LLMModelLoadingError.self) {
            _ = try GenerationConfiguration(huggingFaceGenerationConfigJSON: Data("[1, 2]".utf8))
        }
    }

    @Test("GenerationConfiguration.makeLLMOptions merges overrides with SamplingConfiguration defaults")
    func testMakeLLMOptions() {
        let cfg = GenerationConfiguration(
            eosTokenIDs: [10, 20],
            temperature: 0.25,
            topK: nil,
            topP: 0.85,
            repetitionPenalty: nil,
            maxNewTokens: 512,
            stopStrings: ["STOP"]
        )

        let optionsDefaultMax = cfg.makeLLMOptions()
        let defaults = SamplingConfiguration()
        #expect(optionsDefaultMax.sampling.temperature == 0.25)
        #expect(optionsDefaultMax.sampling.topK == defaults.topK)
        #expect(optionsDefaultMax.sampling.topP == 0.85)
        #expect(optionsDefaultMax.sampling.repetitionPenalty == defaults.repetitionPenalty)
        #expect(optionsDefaultMax.maxTokens == 512)
        #expect(optionsDefaultMax.stopSequences == ["STOP"])
        #expect(optionsDefaultMax.stopTokenIDs == [10, 20])

        let optionsOverrideMax = cfg.makeLLMOptions(maxTokens: 64)
        #expect(optionsOverrideMax.maxTokens == 64)
    }

    @Test("GenerationConfiguration.loadIfPresent and TransformerDecoder.load populate generationConfiguration")
    func testTransformerDecoderLoadPopulatesGenerationConfiguration() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("gen-cfg-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: dir) }
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)

        #expect(try GenerationConfiguration.loadIfPresent(in: dir) == nil)

        let configJSON = """
        {
          "model_type": "llama",
          "hidden_size": 128,
          "num_hidden_layers": 1,
          "num_attention_heads": 4,
          "num_key_value_heads": 4,
          "intermediate_size": 256,
          "vocab_size": 1024,
          "max_position_embeddings": 256,
          "eos_token_id": 2
        }
        """
        try Data(configJSON.utf8).write(to: dir.appendingPathComponent("config.json"))
        try makeMinimalLlamaTokenizerJSON().write(to: dir.appendingPathComponent("tokenizer.json"))

        let cfg = try LLMConfig(huggingFaceConfigJSON: Data(configJSON.utf8))
        let dense = makeMinimalDenseWeights(cfg)
        try MLX.save(arrays: dense, url: dir.appendingPathComponent("model.safetensors"))

        // Without generation_config.json -> nil
        let decoderWithoutGenCfg = try TransformerDecoder.load(directory: dir)
        #expect(decoderWithoutGenCfg.generationConfiguration == nil)
        #expect(decoderWithoutGenCfg.config.eosTokenIDs == [2])

        // With generation_config.json -> populated and eosTokenIDs unioned
        let genJSON = """
        {
          "eos_token_id": [106, 107],
          "temperature": 0.3,
          "max_new_tokens": 128,
          "stop_strings": ["<|end|>"]
        }
        """
        try Data(genJSON.utf8).write(to: dir.appendingPathComponent("generation_config.json"))
        let loaded = try GenerationConfiguration.loadIfPresent(in: dir)
        #expect(loaded?.eosTokenIDs == [106, 107])
        #expect(loaded?.temperature == 0.3)
        #expect(loaded?.maxNewTokens == 128)
        #expect(loaded?.stopStrings == ["<|end|>"])

        let decoderWithGenCfg = try TransformerDecoder.load(directory: dir)
        #expect(decoderWithGenCfg.generationConfiguration == loaded)
        #expect(decoderWithGenCfg.config.eosTokenIDs == [2, 106, 107])
    }

    // MARK: - Sliding Holdback Buffer & stopTokenIDs

    private struct PieceTokenizer: Tokenizer {
        let pieces: [String]
        let promptTokens: [Int]

        init(pieces: [String], promptTokens: [Int] = [0]) {
            self.pieces = pieces
            self.promptTokens = promptTokens
        }

        func tokenize(text: String) -> [String] { [text] }
        func encode(text: String) -> [Int] { promptTokens }
        func decode(tokens: [Int]) -> String {
            var out = ""
            for token in tokens {
                if pieces.indices.contains(token) {
                    out += pieces[token]
                }
            }
            return out
        }
    }

    private func makeLinearChainModel(
        pieces: [String],
        eosTokenIDs: Set<Int> = [],
        architecture: DecoderArchitecture = .llama,
        promptTokens: [Int] = [0]
    ) -> TransformerDecoder {
        let vocab = pieces.count
        var config = LLMConfig(
            vocabSize: vocab,
            numLayers: 0,
            hiddenDim: vocab,
            numHeads: 1,
            maxSeqLen: 32,
            eosTokenIDs: eosTokenIDs
        )
        config.architecture = architecture
        let model = TransformerDecoder(
            config: config,
            tokenizer: PieceTokenizer(pieces: pieces, promptTokens: promptTokens)
        )
        // Token i -> token min(i + 1, vocab - 1) deterministically
        var head = Array(repeating: Float(0), count: vocab * vocab)
        for col in 0..<vocab {
            let nextRow = min(col + 1, vocab - 1)
            head[nextRow * vocab + col] = 1.0
        }
        model.update(parameters: NestedDictionary.unflattened([
            ("embedding.weight", MLXArray.eye(vocab)),
            ("finalNorm.weight", MLXArray.ones([vocab])),
            ("lmHead.weight", MLXArray(head, [vocab, vocab]))
        ]))
        return model
    }

    private func collectDetails(
        _ model: TransformerDecoder,
        prompt: String = "start",
        options: LLMOptions
    ) async throws -> (text: String, chunks: [String], info: GenerationCompletionInfo) {
        var chunks: [String] = []
        var info: GenerationCompletionInfo? = nil
        for try await event in model.generateDetails(prompt: prompt, options: options) {
            switch event {
            case .chunk(let piece):
                chunks.append(piece)
            case .info(let completion):
                info = completion
            }
        }
        let resolvedInfo = try #require(info)
        return (chunks.joined(), chunks, resolvedInfo)
    }

    @Test("stopSequences sliding holdback catches stop sequences split across consecutive tokens")
    func testStopSequenceSplitAcrossTokens() async throws {
        // Generated sequence: "Hello" -> "\nObs" -> "erv" -> "ation:" -> " ignored"
        let pieces = ["<prompt>", "Hello", "\nObs", "erv", "ation:", " ignored"]
        let model = makeLinearChainModel(pieces: pieces)
        let options = LLMOptions(
            sampling: .greedy,
            maxTokens: 10,
            stopSequences: ["\nObservation:", "\nObservation :"]
        )

        let detailed = try await collectDetails(model, options: options)
        #expect(detailed.text == "Hello")
        #expect(detailed.info.stopReason == .stop)
        #expect(detailed.info.generationTokenCount == 4)

        var streamed = ""
        for await piece in try await model.generate(prompt: "start", options: options) {
            streamed += piece
        }
        #expect(streamed == "Hello")
    }

    @Test("stopSequences emits prefix before inline stop sequence and picks earliest match")
    func testStopSequenceInlinePrefixAndEarliest() async throws {
        // Single token contains both prefix and stop sequence; earliest stop wins
        let pieces = ["<prompt>", "Answer=42<end>trailing", "more"]
        let model = makeLinearChainModel(pieces: pieces)
        let options = LLMOptions(
            sampling: .greedy,
            maxTokens: 5,
            stopSequences: ["trailing", "<end>"]
        )

        let detailed = try await collectDetails(model, options: options)
        #expect(detailed.text == "Answer=42")
        #expect(detailed.info.stopReason == .stop)
        #expect(detailed.info.generationTokenCount == 1)
    }

    @Test("stopSequences flushes held-back text cleanly on EOS and maxTokens length limit")
    func testStopSequencesFlushOnEOSAndLength() async throws {
        // Case 1: maxTokens reached without matching stop sequence
        let pieces = ["<prompt>", "Hi", "!", "!!", "extra"]
        let modelLen = makeLinearChainModel(pieces: pieces)
        let optsLen = LLMOptions(sampling: .greedy, maxTokens: 3, stopSequences: ["<|eot_id|>"])
        let lenResult = try await collectDetails(modelLen, options: optsLen)
        #expect(lenResult.text == "Hi!!!")
        #expect(lenResult.info.stopReason == .length(.maxTokens))

        // Case 2: EOS reached before holdback buffer fills
        let modelEOS = makeLinearChainModel(pieces: ["<prompt>", "Hi", "<eos>", "after"], eosTokenIDs: [2])
        let optsEOS = LLMOptions(sampling: .greedy, maxTokens: 5, stopSequences: ["<|eot_id|>"])
        let eosResult = try await collectDetails(modelEOS, options: optsEOS)
        #expect(eosResult.text == "Hi")
        #expect(eosResult.info.stopReason == .stop)
        #expect(eosResult.info.generationTokenCount == 1)
    }

    @Test("LLMOptions.stopTokenIDs stops generation cleanly without emitting the stop token")
    func testOptionsStopTokenIDs() async throws {
        let model = makeLinearChainModel(pieces: ["<prompt>", "One", "Two", "<stop>", "Three"])
        let options = LLMOptions(sampling: .greedy, maxTokens: 6, stopTokenIDs: [3])

        let detailed = try await collectDetails(model, options: options)
        #expect(detailed.text == "OneTwo")
        #expect(detailed.info.stopReason == .stop)
        #expect(detailed.info.generationTokenCount == 2)
    }

    // MARK: - DecoderArchitecture.defaultChatTemplate & generate(messages:)

    @Test("DecoderArchitecture.defaultChatTemplate maps every architecture case")
    func testArchitectureDefaultChatTemplates() {
        #expect(DecoderArchitecture.llama.defaultChatTemplate(date: "26 Nov 2026") == .llama32Instruct(date: "26 Nov 2026"))
        #expect(DecoderArchitecture.mistral.defaultChatTemplate() == .mistral)
        #expect(DecoderArchitecture.gemma2.defaultChatTemplate() == .gemma)
        #expect(DecoderArchitecture.gemma3.defaultChatTemplate() == .gemma)
        #expect(DecoderArchitecture.gemma4.defaultChatTemplate() == .gemma4)
    }

    @Test("TransformerDecoder.generate(messages:) and generateDetails(messages:) merge template stopSequences and avoid duplicate BOS")
    func testMessageGenerationConvenienceOverloads() async throws {
        // Test with a Gemma 4 SentencePieceBPETokenizer to verify single <bos> encoding
        let spTokenizer = try makeGemma4SentencePieceTokenizer()
        var cfg = LLMConfig(vocabSize: 300, numLayers: 0, hiddenDim: 300, numHeads: 1, maxSeqLen: 64)
        cfg.architecture = .gemma4
        let model = TransformerDecoder(config: cfg, tokenizer: spTokenizer)

        // Token 271 is "<turn|>" in makeGemma4SentencePieceTokenizer; make token chain emit "h", "i", "<turn|>"
        let vocab = 300
        var head = Array(repeating: Float(0), count: vocab * vocab)
        // Regardless of previous token, step from prompt's last token (newline=10? wait: use explicit mapping):
        // Chain: any -> 4 ("h"), 4 -> 5 ("i"), 5 -> 271 ("<turn|>"), 271 -> 4
        for col in 0..<vocab {
            if col == 4 {
                head[5 * vocab + col] = 1.0
            } else if col == 5 {
                head[271 * vocab + col] = 1.0
            } else {
                head[4 * vocab + col] = 1.0
            }
        }
        model.update(parameters: NestedDictionary.unflattened([
            ("embedding.weight", MLXArray.eye(vocab)),
            ("finalNorm.weight", MLXArray.ones([vocab])),
            ("lmHead.weight", MLXArray(head, [vocab, vocab]))
        ]))

        let messages = [ChatMessage.user("hi")]
        let options = LLMOptions(sampling: .greedy, maxTokens: 6, stopSequences: ["<turn|>"])

        var detailsChunks: [String] = []
        var info: GenerationCompletionInfo? = nil
        for try await event in model.generateDetails(messages: messages, options: options) {
            switch event {
            case .chunk(let c): detailsChunks.append(c)
            case .info(let i): info = i
            }
        }
        let completion = try #require(info)
        #expect(detailsChunks.joined() == "hi")
        #expect(completion.stopReason == .stop)
        #expect(completion.generationTokenCount == 3)

        let expectedPromptTokens = ChatTemplate.gemma4.encode(messages: messages, using: spTokenizer, addGenerationPrompt: true)
        #expect(expectedPromptTokens.first == 2)
        #expect(expectedPromptTokens.dropFirst().first != 2)
        #expect(completion.promptTokenCount == expectedPromptTokens.count)

        var plain = ""
        for await chunk in try await model.generate(messages: messages, template: .gemma4, options: LLMOptions(sampling: .greedy, maxTokens: 6)) {
            plain += chunk
        }
        #expect(plain == "hi")
    }

    // MARK: - Helpers

    private func makeMinimalDenseWeights(_ config: LLMConfig) -> [String: MLXArray] {
        let hidden = config.hiddenDim
        let kv = hidden / config.numHeads * (config.numKVHeads ?? config.numHeads)
        var w: [String: MLXArray] = [
            "model.embed_tokens.weight": MLXArray.zeros([config.vocabSize, hidden]),
            "model.norm.weight": MLXArray.ones([hidden])
        ]
        if !config.tieWordEmbeddings {
            w["lm_head.weight"] = MLXArray.zeros([config.vocabSize, hidden])
        }
        for i in 0..<config.numLayers {
            let p = "model.layers.\(i)"
            w["\(p).self_attn.q_proj.weight"] = MLXArray.zeros([hidden, hidden])
            w["\(p).self_attn.k_proj.weight"] = MLXArray.zeros([kv, hidden])
            w["\(p).self_attn.v_proj.weight"] = MLXArray.zeros([kv, hidden])
            w["\(p).self_attn.o_proj.weight"] = MLXArray.zeros([hidden, hidden])
            w["\(p).mlp.gate_proj.weight"] = MLXArray.zeros([config.intermediateSize, hidden])
            w["\(p).mlp.up_proj.weight"] = MLXArray.zeros([config.intermediateSize, hidden])
            w["\(p).mlp.down_proj.weight"] = MLXArray.zeros([hidden, config.intermediateSize])
            w["\(p).input_layernorm.weight"] = MLXArray.ones([hidden])
            w["\(p).post_attention_layernorm.weight"] = MLXArray.ones([hidden])
        }
        return w
    }

    private func makeMinimalLlamaTokenizerJSON() throws -> Data {
        let visible = Array(33...126) + Array(161...172) + Array(174...255)
        var vocab: [String: Int] = [:]
        var next = 256
        for byte in 0...255 {
            let scalar = visible.contains(byte) ? byte : next
            if !visible.contains(byte) { next += 1 }
            vocab[String(UnicodeScalar(scalar)!)] = byte
        }
        let special: [[String: Any]] = [[
            "id": 1000, "content": "<|begin_of_text|>", "special": true,
            "single_word": false, "lstrip": false, "rstrip": false, "normalized": false
        ]]
        let regex = #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        let root: [String: Any] = [
            "version": "1.0", "normalizer": NSNull(), "truncation": NSNull(), "padding": NSNull(),
            "added_tokens": special,
            "model": ["type": "BPE", "vocab": vocab, "merges": [], "ignore_merges": true,
                      "dropout": NSNull(), "unk_token": NSNull(), "continuing_subword_prefix": NSNull(),
                      "end_of_word_suffix": NSNull(), "fuse_unk": false, "byte_fallback": false],
            "pre_tokenizer": ["type": "Sequence", "pretokenizers": [
                ["type": "Split", "pattern": ["Regex": regex], "behavior": "Isolated", "invert": false],
                ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false]
            ]],
            "decoder": ["type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true],
            "post_processor": ["type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true]
        ]
        return try JSONSerialization.data(withJSONObject: root)
    }

    private func makeGemma4SentencePieceTokenizer() throws -> SentencePieceBPETokenizer {
        var vocab: [String: Int] = [
            "<unk>": 0,
            "<eos>": 1,
            "<bos>": 2,
            "▁": 3,
            "h": 4,
            "i": 5,
            "hi": 6
        ]
        var nextID = 7
        for byte in 0..<256 {
            vocab[String(format: "<0x%02X>", byte)] = nextID
            nextID += 1
        }
        let tokenizerJSON: [String: Any] = [
            "version": "1.0",
            "added_tokens": [
                ["id": 0, "content": "<unk>", "special": true],
                ["id": 1, "content": "<eos>", "special": true],
                ["id": 2, "content": "<bos>", "special": true],
                ["id": 270, "content": "<|turn>", "special": true],
                ["id": 271, "content": "<turn|>", "special": true]
            ],
            "normalizer": ["type": "Replace", "pattern": ["String": " "], "content": "▁"],
            "pre_tokenizer": [
                "type": "Split",
                "pattern": ["String": " "],
                "behavior": "MergedWithPrevious",
                "invert": false
            ],
            "post_processor": [
                "type": "TemplateProcessing",
                "single": [
                    ["SpecialToken": ["id": "<bos>", "type_id": 0]],
                    ["Sequence": ["id": "A", "type_id": 0]]
                ],
                "special_tokens": [
                    "<bos>": ["id": "<bos>", "ids": [2], "tokens": ["<bos>"]]
                ]
            ],
            "decoder": [
                "type": "Sequence",
                "decoders": [
                    ["type": "Replace", "pattern": ["String": "▁"], "content": " "],
                    ["type": "ByteFallback"],
                    ["type": "Fuse"]
                ]
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
                "merges": ["h i"]
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: tokenizerJSON)
        return try SentencePieceBPETokenizer(tokenizerJSON: data)
    }
}
#endif
