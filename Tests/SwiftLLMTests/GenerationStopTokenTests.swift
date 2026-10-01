import Testing
import MLX
import MLXNN
import SwiftNLP
@testable import SwiftLLM

@Suite("Generation stop tokens", .serialized)
struct GenerationStopTokenTests {
    private struct FixtureTokenizer: Tokenizer {
        let stop: Int
        func tokenize(text: String) -> [String] { [text] }
        func encode(text: String) -> [Int] {
            switch text {
            case "immediate": [1]
            case "stop in prompt": [stop, 0]
            default: [0]
            }
        }
        func decode(tokens: [Int]) -> String {
            tokens.map { ["P", "A", "<end>", "<turn>"][$0] }.joined()
        }
    }

    private func model(stops: Set<Int>, selected: Int = 2) -> TransformerDecoder {
        let config = LLMConfig(vocabSize: 4, numLayers: 0, hiddenDim: 4, numHeads: 1,
            maxSeqLen: 16, eosTokenIDs: stops)
        let model = TransformerDecoder(config: config, tokenizer: FixtureTokenizer(stop: selected))
        var head = Array(repeating: Float(0), count: 16)
        head[4] = 1
        head[selected * 4 + 1] = 1
        head[selected * 4 + selected] = 1
        model.update(parameters: NestedDictionary.unflattened([
            ("embedding.weight", MLXArray.eye(4)),
            ("finalNorm.weight", MLXArray.ones([4])),
            ("lmHead.weight", MLXArray(head, [4, 4]))
        ]))
        return model
    }

    private func output(_ model: TransformerDecoder, prompt: String = "start", throwing: Bool) async throws -> String {
        let options = LLMOptions(temperature: 0, repetitionPenalty: 1, maxTokens: 3)
        var result = ""
        if throwing {
            for try await piece in model.generateStream(prompt: prompt, options: options) { result += piece }
        } else {
            for await piece in try await model.generate(prompt: prompt, options: options) { result += piece }
        }
        return result
    }

    @Test("Both streams suppress every configured stop token", arguments: [false, true])
    func stops(throwing: Bool) async throws {
        for selected in [2, 3] {
            let model = model(stops: [2, 3], selected: selected)
            #expect(try await output(model, throwing: throwing) == "A")
            #expect(try await output(model, prompt: "immediate", throwing: throwing) == "")
            #expect(try await output(model, prompt: "stop in prompt", throwing: throwing) == "A")
        }
    }

    @Test("Unconfigured token IDs retain legacy generation", arguments: [false, true])
    func legacy(throwing: Bool) async throws {
        for stops: Set<Int> in [[], [3]] {
            #expect(try await output(model(stops: stops), throwing: throwing) == "A<end><end>")
        }
    }

    @Test("Only the Llama 3.2 preset opts into its checkpoint stop tokens")
    func defaults() {
        #expect(LLMConfig.llama1B.eosTokenIDs == [128001, 128008, 128009])
        #expect(LLMConfig.llama3_1B.eosTokenIDs == LLMConfig.llama1B.eosTokenIDs)
        #expect(LLMConfig.debug.eosTokenIDs.isEmpty)
        #expect(LLMConfig.llama8B.eosTokenIDs.isEmpty)
    }
}
