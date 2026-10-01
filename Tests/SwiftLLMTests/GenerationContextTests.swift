import Testing
import MLX
import MLXNN
import SwiftNLP
@testable import SwiftLLM

@Suite("Generation context boundaries", .serialized)
struct GenerationContextTests {
    private struct FixtureTokenizer: Tokenizer {
        func tokenize(text: String) -> [String] { [text] }
        func encode(text: String) -> [Int] { Array(repeating: 0, count: text.count) }
        func decode(tokens: [Int]) -> String { String(repeating: "A", count: tokens.count) }
    }

    private func model() -> TransformerDecoder {
        let model = TransformerDecoder(config: LLMConfig(vocabSize: 4, numLayers: 0,
            hiddenDim: 4, numHeads: 1, maxSeqLen: 4), tokenizer: FixtureTokenizer())
        var head = Array(repeating: Float(0), count: 16)
        head[4] = 1
        head[5] = 1
        model.update(parameters: NestedDictionary.unflattened([
            ("embedding.weight", MLXArray.eye(4)),
            ("finalNorm.weight", MLXArray.ones([4])),
            ("lmHead.weight", MLXArray(head, [4, 4]))
        ]))
        return model
    }

    private func output(prompt: String, throwing: Bool, maxTokens: Int = 3) async throws -> String {
        let model = model()
        let options = LLMOptions(sampling: .greedy, maxTokens: maxTokens)
        var text = ""
        if throwing {
            for try await chunk in model.generateStream(prompt: prompt, options: options) { text += chunk }
        } else {
            for await chunk in try await model.generate(prompt: prompt, options: options) { text += chunk }
        }
        return text
    }

    @Test("Oversized prompts are rejected without trimming", arguments: [false, true])
    func oversized(throwing: Bool) async {
        do {
            _ = try await output(prompt: "12345", throwing: throwing)
            Issue.record("Expected oversized prompt to throw")
        } catch {}
    }

    @Test("A full context emits no additional token", arguments: [false, true])
    func full(throwing: Bool) async throws {
        #expect(try await output(prompt: "1234", throwing: throwing) == "")
    }

    @Test("A large output ceiling cannot exceed remaining context", arguments: [false, true])
    func remaining(throwing: Bool) async throws {
        #expect(try await output(prompt: "123", throwing: throwing, maxTokens: Int.max) == "A")
    }
}
