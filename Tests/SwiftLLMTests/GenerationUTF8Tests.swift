import Testing
import MLX
import MLXNN
import SwiftNLP
@testable import SwiftLLM

@Suite("Generation UTF-8 boundaries", .serialized)
struct GenerationUTF8Tests {
    private struct ByteTokenizer: Tokenizer {
        let bytes: [[UInt8]] = [[], [0xe4], [0xb8], [0x96], [0x21], [], [], []]
        func tokenize(text: String) -> [String] { [text] }
        func encode(text: String) -> [Int] { [0] }
        func decode(tokens: [Int]) -> String {
            String(decoding: tokens.flatMap { bytes[$0] }, as: UTF8.self)
        }
        func makeStreamDecoder() -> TokenStreamDecoder {
            TokenStreamDecoder(decodeBytes: { self.bytes[$0] })
        }
    }

    @Test("Both public streams retain split bytes and flush truncated output", arguments: [false, true])
    func publicStreams(throwing: Bool) async throws {
        let model = TransformerDecoder(config: LLMConfig(vocabSize: 8, numLayers: 0,
            hiddenDim: 8, numHeads: 1, maxSeqLen: 16), tokenizer: ByteTokenizer())
        var head = Array(repeating: Float(0), count: 64)
        for index in 0..<4 { head[(index + 1) * 8 + index] = 1 }
        model.update(parameters: NestedDictionary.unflattened([
            ("embedding.weight", MLXArray.eye(8)),
            ("finalNorm.weight", MLXArray.ones([8])),
            ("lmHead.weight", MLXArray(head, [8, 8]))
        ]))
        for limit in 1...4 {
            let options = LLMOptions(temperature: 0, repetitionPenalty: 1, maxTokens: limit)
            var actual = ""
            if throwing {
                for try await piece in model.generateStream(prompt: "start", options: options) { actual += piece }
            } else {
                for await piece in try await model.generate(prompt: "start", options: options) { actual += piece }
            }
            let expected = ByteTokenizer().decode(tokens: Array(1...limit))
            #expect(Array(actual.utf8) == Array(expected.utf8))
        }
    }
}
