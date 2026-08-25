#if os(macOS)
import Testing
import Foundation
import MLX
@testable import SwiftLLM

@Suite("JSON Grammar Constrained Decoding Tests")
struct JSONGrammarDecoderTests {

    struct UserProfile: Codable, Equatable {
        let name: String
        let age: Int
        let active: Bool
    }

    @Test("JSONGrammarDecoder advances through valid JSON sequence")
    func testGrammarStateTransitions() throws {
        let decoder = JSONGrammarDecoder()
        #expect(decoder.state == .expectRootOpen)

        decoder.advance(with: "{")
        #expect(decoder.state == .expectKeyOrClose)

        decoder.advance(with: "\"name\"")
        #expect(decoder.state == .expectColon)

        decoder.advance(with: ":")
        #expect(decoder.state == .expectValue)

        decoder.advance(with: "\"Alice\"")
        #expect(decoder.state == .expectCommaOrClose)

        decoder.advance(with: ",")
        #expect(decoder.state == .expectKeyOrClose)

        decoder.advance(with: "\"age\": 30 ")
        #expect(decoder.state == .expectCommaOrClose)

        decoder.advance(with: "}")
        #expect(decoder.state == .completed)

        let jsonData = decoder.accumulatedText.data(using: .utf8)!
        let parsed = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any]
        #expect(parsed?["name"] as? String == "Alice")
        #expect(parsed?["age"] as? Int == 30)
    }

    @Test("JSONGrammarDecoder masks invalid logits")
    func testLogitMasking() throws {
        let decoder = JSONGrammarDecoder()
        let logits = MLXArray([Float]([10.0, 5.0, 2.0, 8.0]))
        let vocab: [Int: String] = [
            0: "hello",
            1: "{",
            2: "world",
            3: " {"
        ]

        let masked = decoder.maskLogits(logits, vocab: vocab)
        let maskedVals = masked.asArray(Float.self)

        // Only tokens 1 and 3 contain "{" and are valid in expectRootOpen state
        #expect(maskedVals[0] < -1e8)
        #expect(maskedVals[1] == 5.0)
        #expect(maskedVals[2] < -1e8)
        #expect(maskedVals[3] == 8.0)
    }
}
#endif
