import Testing
import MLX
import MLXNN
import SwiftNLP
@testable import SwiftLLM

@Suite("Structured generation results", .serialized)
struct GenerationResultTests {
    private struct FixtureTokenizer: Tokenizer {
        var bytes = false
        var emptyByteToken = false
        var cancel = false
        var secondTokenText: String? = nil
        func tokenize(text: String) -> [String] { [text] }
        func encode(text: String) -> [Int] { Array(repeating: 0, count: text.count) }
        func decode(tokens: [Int]) -> String {
            if cancel { withUnsafeCurrentTask { $0?.cancel() } }
            return tokens.map { token in
                if token == 2, let secondTokenText { return secondTokenText }
                return ["P", "A", "B", "C", "D", "E", "F", "G"][token]
            }.joined()
        }
        func makeStreamDecoder() -> TokenStreamDecoder {
            if bytes {
                return TokenStreamDecoder(decodeBytes: {
                    if emptyByteToken && $0 == 2 { return [] }
                    return [[], [0xe4], [0xb8], [0x96], [0x21], [], [], []][$0]
                })
            }
            return TokenStreamDecoder(decodeToken: { self.decode(tokens: [$0]) })
        }
    }

    private func model(capacity: Int = 8, stops: Set<Int> = [], bytes: Bool = false,
                       cancel: Bool = false, emptyByteToken: Bool = false,
                       secondTokenText: String? = nil) -> TransformerDecoder {
        let model = TransformerDecoder(config: LLMConfig(vocabSize: 8, numLayers: 0,
            hiddenDim: 8, numHeads: 1, maxSeqLen: capacity, eosTokenIDs: stops),
            tokenizer: FixtureTokenizer(bytes: bytes, emptyByteToken: emptyByteToken, cancel: cancel,
                                        secondTokenText: secondTokenText))
        var head = Array(repeating: Float(0), count: 64)
        for column in 0..<8 { head[min(column + 1, 7) * 8 + column] = 1 }
        model.update(parameters: NestedDictionary.unflattened([
            ("embedding.weight", MLXArray.eye(8)),
            ("finalNorm.weight", MLXArray.ones([8])),
            ("lmHead.weight", MLXArray(head, [8, 8]))
        ]))
        return model
    }

    private func events(_ model: TransformerDecoder, prompt: String = "P", maxTokens: Int = 5) async throws -> [GenerationEvent] {
        var result: [GenerationEvent] = []
        for try await event in model.generateDetails(prompt: prompt,
            options: LLMOptions(sampling: .greedy, maxTokens: maxTokens)) {
            result.append(event)
        }
        return result
    }

    private func info(_ events: [GenerationEvent]) throws -> GenerationCompletionInfo {
        let completions = events.compactMap { event -> GenerationCompletionInfo? in
            if case .info(let info) = event { return info }
            return nil
        }
        #expect(completions.count == 1)
        let result = try #require(completions.first)
        #expect(events.last == .info(result))
        return result
    }

    @Test("EOS completes normally and is excluded from generated counts", arguments: [1, 3])
    func eos(stop: Int) async throws {
        let result = try await events(model(stops: [stop]))
        let completion = try info(result)
        #expect(completion.stopReason == .stop)
        #expect(completion.promptTokenCount == 1)
        #expect(completion.generationTokenCount == stop - 1)
        #expect(result.compactMap(\.chunk).joined() == (stop == 1 ? "" : "AB"))
    }

    @Test("Output and context bounds are distinguished without integer overflow", arguments: [0, 1, 3, 4, Int.max])
    func limits(maxTokens: Int) async throws {
        let result = try await events(model(capacity: 4), maxTokens: maxTokens)
        let completion = try info(result)
        #expect(completion.generationTokenCount == min(maxTokens, 3))
        #expect(completion.stopReason == .length(maxTokens <= 3 ? .maxTokens : .contextWindow))
        #expect(result.compactMap(\.chunk).joined() == String("ABC".prefix(min(maxTokens, 3))))
    }

    @Test("A full prompt returns context metadata without output")
    func fullContext() async throws {
        let result = try await events(model(capacity: 4), prompt: "1234")
        #expect(result.count == 1)
        let completion = try info(result)
        #expect(completion.promptTokenCount == 4)
        #expect(completion.generationTokenCount == 0)
        #expect(completion.stopReason == .length(.contextWindow))
    }

    @Test("The legacy empty-prompt fallback is counted")
    func emptyPrompt() async throws {
        let result = try await events(model(), prompt: "", maxTokens: 1)
        #expect(try info(result).promptTokenCount == 1)
        #expect(result.compactMap(\.chunk).joined() == "A")
    }

    @Test("UTF-8 flush precedes completion and counts tokens rather than chunks", arguments: [1, 2, 3, 4])
    func bytes(maxTokens: Int) async throws {
        let result = try await events(model(bytes: true), maxTokens: maxTokens)
        let expected = maxTokens < 3 ? "�" : (maxTokens == 3 ? "世" : "世!")
        #expect(result.compactMap(\.chunk).joined() == expected)
        #expect(try info(result).generationTokenCount == maxTokens)
    }

    @Test("An empty byte token stops generation and flushes the incomplete scalar")
    func emptyByteTokenStops() async throws {
        let model = model(bytes: true, emptyByteToken: true)
        let result = try await events(model)
        let completion = try info(result)
        #expect(completion.stopReason == .stop)
        #expect(completion.generationTokenCount == 1)
        #expect(result.compactMap(\.chunk).joined() == "�")
        let options = LLMOptions(sampling: .greedy, maxTokens: 5)
        var plain = ""
        for await chunk in try await model.generate(prompt: "P", options: options) { plain += chunk }
        var throwing = ""
        for try await chunk in model.generateStream(prompt: "P", options: options) { throwing += chunk }
        #expect(plain == "�")
        #expect(throwing == plain)
    }

    @Test("Configured EOS IDs take precedence over empty or unknown decoded text", arguments: ["", "<unk>"])
    func configuredEOSOwnsStopping(secondTokenText: String) async throws {
        let model = model(stops: [4], secondTokenText: secondTokenText)
        let result = try await events(model)
        let completion = try info(result)
        let expected = "A" + secondTokenText + "C"
        #expect(completion.stopReason == .stop)
        #expect(completion.generationTokenCount == 3)
        #expect(result.compactMap(\.chunk).joined() == expected)
        #expect(!result.contains(.chunk("")))
        let options = LLMOptions(sampling: .greedy, maxTokens: 5)
        var plain = ""
        for await chunk in try await model.generate(prompt: "P", options: options) { plain += chunk }
        var throwing = ""
        for try await chunk in model.generateStream(prompt: "P", options: options) { throwing += chunk }
        #expect(plain == expected)
        #expect(throwing == expected)
    }

    @Test("An empty byte token consumes budget without stopping when EOS is configured")
    func configuredEOSEmptyBytes() async throws {
        let result = try await events(model(stops: [4], bytes: true, emptyByteToken: true), maxTokens: 2)
        let completion = try info(result)
        #expect(completion.stopReason == .length(.maxTokens))
        #expect(completion.generationTokenCount == 2)
        #expect(result.compactMap(\.chunk).joined() == "�")
        #expect(!result.contains(.chunk("")))
    }

    @Test("Legacy configurations retain unknown-text stopping")
    func legacyUnknownText() async throws {
        let result = try await events(model(secondTokenText: "<unk>"))
        #expect(try info(result).stopReason == .stop)
        #expect(try info(result).generationTokenCount == 1)
        #expect(result.compactMap(\.chunk).joined() == "A")
    }

    @Test("Public generation preserves prompts spanning multiple prefill chunks")
    func multiChunkPrompt() async throws {
        let model = model(capacity: 2053)
        let prompt = String(repeating: "P", count: 2050)
        let result = try await events(model, prompt: prompt, maxTokens: 2)
        let completion = try info(result)
        #expect(completion.promptTokenCount == 2050)
        #expect(completion.generationTokenCount == 2)
        #expect(completion.stopReason == .length(.maxTokens))
        #expect(result.compactMap(\.chunk).joined() == "AB")
        let options = LLMOptions(sampling: .greedy, maxTokens: 2)
        var plain = ""
        for await chunk in try await model.generate(prompt: prompt, options: options) { plain += chunk }
        var throwing = ""
        for try await chunk in model.generateStream(prompt: prompt, options: options) { throwing += chunk }
        #expect(plain == "AB" && throwing == plain)
    }

    @Test("Both legacy adapters preserve detailed-stream text")
    func adapters() async throws {
        let model = model(stops: [3])
        let options = LLMOptions(sampling: .greedy, maxTokens: 5)
        let detail = try await events(model).compactMap(\.chunk).joined()
        var plain = ""
        for await chunk in try await model.generate(prompt: "P", options: options) { plain += chunk }
        var throwing = ""
        for try await chunk in model.generateStream(prompt: "P", options: options) { throwing += chunk }
        #expect(detail == "AB")
        #expect(plain == detail)
        #expect(throwing == detail)
    }

    @Test("All entry points return the typed request error", arguments: [0, 1, 2])
    func invalidRequests(api: Int) async throws {
        for (capacity, prompt, budget, expected): (Int, String, Int, GenerationError) in [
            (4, "12345", 1, .promptTooLong(promptTokens: 5, capacity: 4)),
            (4, "P", -1, .invalidMaxTokens(-1)),
            (0, "P", 1, .invalidContextLimit(0)),
            (-1, "P", 1, .invalidContextLimit(-1)),
            (Int.min, "P", 1, .invalidContextLimit(Int.min))
        ] {
            let model = model(capacity: capacity)
            #expect(model.config.maxSeqLen == capacity)
            var emitted = 0
            let options = LLMOptions(sampling: .greedy, maxTokens: budget)
            do {
                switch api {
                case 0:
                    for await _ in try await model.generate(prompt: prompt, options: options) { emitted += 1 }
                case 1:
                    for try await _ in model.generateStream(prompt: prompt, options: options) { emitted += 1 }
                default:
                    for try await _ in model.generateDetails(prompt: prompt, options: options) { emitted += 1 }
                }
                Issue.record("Expected request error \(expected)")
            } catch {
                #expect(error as? GenerationError == expected)
            }
            #expect(emitted == 0)
        }
    }

    @Test("Producer cancellation cannot be mistaken for normal completion")
    func cancelledProducer() async throws {
        let result = try await events(model(cancel: true))
        let completion = try info(result)
        #expect(result.compactMap(\.chunk).joined() == "A")
        #expect(completion.stopReason == .cancelled)
        #expect(completion.generationTokenCount == 1)
    }

    private struct UnexpectedTokenizer: Tokenizer {
        func tokenize(text: String) -> [String] { [text] }
        func encode(text: String) -> [Int] {
            Issue.record("A cancelled caller must not start tokenization")
            return [0]
        }
        func decode(tokens: [Int]) -> String {
            Issue.record("A cancelled caller must not start generation")
            return "A"
        }
    }

    @Test("A cancelled caller cannot start generation", arguments: [0, 1, 2])
    func cancelledCaller(api: Int) async throws {
        let model = TransformerDecoder(config: LLMConfig(vocabSize: 8, numLayers: 0,
            hiddenDim: 8, numHeads: 1, maxSeqLen: 8), tokenizer: UnexpectedTokenizer())
        let cancelled = await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            let options = LLMOptions(sampling: .greedy, maxTokens: 1)
            do {
                switch api {
                case 0:
                    _ = try await model.generate(prompt: "P", options: options)
                case 1:
                    for try await _ in model.generateStream(prompt: "P", options: options) {}
                default:
                    for try await _ in model.generateDetails(prompt: "P", options: options) {}
                }
                return api != 0 && Task.isCancelled
            } catch is CancellationError {
                return true
            } catch {
                return false
            }
        }.value
        #expect(cancelled)
    }
}
