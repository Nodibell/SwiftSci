import Testing
import Foundation
import SwiftAgent
import SwiftLLM

private actor CompletionToolCounter {
    var calls = 0
    func record() { calls += 1 }
}

private enum CompletionMockError: Error, LocalizedError {
    case legacyPath
    case streamFailed

    var errorDescription: String? {
        switch self {
        case .legacyPath: "The rich model used its legacy text path."
        case .streamFailed: "Injected streaming failure."
        }
    }
}

private struct DetailedModel: CompletionReportingLLMModel {
    let events: @Sendable (String) -> [GenerationEvent]
    var failure: CompletionMockError? = nil
    var legacyCalls: CompletionToolCounter? = nil

    func generate(prompt: String, options: LLMOptions) async throws -> AsyncStream<String> {
        await legacyCalls?.record()
        throw CompletionMockError.legacyPath
    }

    func generateDetails(prompt: String, options: LLMOptions) -> AsyncThrowingStream<GenerationEvent, any Error> {
        AsyncThrowingStream { continuation in
            for event in events(prompt) { continuation.yield(event) }
            if let failure { continuation.finish(throwing: failure) }
            else { continuation.finish() }
        }
    }
}

private struct TruncatedModel: CompletionReportingLLMModel {
    let text = "Thought: Use the tool\nAction: Echo\nAction Input: truncated"

    func generate(prompt: String, options: LLMOptions) async throws -> AsyncStream<String> {
        AsyncStream { continuation in
            continuation.yield(text)
            continuation.finish()
        }
    }

    func generateDetails(prompt: String, options: LLMOptions) -> AsyncThrowingStream<GenerationEvent, any Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(.chunk(text))
            continuation.yield(.info(GenerationCompletionInfo(
                promptTokenCount: 17, generationTokenCount: 9, stopReason: .length(.maxTokens))))
            continuation.finish()
        }
    }
}

@Suite("Completion reporting at the native agent boundary")
struct CompletionReportingTests {
    private let action = "Thought: Use the tool\nAction: Echo\nAction Input: partial"
    private let answer = "Thought: Done\nFinal Answer: partial"

    private func info(_ reason: GenerationStopReason = .stop) -> GenerationCompletionInfo {
        GenerationCompletionInfo(promptTokenCount: 17, generationTokenCount: 9, stopReason: reason)
    }

    private func expectRejected(_ events: [GenerationEvent], failure: CompletionMockError? = nil) async throws {
        let counter = CompletionToolCounter()
        let legacy = CompletionToolCounter()
        let memory = SlidingWindowMemory()
        let tool = CustomAgentTool(name: "Echo", description: "Count calls") { input in
            await counter.record()
            return input
        }
        let model: any LLMModel = DetailedModel(events: { _ in events }, failure: failure, legacyCalls: legacy)
        let agent = ReActAgent(tools: [tool], maxSteps: 1, memory: memory)
        await #expect(throws: AgentError.self) {
            _ = try await agent.run(query: "Test", model: model)
        }
        #expect(await counter.calls == 0)
        #expect(await legacy.calls == 0)
        let records = try await memory.retrieveContext(query: "", limit: 20)
        #expect(records.allSatisfy { $0.role != "assistant" })
    }

    @Test("A known truncated action cannot execute a tool")
    func truncatedAction() async throws {
        let counter = CompletionToolCounter()
        let tool = CustomAgentTool(name: "Echo", description: "Count calls") { input in
            await counter.record()
            return input
        }
        let model: any LLMModel = TruncatedModel()
        let agent = ReActAgent(tools: [tool], maxSteps: 1)
        await #expect(throws: AgentError.self) {
            _ = try await agent.run(query: "Test", model: model)
        }
        #expect(await counter.calls == 0)
    }

    @Test("Rich dispatch preserves exact counts through a base existential")
    func completeAnswer() async throws {
        let completion = info()
        let model: any LLMModel = DetailedModel(events: { _ in [
            .chunk("Thought: Done\n"), .chunk("Final Answer: yes"), .info(completion)
        ] })
        let result = try await ReActAgent().run(query: "Test", model: model)
        #expect(result.finalAnswer == "yes")
        #expect(result.trace.count == 1)
        #expect(result.trace[0].completionInfo == completion)
    }

    @Test("Known incomplete action and answer text fail before side effects", arguments: [
        GenerationStopReason.length(.maxTokens), .length(.contextWindow), .cancelled
    ])
    func incompleteStops(reason: GenerationStopReason) async throws {
        for text in [action, answer] {
            try await expectRejected([.chunk(text), .info(info(reason))])
        }
    }

    @Test("Missing, duplicate and nonterminal metadata cannot reach parsing")
    func malformedStreams() async throws {
        let completion = GenerationEvent.info(info())
        let sequences: [[GenerationEvent]] = [
            [], [.chunk(action)], [.chunk(action), completion, completion],
            [.chunk(action), completion, .chunk("")]
        ]
        for events in sequences { try await expectRejected(events) }
    }

    @Test("Stream failure before or after stop metadata rejects the whole response")
    func throwingStreams() async throws {
        let sequences: [[GenerationEvent]] = [
            [.chunk(action)], [.chunk(action), .info(info())], [.chunk(answer), .info(info())]
        ]
        for events in sequences { try await expectRejected(events, failure: .streamFailed) }
    }

    @Test("Each accepted tool or unknown-tool step retains its originating metadata", arguments: ["Echo", "Missing"])
    func trajectoryMetadata(toolName: String) async throws {
        let counter = CompletionToolCounter()
        let first = info()
        let second = GenerationCompletionInfo(promptTokenCount: 38, generationTokenCount: 4, stopReason: .stop)
        let tool = CustomAgentTool(name: "Echo", description: "Count calls") { input in
            await counter.record()
            return input
        }
        let model: any LLMModel = DetailedModel(events: { prompt in
            if prompt.contains("Previous History:") {
                return [.chunk("Final Answer: done"), .info(second)]
            }
            return [.chunk("Thought: Check\nAction: \(toolName)\nAction Input: accepted"), .info(first)]
        })
        let result = try await ReActAgent(tools: [tool], maxSteps: 2).run(query: "Test", model: model)
        #expect(result.finalAnswer == "done")
        #expect(result.trace.count == 2)
        #expect(result.trace[0].completionInfo == first)
        #expect(result.trace[1].completionInfo == second)
        #expect(await counter.calls == (toolName == "Echo" ? 1 : 0))
        if toolName == "Missing" { #expect(result.trace[0].observation?.contains("not recognized") == true) }
    }

    @Test("An invalid later turn does not execute another tool or become an answer")
    func invalidLaterTurn() async throws {
        let counter = CompletionToolCounter()
        let memory = SlidingWindowMemory()
        let first = info()
        let incomplete = info(.length(.contextWindow))
        let text = action
        let tool = CustomAgentTool(name: "Echo", description: "Count calls") { input in
            await counter.record()
            return input
        }
        let model: any LLMModel = DetailedModel(events: { prompt in
            [.chunk(text), .info(prompt.contains("Previous History:") ? incomplete : first)]
        })
        let agent = ReActAgent(tools: [tool], maxSteps: 3, memory: memory)
        await #expect(throws: AgentError.self) { _ = try await agent.run(query: "Test", model: model) }
        #expect(await counter.calls == 1)
        let records = try await memory.retrieveContext(query: "", limit: 20)
        #expect(records.allSatisfy { $0.role != "assistant" })
    }

    @Test("A final stop record without chunks retains existing empty-text parsing")
    func emptyCompleteResponse() async throws {
        let completion = info()
        let model: any LLMModel = DetailedModel(events: { _ in [.info(completion)] })
        let result = try await ReActAgent(maxSteps: 1).run(query: "Test", model: model)
        #expect(result.trace.count == 1)
        #expect(result.trace[0].completionInfo == completion)
    }

    @Test("Legacy model and String closures preserve their APIs and unknown metadata")
    func legacyCompatibility() async throws {
        struct TextModel: LLMModel {
            func generate(prompt: String, options: LLMOptions) async throws -> AsyncStream<String> {
                AsyncStream { continuation in
                    continuation.yield("Final Answer: legacy")
                    continuation.finish()
                }
            }
        }
        let agent = ReActAgent()
        let model: any LLMModel = TextModel()
        let native = try await agent.run(query: "Test", model: model)
        #expect(native.finalAnswer == "legacy")
        #expect(native.trace.allSatisfy { $0.completionInfo == nil })
        let closure: @Sendable (String) async throws -> String = { _ in "Final Answer: closure" }
        let result = try await agent.run(query: "Test", llm: closure)
        #expect(result.finalAnswer == "closure")
        #expect(result.trace.allSatisfy { $0.completionInfo == nil })
        var completed = false
        for await event in await agent.stream(query: "Test", llm: closure) {
            if case .completed(let answer, let trace) = event {
                completed = true
                #expect(answer == "closure")
                #expect(trace.allSatisfy { $0.completionInfo == nil })
            }
        }
        #expect(completed)
        let makeStep: (String, String?, String?, String?) -> AgentStep = AgentStep.init
        #expect(makeStep("legacy", nil, nil, nil).completionInfo == nil)
    }
}
