import Testing
import Foundation
import SwiftLLM
import SwiftNLP
@testable import SwiftAgent

@Suite("ReActAgent 3.14.0 Stream & ChatTemplate Tests")
struct ReActAgent3140Tests {

    private actor RecordedCalls {
        var prompts: [String] = []
        var options: [LLMOptions] = []

        func append(prompt: String, option: LLMOptions) {
            prompts.append(prompt)
            options.append(option)
        }
    }

    private struct RecordingCompletionModel: CompletionReportingLLMModel {
        let calls: RecordedCalls
        let responder: @Sendable (String) -> String

        func generate(prompt: String, options: LLMOptions) async throws -> AsyncStream<String> {
            await calls.append(prompt: prompt, option: options)
            let reply = responder(prompt)
            return AsyncStream { continuation in
                continuation.yield(reply)
                continuation.finish()
            }
        }

        func generateDetails(prompt: String, options: LLMOptions) -> AsyncThrowingStream<GenerationEvent, any Error> {
            AsyncThrowingStream { continuation in
                Task {
                    await calls.append(prompt: prompt, option: options)
                    let reply = responder(prompt)
                    continuation.yield(.chunk(reply))
                    continuation.yield(.info(GenerationCompletionInfo(
                        promptTokenCount: 12,
                        generationTokenCount: 6,
                        stopReason: .stop
                    )))
                    continuation.finish()
                }
            }
        }
    }

    private struct PlainTextModel: LLMModel {
        let calls: RecordedCalls
        let responder: @Sendable (String) -> String

        func generate(prompt: String, options: LLMOptions) async throws -> AsyncStream<String> {
            await calls.append(prompt: prompt, option: options)
            let reply = responder(prompt)
            return AsyncStream { continuation in
                continuation.yield(reply)
                continuation.finish()
            }
        }
    }

    @Test("ReActAgent.stream(query:model:options:chatTemplate:) streams tool lifecycle and completion")
    func testStreamWithModelAndChatTemplate() async throws {
        let calls = RecordedCalls()
        let tool = CustomAgentTool(name: "Lookup", description: "Looks up facts") { input in
            "Result for \(input)"
        }
        let model = RecordingCompletionModel(calls: calls) { prompt in
            if !prompt.contains("Previous History:") {
                return "Thought: Need lookup.\nAction: Lookup\nAction Input: swift"
            }
            return "Thought: Have result.\nFinal Answer: Found Swift."
        }

        let agent = ReActAgent(tools: [tool], maxSteps: 3)
        let inputOptions = LLMOptions(temperature: 0.1, maxTokens: 256, stopSequences: ["\nObservation:", "CUSTOM"])
        var events: [AgentStreamEvent] = []
        for await event in await agent.stream(
            query: "What is Swift?",
            model: model,
            options: inputOptions,
            chatTemplate: .chatML
        ) {
            events.append(event)
        }

        var sawThought = false
        var sawScheduled = false
        var sawStarted = false
        var sawCompletedTool = false
        var sawAnswer = false
        var completedAnswer: String? = nil
        var completedTrace: [AgentStep] = []

        for event in events {
            switch event {
            case .thoughtDelta:
                sawThought = true
            case .toolCallScheduled(_, let name, let args):
                sawScheduled = (name == "Lookup" && args["input"] == "swift")
            case .toolExecutionStarted(_, let name):
                sawStarted = (name == "Lookup")
            case .toolExecutionCompleted(_, let name, let out, _):
                sawCompletedTool = (name == "Lookup" && out.text == "Result for swift")
            case .answerDelta(let ans):
                sawAnswer = (ans == "Found Swift.")
            case .completed(let finalAnswer, let trace):
                completedAnswer = finalAnswer
                completedTrace = trace
            default:
                break
            }
        }

        #expect(sawThought)
        #expect(sawScheduled)
        #expect(sawStarted)
        #expect(sawCompletedTool)
        #expect(sawAnswer)
        #expect(completedAnswer == "Found Swift.")
        #expect(completedTrace.count == 2)
        #expect(completedTrace[0].completionInfo?.stopReason == .stop)

        // Verify ChatML rendering and deduplicated stop sequences
        let recordedPrompts = await calls.prompts
        let recordedOptions = await calls.options
        #expect(recordedPrompts.count == 2)
        #expect(recordedPrompts[0].contains("<|im_start|>system\n"))
        #expect(recordedPrompts[0].contains("<|im_start|>user\nWhat is Swift?<|im_end|>\n<|im_start|>assistant\n"))
        #expect(recordedPrompts[1].contains("Previous History:"))

        let stops = try #require(recordedOptions.first?.stopSequences)
        #expect(stops.contains("<|im_end|>"))
        #expect(stops.contains("\nObservation:"))
        #expect(stops.contains("\nObservation :"))
        #expect(stops.contains("CUSTOM"))
        #expect(stops.filter { (s: String) -> Bool in s == "\nObservation:" }.count == 1)
    }

    @Test("ReActAgent.run preserves plain-string prompting when chatTemplate is nil on non-TransformerDecoder model while injecting Observation stops")
    func testRunPreservesPlainPromptForNonDecoderModel() async throws {
        let calls = RecordedCalls()
        let model = PlainTextModel(calls: calls) { _ in
            "Thought: Direct.\nFinal Answer: Plain OK"
        }
        let agent = ReActAgent()
        let result = try await agent.run(query: "Hello?", model: model)
        #expect(result.finalAnswer == "Plain OK")

        let recordedPrompts = await calls.prompts
        let recordedOptions = await calls.options
        #expect(recordedPrompts.count == 1)
        #expect(recordedPrompts[0].hasPrefix("You are an autonomous AI analyst answering the following question: 'Hello?'."))
        #expect(!recordedPrompts[0].contains("<|im_start|>"))
        #expect(recordedOptions[0].stopSequences == ["\nObservation:", "\nObservation :"])
    }

    @Test("ReActAgent.run formats prompt with explicit chatTemplate")
    func testRunWithExplicitChatTemplate() async throws {
        let calls = RecordedCalls()
        let model = PlainTextModel(calls: calls) { _ in
            "Thought: Formatted.\nFinal Answer: Llama3 OK"
        }
        let agent = ReActAgent()
        let result = try await agent.run(
            query: "Explain MLX",
            model: model,
            options: LLMOptions(temperature: 0.2, maxTokens: 128),
            chatTemplate: .llama3
        )
        #expect(result.finalAnswer == "Llama3 OK")

        let recordedPrompts = await calls.prompts
        let recordedOptions = await calls.options
        #expect(recordedPrompts.count == 1)
        #expect(recordedPrompts[0].contains("<|start_header_id|>system<|end_header_id|>\n\n"))
        #expect(recordedPrompts[0].contains("<|start_header_id|>user<|end_header_id|>\n\nExplain MLX<|eot_id|>"))
        #expect(recordedPrompts[0].hasSuffix("<|start_header_id|>assistant<|end_header_id|>\n\n"))
        #expect(recordedOptions[0].stopSequences == ["<|eot_id|>", "\nObservation:", "\nObservation :"])
    }

    @Test("Pre-cancelled ReActAgent.stream(query:model:) finishes immediately without invoking model")
    func testPreCancelledStreamWithModel() async {
        let calls = RecordedCalls()
        let model = PlainTextModel(calls: calls) { _ in "Final Answer: should not run" }
        let agent = ReActAgent()

        let finished = await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            var events: [AgentStreamEvent] = []
            for await event in await agent.stream(query: "Test", model: model, chatTemplate: .gemma4) {
                events.append(event)
            }
            return events.isEmpty
        }.value

        #expect(finished)
        #expect(await calls.prompts.isEmpty)
    }
}
