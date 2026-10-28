import Testing
import SwiftAgent

private struct CancellationSignal: Sendable {
    let stream: AsyncStream<Void>
    let continuation: AsyncStream<Void>.Continuation

    init() {
        let pair = AsyncStream<Void>.makeStream()
        stream = pair.stream
        continuation = pair.continuation
    }

    func fire() {
        continuation.yield(())
        continuation.finish()
    }

    func wait() async -> Bool {
        for await _ in stream { return true }
        return false
    }
}

private actor IgnoringCancellationGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private var released = false

    func wait() async {
        if released { return }
        await withCheckedContinuation { continuation = $0 }
    }

    func release() {
        released = true
        continuation?.resume()
        continuation = nil
    }
}

private actor CancellationCounters {
    var modelCalls = 0
    var toolCalls = 0
    func modelStarted() { modelCalls += 1 }
    func toolStarted() { toolCalls += 1 }
}

private final class ProducerLifetime: Sendable {
    let finished: CancellationSignal
    init(finished: CancellationSignal) { self.finished = finished }
    deinit { finished.fire() }
}

private enum ProducerContext {
    @TaskLocal static var lifetime: ProducerLifetime?
}

@Suite("ReAct consumer cancellation", .timeLimit(.minutes(2)))
struct ReActCancellationTests {
    // Readiness is a handshake; the suite deadline guards against deadlock.
    private func observed(_ signal: CancellationSignal) async -> Bool {
        await signal.wait()
    }

    @Test("A pre-cancelled run throws instead of returning empty success")
    func preCancelledRun() async {
        let agent = ReActAgent()
        let consumer = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            do {
                _ = try await agent.run(query: "Test", llm: { _ in "Final Answer: unexpected" })
                return false
            } catch is CancellationError {
                return true
            } catch {
                Issue.record("Expected CancellationError, received \(error)")
                return false
            }
        }
        #expect(await consumer.value)
    }

    @Test("Producer lifetime handshake observes an ordinary completed run")
    func lifetimeControl() async throws {
        let finished = CancellationSignal()
        let agent = ReActAgent()
        var consumer: Task<String, any Error>? = ProducerContext.$lifetime.withValue(ProducerLifetime(finished: finished)) {
            Task { try await agent.run(query: "Test", llm: { _ in "Final Answer: done" }).finalAnswer }
        }
        let answer = try await consumer!.value
        consumer = nil
        #expect(answer == "done")
        try #require(await observed(finished))
    }

    @Test("Cancellation reaches the producer and blocks ignored late output", arguments: [
        "Action: Echo\nAction Input: late", "Final Answer: late"
    ])
    func ignoredCompletion(text: String) async throws {
        let entered = CancellationSignal()
        let cancelled = CancellationSignal()
        let finished = CancellationSignal()
        let gate = IgnoringCancellationGate()
        let counters = CancellationCounters()
        let memory = SlidingWindowMemory()
        let tool = CustomAgentTool(name: "Echo", description: "Count calls") { input in
            await counters.toolStarted()
            return input
        }
        let agent = ReActAgent(tools: [tool], maxSteps: 2, memory: memory)
        var consumer: Task<Bool, Never>? = ProducerContext.$lifetime.withValue(ProducerLifetime(finished: finished)) {
            Task {
                do {
                    _ = try await agent.run(query: "Test", llm: { _ in
                        await counters.modelStarted()
                        return await withTaskCancellationHandler {
                            entered.fire()
                            await gate.wait()
                            return text
                        } onCancel: {
                            cancelled.fire()
                        }
                    })
                    return false
                } catch is CancellationError {
                    return true
                } catch {
                    Issue.record("Expected CancellationError, received \(error)")
                    return false
                }
            }
        }
        let started = await observed(entered)
        consumer!.cancel()
        let rejected = await consumer!.value
        consumer = nil
        let producerCancelled = await observed(cancelled)
        await gate.release()
        let producerFinished = await observed(finished)
        #expect(started)
        #expect(rejected)
        #expect(producerCancelled)
        try #require(producerFinished)
        #expect(await counters.modelCalls == 1)
        #expect(await counters.toolCalls == 0)
        let records = try await memory.retrieveContext(query: "", limit: 20)
        #expect(records.allSatisfy { $0.role != "assistant" })
    }

    @Test("Cancelling a public stream consumer cancels its producer")
    func cancelledStream() async throws {
        let entered = CancellationSignal()
        let cancelled = CancellationSignal()
        let finished = CancellationSignal()
        let gate = IgnoringCancellationGate()
        let counters = CancellationCounters()
        let tool = CustomAgentTool(name: "Echo", description: "Count calls") { input in
            await counters.toolStarted()
            return input
        }
        let agent = ReActAgent(tools: [tool], maxSteps: 1)
        var consumer: Task<Bool, Never>? = ProducerContext.$lifetime.withValue(ProducerLifetime(finished: finished)) {
            Task {
                let stream = await agent.stream(query: "Test", llm: { _ in
                    await withTaskCancellationHandler {
                        entered.fire()
                        await gate.wait()
                        return "Action: Echo\nAction Input: late"
                    } onCancel: {
                        cancelled.fire()
                    }
                })
                var completed = false
                for await event in stream {
                    if case .completed = event { completed = true }
                }
                return completed
            }
        }
        let started = await observed(entered)
        consumer!.cancel()
        let completed = await consumer!.value
        consumer = nil
        let producerCancelled = await observed(cancelled)
        await gate.release()
        let producerFinished = await observed(finished)
        #expect(started)
        #expect(!completed)
        #expect(producerCancelled)
        try #require(producerFinished)
        #expect(await counters.toolCalls == 0)
    }

    @Test("Pre-cancelled stream callers start no producer")
    func preCancelledStream() async throws {
        let finished = CancellationSignal()
        let counters = CancellationCounters()
        let agent = ReActAgent()
        var consumer: Task<Bool, Never>? = ProducerContext.$lifetime.withValue(ProducerLifetime(finished: finished)) {
            Task {
                withUnsafeCurrentTask { $0?.cancel() }
                let stream = await agent.stream(query: "Test", llm: { _ in
                    await counters.modelStarted()
                    return "Final Answer: unexpected"
                })
                var completed = false
                for await event in stream {
                    if case .completed = event { completed = true }
                }
                return completed
            }
        }
        let completed = await consumer!.value
        consumer = nil
        #expect(!completed)
        try #require(await observed(finished))
        #expect(await counters.modelCalls == 0)
    }

    @Test("Consumer cancellation reaches running tools and stops the next turn", arguments: [
        (0.0, 0.0), (30.0, 0.0), (30.0, 6.0)
    ])
    func runningTool(timeout: Double, startupDelay: Double) async throws {
        let entered = CancellationSignal()
        let cancelled = CancellationSignal()
        let released = CancellationSignal()
        let finished = CancellationSignal()
        let counters = CancellationCounters()
        let tool = CustomAgentTool(name: "Echo", description: "Wait cooperatively") { _ in
            await counters.toolStarted()
            return try await withTaskCancellationHandler {
                entered.fire()
                _ = await released.wait()
                try Task.checkCancellation()
                return "released"
            } onCancel: {
                cancelled.fire()
            }
        }
        let agent = ReActAgent(tools: [tool], maxSteps: 2, toolTimeoutSeconds: timeout)
        var consumer: Task<Bool, Never>? = ProducerContext.$lifetime.withValue(ProducerLifetime(finished: finished)) {
            Task {
                do {
                    _ = try await agent.run(query: "Test", llm: { prompt in
                        await counters.modelStarted()
                        if startupDelay > 0 { try await Task.sleep(for: .seconds(startupDelay)) }
                        return prompt.contains("Previous History:") ? "Final Answer: unexpected" : "Action: Echo\nAction Input: wait"
                    })
                    return false
                } catch is CancellationError {
                    return true
                } catch {
                    Issue.record("Expected CancellationError, received \(error)")
                    return false
                }
            }
        }
        let started = await observed(entered)
        consumer!.cancel()
        let rejected = await consumer!.value
        consumer = nil
        let toolCancelled = await observed(cancelled)
        released.fire()
        let producerFinished = await observed(finished)
        #expect(started)
        #expect(rejected)
        #expect(toolCancelled)
        try #require(producerFinished)
        #expect(await counters.modelCalls == 1)
        #expect(await counters.toolCalls == 1)
    }

    @Test("An independent tool cancellation error keeps ordinary observation behavior")
    func independentToolError() async throws {
        let tool = CustomAgentTool(name: "Echo", description: "Throw independently") { _ in
            throw CancellationError()
        }
        let agent = ReActAgent(tools: [tool], maxSteps: 2)
        let result = try await agent.run(query: "Test", llm: { prompt in
            prompt.contains("Previous History:") ? "Final Answer: recovered" : "Action: Echo\nAction Input: error"
        })
        #expect(result.finalAnswer == "recovered")
        #expect(result.trace.count == 2)
        #expect(result.trace[0].observation?.hasPrefix("Error:") == true)
    }
}
