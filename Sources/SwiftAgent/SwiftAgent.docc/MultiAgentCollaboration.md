# Multi-Agent Coordination & Protocols

Orchestrate teams of specialized autonomous agents over an asynchronous message bus with concurrent parallel tool calling.

## Overview

Complex analytical and reasoning workflows are frequently error-prone when performed by a single monolithic model. Decomposing problems into specialized roles—such as **Planner**, **Analyst**, and **Critic**—drastically improves reasoning accuracy, reduces hallucinations, and permits parallelized tool execution.

`SwiftAgent` provides `MultiAgentOrchestrator` and `AgentMessageBus`, a reactive multi-agent framework built on Swift's structured concurrency primitives (`AsyncStream` and `withTaskGroup`).

## Architectural Topology

```
                         ┌───────────────────────┐
                         │   User / Prompt Task  │
                         └───────────┬───────────┘
                                     │
                                     ▼
                       ┌───────────────────────────┐
                       │      AgentMessageBus      │
                       │       (AsyncStream)       │
                       └─────────────┬─────────────┘
                                     │ Broadcasts
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│     Planner      │       │     Analyst      │       │      Critic      │
│ Decomposes steps │       │ Queries & Models │       │ Validates logic  │
└────────┬─────────┘       └────────┬─────────┘       └────────┬─────────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    │ Parallel Tool Calls
                                    ▼
                     ┌─────────────────────────────┐
                     │ MultiAgentOrchestrator      │
                     │ withTaskGroup Tool Executor │
                     └──────────────┬──────────────┘
                                    │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
      DataFrame Query Tool                      Statistical Model Tool
```

## Parallel Tool Calling

When an agent requests multiple tool executions in a single reasoning step (e.g. computing standard deviations across three columns concurrently), `MultiAgentOrchestrator.executeToolsInParallel(_:)` executes them simultaneously across available CPU cores using `withTaskGroup`.

This reduces total latency from $\sum_{i=1}^M T_i$ to $\max_{i} T_i$.

## Example Usage

```swift
import SwiftAgent
import SwiftDataFrame

// 1. Initialize orchestrator and message bus
let orchestrator = MultiAgentOrchestrator(maxRounds: 5)

// 2. Register specialized agents
orchestrator.registerAgent(PlannerAgent())
orchestrator.registerAgent(DataAnalystAgent())
orchestrator.registerAgent(CriticAgent())

// 3. Register tools
orchestrator.registerTool(DataFrameAgentTool(dataframe: salesData))

// 4. Run collaborative execution
let solution = try await orchestrator.run(
    taskPrompt: "Identify top 3 departments by salary variance and critique findings."
)
print("Consensus Answer: \(solution)")
```

## Topics

### Multi-Agent Coordination
- ``MultiAgentOrchestrator``
- ``AgentMessageBus``
- ``AgentMessage``
- ``AgentToolCall``
- ``AgentToolResult``
- ``SpecializedAgent``
