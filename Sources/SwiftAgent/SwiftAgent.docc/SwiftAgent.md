# ``SwiftAgent``

Agentic REPL Execution Sandbox & RAG Summary Engine.

## Overview

`SwiftAgent` provides safe REPL execution environments and RAG context summary generators for LLM tool calling.

### Key Capabilities

- **REPL Execution Sandbox**: `SwiftAgentEvaluator` for executing dynamic Swift expressions safely.
- **RAG Context Summaries**: `RAGSummaryGenerator` creating structured prompt context from DataFrames.
- **Tool Calling Integration**: Seamless integration for local LLM agents (e.g. `SwiftLLM`).

### Example Usage

```swift
import SwiftAgent

let summary = RAGSummaryGenerator.generateSummary(for: df)
```

## Topics

### Guides & Tutorials
- <doc:AgenticReplAndRAG>
