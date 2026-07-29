# Safe Swift REPL & RAG Context Generator

Execute dynamic Swift code expressions in a sandboxed REPL and summarize DataFrame context for LLM agent function calling.

## Overview

Empower local LLM agents to programmatically query, inspect, and manipulate SwiftSci DataFrames in real time.

### 1. REPL Evaluation

```swift
import SwiftAgent

let evaluator = SwiftAgentEvaluator()
let result = try await evaluator.evaluate("df.filter(\"Age\" > 30.0).count")
print("Agent REPL Output: \(result)")
```

### 2. RAG Context Summary Generation

```swift
let summary = RAGSummaryGenerator.generateSummary(for: df)
print("RAG Prompt Context:\n\(summary)")
```
