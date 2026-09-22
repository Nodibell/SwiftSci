# Autonomous Data Analysis Agents & ReAct Reasoning Loops

Build autonomous, sandboxed AI agents that analyze `SwiftDataFrame` tables, execute statistical aggregations, and resolve tools using fuzzy matching.

## Overview

`SwiftAgent` implements an agentic reasoning loop based on the **ReAct (Reason + Act)** paradigm. The agent iteratively reasons about a user query, selects analytical tools via fuzzy matching, inspects sandboxed observations, and returns structured data insights.

```
                      User Prompt
                           │
                           ▼
          ┌──────────────────────────────────┐
     ┌───►│ Thought: Analyze missing values  │
     │    └────────────────┬─────────────────┘
     │                     ▼
     │    ┌──────────────────────────────────┐
     │    │ Action: Run 'data_summary' tool  │
     │    └────────────────┬─────────────────┘
     │                     ▼
     │    ┌──────────────────────────────────┐
ReAct│    │ Fuzzy Tool Resolution            │
Loop │    │ (handles typos/case/spacing)     │
     │    └────────────────┬─────────────────┘
     │                     ▼
     │    ┌──────────────────────────────────┐
     │    │ Observation: 120 rows, 4 columns │
     │    └────────────────┬─────────────────┘
     │                     │
     └─────────────────────┘ (Repeat until final answer)
                           │
                           ▼
                     Final Answer
```

---

## 1. Running an Autonomous ReAct Agent

Configure a `ReActAgent` with custom tools and an LLM generation closure:

```swift
import Foundation
import SwiftAgent
import SwiftDataFrame

// 1. Prepare sample dataset
let df = try DataFrame(columns: [
    TypedColumn<String>(name: "department", values: ["Engineering", "Sales", "Engineering", "Marketing"]),
    TypedColumn<Double>(name: "salary", values: [120000.0, 95000.0, 135000.0, 85000.0])
])

// 2. Define analytical tools
let summaryTool = CustomAgentTool(
    name: "dataset_summary",
    description: "Returns row count and column data types of the DataFrame."
) { _ in
    return "DataFrame has \(df.shape.rows) rows and \(df.shape.columns) columns: [department: String, salary: Double]"
}

// 3. Mock LLM reasoning response closure
let llmModel: @Sendable (String) async throws -> String = { prompt in
    if prompt.contains("Action:") {
        return "Thought: I need to inspect the dataset summary.\nAction: dataset_summary\nAction Input: {}\n"
    } else {
        return "Thought: I have the information.\nFinal Answer: The dataset contains 4 employee records across Engineering, Sales, and Marketing."
    }
}

// 4. Initialize and execute the ReAct Agent
let agent = ReActAgent(
    tools: [summaryTool],
    llm: llmModel,
    maxSteps: 5
)

let trajectory = try await agent.run(question: "How many employees are in the dataset?")

print("=== Agent Trajectory Completed ===")
print("Final Answer: \(trajectory.finalAnswer)")
print("Total Steps Executed: \(trajectory.steps.count)")
```

---

## 2. Sandboxed Command Interpretation (`SwiftAgentEvaluator`)

Safely evaluate structured AST commands without arbitrary code execution vulnerabilities:

```swift
import SwiftAgent

let evaluator = SwiftAgentEvaluator(dataFrame: df)

// 1. Parse and execute pipeline commands
let filtered = try evaluator.execute("filter salary > 100000")
print("Filtered count: \(filtered.shape.rows) employees with salary > $100k.")

// 2. GroupBy aggregation
let grouped = try evaluator.execute("groupby department")
print("Department groupings: \(grouped.shape.rows) rows.")
```

---

## 3. Tool Parameter Schema Validation & Error Resilience

### Defining Structured Tool Schemas

Tools implementing `AgentTool` define their expected parameters using `AgentParameterSchema`:

```swift
import SwiftAgent

let mathTool = CustomAgentTool(
    name: "calculator",
    description: "Evaluates mathematical expressions.",
    schema: AgentParameterSchema(
        properties: [
            "expression": AgentParameterProperty(
                type: "string",
                description: "Arithmetic expression to evaluate (e.g. '2 + 2')"
            )
        ],
        required: ["expression"]
    )
) { args in
    guard let expr = args["expression"] else {
        throw AgentError.invalidInput("Missing 'expression'")
    }
    return "Result of \(expr): 4"
}
```

### Self-Correction & Reasoning Loop Resilience

If the language model generates malformed JSON syntax or omits mandatory parameters, `ReActAgent` validates the payload against `AgentParameterSchema`:
- Instead of terminating the program or throwing unhandled errors, `ReActAgent` captures `SchemaValidationError`.
- The validation error description (e.g. `"Error: Malformed tool call: Missing required parameter 'expression'. Please correct the parameters and retry."`) is captured as the tool's `observation`.
- This feedback is injected into the subsequent prompt step, empowering the LLM to autonomously correct its parameters and retry on the next reasoning turn.

