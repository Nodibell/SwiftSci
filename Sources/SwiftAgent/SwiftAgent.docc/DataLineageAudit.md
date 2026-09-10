# Data Lineage & Transformation Audit Trails

Track and inspect the complete lifecycle and data flow of tabular transformations in autonomous agents.

## Overview

When autonomous LLM agents manipulate datasets via DSL commands or tool execution, compliance and debugging require a complete record of every transformation.

`SwiftAgentEvaluator` automatically generates an immutable audit trail of ``LineageRecord`` entries for each command evaluated.

## LineageRecord Structure

Each ``LineageRecord`` contains:
- `stepIndex`: Sequential index of the step within the agent trajectory.
- `operation`: The normalized DSL command executed (e.g., `filter(age)`, `groupBy(category,mean)`).
- `inputRows`: Dataset row count immediately prior to transformation.
- `outputRows`: Dataset row count produced by the transformation.
- `timestamp`: The exact timestamp of execution.

## Example: Accessing Audit Trails

```swift
import SwiftAgent
import SwiftDataFrame

let evaluator = SwiftAgentEvaluator()
let filteredDF = try await evaluator.evaluate(command: "filter age > 25", on: rawDF)
let aggregatedDF = try await evaluator.evaluate(command: "groupby department mean", on: filteredDF)

let auditTrail = await evaluator.lineage
for record in auditTrail {
    print("Step \(record.stepIndex): \(record.operation) [\(record.inputRows) -> \(record.outputRows) rows] at \(record.timestamp)")
}

// Reset audit trail when starting a fresh session:
await evaluator.clearLineage()
```

## Topics

### Audit Types
- ``LineageRecord``
- ``SwiftAgentEvaluator``
