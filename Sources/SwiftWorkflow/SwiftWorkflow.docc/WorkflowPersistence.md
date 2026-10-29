# Deploying and Persisting Scientific Workflows

Learn how to safely persist, reload, and evaluate end-to-end machine learning workflows across processes.

## Overview

In production scientific machine learning systems, data leakage between training and evaluation phases is one of the most common causes of overly optimistic validation metrics. `SwiftWorkflow` eliminates leakage by enforcing structural invariants at the API level:

1. **Explicit Supervised Batch Splitting**: Workflows accept `PreparedSupervisedBatch` instances where row selection explicitly tracks `originalRowIndices`.
2. **Immutable Stages**: Once trained, a workflow exposes no mutable hyperparameters or refitting paths.
3. **Atomic Schema Validation**: When loading from JSON, dimensions and feature names are checked against the artifact header before initializing models.

### End-to-End Persistence Pattern

```swift
import Foundation
import SwiftDataFrame
import SwiftWorkflow

// Train workflow on designated training rows
let workflow = try await PCAClassificationWorkflow.train(
    training: trainingBatch,
    components: 2,
    learningRate: 0.125,
    epochs: 64
)

// Serialize fitted stages to JSON artifact
let artifactURL = URL(fileURLWithPath: "PCAWorkflow.json")
try workflow.save(to: artifactURL)

// Reload across independent processes with verified dimensions
let reloaded = try PCAClassificationWorkflow.load(from: artifactURL)
let predictions = try await reloaded.predict(unlabeledBatch)
```
