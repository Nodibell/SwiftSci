# ``SwiftWorkflow``

Production Machine Learning Pipelines, Preprocessing Composition, and Artifact Serialization.

## Overview

`SwiftWorkflow` provides immutable, production-grade pipelines that compose missing-value imputation, feature scaling, dimensionality reduction, and classification into single deployable units.

### Key Capabilities

- **Fitted End-to-End Pipeline**: `PCAClassificationWorkflow` integrating mean imputation, standard scaling, SVD-based PCA projection, and binary logistic regression.
- **Strict Row Provenance**: `Predictions` preserving `sourceRowIndices` directly from the input `PreparedSupervisedBatch` or `PreparedNumericBatch` across non-contiguous batch selections.
- **Training-Only Fitting**: Fitted state is computed solely on the designated training batch, preventing data leakage from test or validation sets.
- **Atomic JSON Serialization**: Lightweight, versioned JSON serialization (`save(to:)`, `load(from:)`) enabling cross-process persistence and cold-start inference without retaining raw datasets.

### Example Usage

```swift
import Foundation
import SwiftDataFrame
import SwiftWorkflow

// 1. Prepare supervised numeric batch
let batch = try rawDF.prepareNumericBatch(["signal", "companion", "noise", "target"])
let supervised = try PreparedSupervisedBatch(batch, targetColumn: "target")

// 2. Train workflow strictly on training partition
let trainingBatch = try supervised.selectingRows([0, 1, 2, 4, 5])
let workflow = try await PCAClassificationWorkflow.train(
    training: trainingBatch,
    components: 2,
    learningRate: 0.125,
    epochs: 64
)

// 3. Atomically persist to disk
let fileURL = URL(fileURLWithPath: "model.json")
try workflow.save(to: fileURL)

// 4. Reload and predict with zero test leakage
let loadedWorkflow = try PCAClassificationWorkflow.load(from: fileURL)
let predictions = try await loadedWorkflow.predict(heldOutBatch.features)
```

## Topics

### Workflows
- ``PCAClassificationWorkflow``

### Articles
- <doc:WorkflowPersistence>

