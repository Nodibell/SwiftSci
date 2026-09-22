# Token Sampling & Decoding Strategies

Control autoregressive language generation with Temperature, Top-P Nucleus, Top-K, and Repetition Penalty.

## Overview

During autoregressive language generation, logits from the transformer output layer are converted into probability distributions over the vocabulary. `SwiftLLM` executes sampling transformations strictly at the unnormalized logits level prior to softmax evaluation:

$$\text{Logits} \xrightarrow{\text{Repetition Penalty}} \xrightarrow{\text{Temperature}} \xrightarrow{\text{Top-K Mask (-inf)}} \xrightarrow{\text{Top-P Mask (-inf)}} \xrightarrow{\text{Softmax}} \xrightarrow{\text{Sample}}$$

## 1. Supported Sampling Methods

| Parameter | Mathematical Formulation | Semantics & Defaults |
| :--- | :--- | :--- |
| **Temperature (`temperature`)** | $z_i \leftarrow z_i / T$ | Controls distribution entropy. `1.0` = identity; `0.0` = greedy deterministic argmax. Default: `0.7`. |
| **Top-K (`topK`)** | Masks all tokens outside top $K$ with $-\infty$ | Eliminates low-probability tail tokens. `0` disables Top-K filtering. Default: `40`. |
| **Top-P Nucleus (`topP`)** | Retains cumulative probability $\sum P(w_i) \ge p$ | Dynamically bounds candidates based on probability mass. `1.0` disables Top-P. Default: `0.9`. |
| **Repetition Penalty (`repetitionPenalty`)** | $z_i \leftarrow z_i > 0 \ ? \ z_i / \theta : z_i \cdot \theta$ | Penalizes tokens previously emitted in generation history. `1.0` disables penalty. Default: `1.0`. |

## 2. Configuration & Code Example

Generation options are unified within `SamplingConfiguration`, serving as the single source of truth inside `LLMOptions`:

```swift
import SwiftLLM

// 1. Define decoupled sampling parameters
let sampling = SamplingConfiguration(
    temperature: 0.7,
    topK: 40,
    topP: 0.9,
    repetitionPenalty: 1.1
)

// 2. Configure model generation options
let options = LLMOptions(
    maxTokens: 128,
    stopTokens: [128001, 128009],
    sampling: sampling
)

// 3. Stream generated tokens asynchronously
for try await token in model.generateStream(prompt: "Explain rotary positional embeddings:", options: options) {
    print(token, terminator: "")
}
```

## Topics

### Sampling Types
- ``Sampler``
- ``SamplingConfiguration``
- ``LLMOptions``
