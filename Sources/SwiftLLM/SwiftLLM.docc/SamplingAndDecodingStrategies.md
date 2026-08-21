# Token Sampling & Decoding Strategies

Control autoregressive language generation with Temperature, Top-P Nucleus, Top-K, and Repetition Penalty.

## Overview

During autoregressive language generation, logits from the transformer output layer are converted into probability distributions over the vocabulary:

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

Where $T$ is the **Temperature**.

## 1. Supported Sampling Methods

| Parameter | Mathematical Formulation | Effect |
| :--- | :--- | :--- |
| **Temperature ($T$)** | Softmax scaling factor | Lower = deterministic/focused; Higher = creative/diverse |
| **Top-K** | Truncate to top $K$ probabilities | Eliminates low-probability tail tokens |
| **Top-P (Nucleus)** | Smallest set with $\sum P(w) \ge p$ | Dynamically adjusts pool size based on confidence |
| **Repetition Penalty** | $z_i \leftarrow z_i / \theta$ if $w_i \in \text{history}$ | Penalizes tokens already generated in current context |

## 2. Code Example

```swift
import SwiftLLM

let params = GenerationParameters(
    temperature: 0.7,
    topP: 0.9,
    topK: 40,
    repetitionPenalty: 1.1,
    maxTokens: 100
)

let outputText = try await model.generate(prompt: "Explain neural attention:", parameters: params)
```

## Topics

### Sampling Types
- ``GenerationParameters``
