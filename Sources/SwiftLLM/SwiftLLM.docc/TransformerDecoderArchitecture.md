# Local GPU Transformer Decoder

Run local causal autoregressive transformer decoding with Rotary Position Embeddings (RoPE) and SwiGLU activation functions.

## Overview

Execute LLM token generation pipelines natively on Apple Silicon UMA GPUs via MLX integration.

### 1. Model Forward Pass

```swift
import SwiftLLM

let config = LLMConfig(vocabSize: 32000, hiddenDim: 4096, numLayers: 32, numHeads: 32)
let model = LLMModel(config: config)

let inputTokens: [Int] = [1, 512, 1024]
let logits = try await model.forward(tokens: inputTokens)
```
