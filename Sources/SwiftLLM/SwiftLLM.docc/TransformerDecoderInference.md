# Autoregressive Transformer Decoder Architecture

Explore the low-level mechanics of Metal GPU-accelerated transformer decoders and KV-Caching in SwiftLLM.

## Overview

`SwiftLLM` implements modern decoder-only generative transformer models (such as Llama, Qwen, and Gemma) leveraging MLX Metal acceleration on Apple Silicon unified memory.

## 1. Key Architectural Blocks

* **Rotary Position Embedding (RoPE)**: Encodes positional information by rotating query and key vectors in 2D planes, preserving relative distance relationships.
* **RMSNorm**: Root Mean Square Layer Normalization provides training stability with reduced computational overhead compared to LayerNorm.
* **SwiGLU Activation**: Gated linear unit activation `SwiGLU(x) = Swish(x · W_1) ⊙ (x · W_2)` for enhanced non-linear representation.
* **Paged KV-Cache**: Stores precomputed Key and Value projection matrices across generation steps, eliminating `O(N²)` recomputation.

## 2. Generating Tokens

```swift
import SwiftLLM

// Load model weights and tokenizer
let config = LLMConfiguration(
    vocabSize: 32000,
    hiddenDimension: 4096,
    numLayers: 32,
    numHeads: 32,
    contextLength: 4096
)

let model = TransformerDecoder(configuration: config)

// Stream autoregressive tokens
for try await token in model.generateStream(prompt: "The future of scientific computing", maxTokens: 50) {
    print(token, terminator: "")
}
```

## Topics

### LLM Components
- ``TransformerDecoder``
- ``KVCache``
