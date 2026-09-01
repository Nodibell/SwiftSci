# Transformer Architecture & LLM Inference Guide

High-throughput, on-device causal language model execution on Apple Silicon Unified Memory Architecture using `TransformerDecoder`, `PagedKVCache`, `GGUFParser`, and `SafeTensorsParser`.

## Architecture Overview

`SwiftLLM` implements the modern Llama-3 / Mistral autoregressive causal decoder architecture natively in pure Swift and MLX:

```
                          ┌────────────────────────┐
                          │    Input Token IDs     │
                          └───────────┬────────────┘
                                      ▼
                          ┌────────────────────────┐
                          │   Token Embeddings     │
                          └───────────┬────────────┘
                                      ▼
                      ┌────────────────────────────────┐
                      │ ┌────────────────────────────┐ │
                      │ │          RMSNorm           │ │
                      │ └─────────────┬──────────────┘ │
                      │               ▼                │
                      │ ┌────────────────────────────┐ │
                      │ │ Multi-Head Attention + RoPE│ │
                      │ └─────────────┬──────────────┘ │
                      │               ▼                │
                      │ ┌────────────────────────────┐ │
                      │ │          Residual          │ │
                      │ └─────────────┬──────────────┘ │
                      │               ▼                │
                      │ ┌────────────────────────────┐ │
                      │ │          RMSNorm           │ │
                      │ └─────────────┬──────────────┘ │
                      │               ▼                │
                      │ ┌────────────────────────────┐ │
                      │ │      SwiGLU FFN (SiLU)     │ │
                      │ └─────────────┬──────────────┘ │
                      │               ▼                │
                      │ ┌────────────────────────────┐ │
                      │ │          Residual          │ │
                      │ └─────────────┬──────────────┘ │
                      │ └─────────────┬──────────────┘ │ x N Layers
                      └───────────────┼────────────────┘
                                      ▼
                          ┌────────────────────────┐
                          │     Final RMSNorm      │
                          └───────────┬────────────┘
                                      ▼
                          ┌────────────────────────┐
                          │  LM Head (Logits)      │
                          └────────────────────────┘
```

---

## 1. Core Architectural Components

### RoPE (Rotary Position Embedding)
Applies 2D rotation to query and key vectors based on absolute position index `m`, preserving relative token distance properties without learned position embeddings.

### SwiGLU Feed-Forward Network
Replaces legacy ReLU/GELU activations with Swish-Gated Linear Units (Llama-style):

> **Formula:** `SwiGLU(x) = (SiLU(x · W_gate) ⊙ (x · W_up)) · W_down`

### Paged KV-Cache
Manages attention Key-Value projection states in fixed-size contiguous memory pages (vLLM architecture), eliminating RAM fragmentation during dynamic multi-turn autoregressive decoding.

---

## 2. Parsing Weight Formats (SafeTensors & GGUF)

`SwiftLLM` includes strict parsers for both HuggingFace `.safetensors` headers and `GGUF` quantized binary models:

```swift
import Foundation
import SwiftLLM

// 1. Parse SafeTensors weight metadata
let safeTensorsURL = URL(fileURLWithPath: "model.safetensors")
let (tensors, metadata) = try SafeTensorsParser.parse(fileURL: safeTensorsURL)

print("Found \(tensors.count) tensor layers in SafeTensors archive.")
for (tensorName, info) in tensors.prefix(5) {
    print("Layer '\(tensorName)': dtype=\(info.dtype), shape=\(info.shape)")
}

// 2. Parse GGUF Quantized Binary Archive
let ggufURL = URL(fileURLWithPath: "llama-3-8b.Q4_K_M.gguf")
let ggufModel = try GGUFParser.parse(fileURL: ggufURL)
print("Loaded GGUF Model with \(ggufModel.tensors.count) quantized tensors.")
```

---

## 3. End-to-End Decoder Configuration & Inference

Construct an N-layer `TransformerDecoder` using standard presets (`LLMConfig.llama3_8B` or `LLMConfig.custom`) and generate tokens:

```swift
import SwiftLLM

// 1. Initialize TransformerDecoder with Llama-3-8B configuration
let config = LLMConfig.llama3_8B
let decoder = TransformerDecoder(config: config)

// 2. Load model weights into memory
let weightsDictionary: [String: [Double]] = [:] // Populated from SafeTensors/GGUF
try decoder.loadWeights(weightsDictionary)

// 3. Configure sampling strategy
let sampler = TopKSampler(temperature: 0.7, topK: 40, topP: 0.9)

// 4. Autoregressive token generation loop
let promptTokens: [Int] = [128000, 791, 7453, 374] // Tokenized input
var generatedTokens = promptTokens
let maxNewTokens = 50

print("Generating output tokens...")
for _ in 0..<maxNewTokens {
    let logits = try decoder.forward(tokens: generatedTokens)
    let nextToken = sampler.sample(logits: logits)
    
    generatedTokens.append(nextToken)
    if nextToken == 128001 { // End of Sequence token (EOS)
        break
    }
}

print("Generated sequence length: \(generatedTokens.count) tokens.")
```

---

## 4. Structured JSON Grammar Decoding

Enforce strict output conformance to Swift `Codable` schemas using `JSONGrammarDecoder`:

```swift
import SwiftLLM

struct SentimentResponse: Codable {
    let label: String
    let confidence: Double
}

let grammarDecoder = JSONGrammarDecoder(schema: SentimentResponse.self)
let constrainedTokens = try grammarDecoder.constrainSampling(logits: logits, validTokens: vocabulary)
```
