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

### RoPE (Rotary Position Embedding) with Dynamic Offset
Applies 2D rotation to Query and Key projections based on position index $m$:
$$Q_{\text{rot}} = \text{RoPE}(Q, m), \quad K_{\text{rot}} = \text{RoPE}(K, m)$$
During incremental autoregressive decoding, `RoPEEmbedding` dynamically incorporates `positionOffset` (corresponding to the number of prior cached tokens), ensuring correct relative attention geometry across arbitrary context lengths without learned position embeddings.

### Two-Stage KV-Cache Generation (Prefill + Decode)
Generation executes in two distinct stages:
1. **Prefill Pass**: Evaluates the entire prompt sequence $[0 ..< N]$ in parallel, writing Key and Value projections into `KVCache`.
2. **Incremental Decode Steps**: Processes only the single newest token $[N+1]$ at each step. The new Query vector performs dot-product attention against all cached Key and Value vectors ($Q_{N+1} \cdot K_{\text{accumulated}}^T$), eliminating redundant $O(N^2)$ prompt recomputation.

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

// 2. Parse GGUF Quantized Binary Archive (Zero-Copy)
let ggufURL = URL(fileURLWithPath: "llama-3-8b.Q4_K_M.gguf")
let ggufModel = try GGUFParser.parse(fileURL: ggufURL)
print("Loaded GGUF Model with \(ggufModel.tensors.count) quantized tensors.")
```

---

## 3. End-to-End Decoder Configuration & Two-Stage Inference

Construct a `TransformerDecoder` and perform two-stage cached generation:

```swift
import SwiftLLM

// 1. Initialize TransformerDecoder with Llama-3-8B configuration
let config = LLMConfig.llama3_8B
let decoder = TransformerDecoder(config: config)
let cache = KVCache(config: config)

let promptTokens: [Int] = [128000, 791, 7453, 374] // Tokenized input
var generatedTokens = promptTokens
let maxNewTokens = 50

// 2. Stage 1: Prefill prompt tokens into KV-cache
_ = try decoder.forward(tokens: promptTokens, cache: cache, isPrefill: true)

// 3. Configure sampling options
let sampling = SamplingConfiguration(temperature: 0.7, topK: 40, topP: 0.9, repetitionPenalty: 1.1)

// 4. Stage 2: Incremental single-token decoding loop
var currentToken = promptTokens.last!
for _ in 0..<maxNewTokens {
    let logits = try decoder.forward(tokens: [currentToken], cache: cache, isPrefill: false)
    let nextToken = Sampler.sample(logits: logits, config: sampling, pastTokens: generatedTokens)
    
    generatedTokens.append(nextToken)
    currentToken = nextToken
    if nextToken == 128001 || nextToken == 128009 { // EOS tokens
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
