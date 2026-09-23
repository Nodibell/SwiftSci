# Autoregressive Transformer Decoder Inference

Execute two-stage autoregressive generation with full prompt prefill and incremental single-token decoding using Apple Silicon KV-caching.

## Overview

`SwiftLLM` implements modern decoder-only causal transformer architectures (such as Llama 3, Mistral, and Gemma) leveraging MLX unified memory acceleration on Apple Silicon GPUs. Inference is divided into two distinct computational phases:

1. **Prompt Prefill Phase (`isPrefill: true`)**: Processes all initial prompt tokens in parallel, computing attention matrices and populating the key-value cache across all transformer layers.
2. **Incremental Decoding Phase (`isPrefill: false`)**: Processes a single token per step, computing query projections and attending over the accumulated key-value history in $O(1)$ step time.

---

## 1. Two-Stage Generation Loop

The following complete example demonstrates configuring a model, prefilling the KV-cache, and executing an autoregressive generation loop:

```swift
import Foundation
import SwiftLLM
import MLX

// 1. Initialize configuration and decoder
let config = LLMConfig.llama3_8B
let decoder = TransformerDecoder(config: config)
let cache = KVCache(config: config)

// Tokenized prompt sequence
let promptTokens: [Int] = [128000, 791, 7453, 374]
var generatedTokens = promptTokens
let maxNewTokens = 64

// 2. Stage 1: Prefill prompt tokens into KV-cache
let promptLogits = try decoder.forward(
    tokens: promptTokens,
    cache: cache,
    isPrefill: true
)

// 3. Configure sampling parameters
let sampling = SamplingConfiguration(
    temperature: 0.7,
    topK: 40,
    topP: 0.9,
    repetitionPenalty: 1.1
)

// 4. Sample first generated token from prefill logits
var currentToken = Sampler.sample(
    logits: promptLogits,
    config: sampling,
    pastTokens: generatedTokens
)
generatedTokens.append(currentToken)

// 5. Stage 2: Incremental single-token decoding loop
for _ in 1..<maxNewTokens {
    // Forward single token with isPrefill: false
    let stepLogits = try decoder.forward(
        tokens: [currentToken],
        cache: cache,
        isPrefill: false
    )
    
    // Sample next token ID
    let nextToken = Sampler.sample(
        logits: stepLogits,
        config: sampling,
        pastTokens: generatedTokens
    )
    
    generatedTokens.append(nextToken)
    currentToken = nextToken
    
    // Check for EOS (End of Sequence) tokens
    if nextToken == 128001 || nextToken == 128009 {
        break
    }
}

print("Generated total sequence: \(generatedTokens.count) tokens.")
```

---

## 2. Rotary Position Embeddings (RoPE) During Incremental Decoding

During single-token decoding (`isPrefill: false`), position frequencies must match the actual sequential position in the generated sequence rather than index `0`. 

`RoPEEmbedding` tracks this via the `positionOffset` parameter:

```swift
// Internal RoPE position calculation:
// position = positionOffset + tokenIndex
let rotatedQ = rope.apply(x: query, positionOffset: cache.currentLength)
```

Because `KVCache` dynamically tracks `currentLength`, positions increment faithfully from the end of the prefill boundary through every subsequent token generation step.

---

## 3. High-Level Streaming API

For streaming applications, `LLMModel` wraps the decoder and cache lifecycle into an asynchronous sequence:

```swift
import SwiftLLM

let options = LLMOptions(
    maxTokens: 128,
    sampling: SamplingConfiguration(temperature: 0.7, topK: 40, topP: 0.9)
)

for try await token in model.generateStream(prompt: "Explain rotary embeddings:", options: options) {
    print(token, terminator: "")
}
```

---

## Topics

### Core Inference Components
- ``TransformerDecoder``
- ``KVCache``
- ``PagedKVCache``
- ``LLMConfig``
- ``RoPEEmbedding``

### Sampling & Decoding
- ``Sampler``
- ``SamplingConfiguration``
- ``LLMOptions``
- <doc:TransformerDecoderArchitecture>
- <doc:SamplingAndDecodingStrategies>
