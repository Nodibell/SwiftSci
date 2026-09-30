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

### Grouped-query attention

`LLMConfig.numHeads` is the query-head count. `numKVHeads` sets the key/value head count. Its default, `nil`, follows the current query-head count and preserves ordinary multi-head attention. Setting it to one selects multi-query attention. Other positive divisors select grouped-query attention.

```swift
let config = LLMConfig(
	vocabSize: 128_256,
	numLayers: 16,
	hiddenDim: 2048,
	numHeads: 32,
	numKVHeads: 8,
	intermediateSize: 8192
)
```

The head dimension is `hiddenDim / numHeads`. Query and output projections retain the hidden width. Key and value projections each have shape `[numKVHeads * headDimension, hiddenDim]`. For the example above, K/V weights are `[512, 2048]` and cached tensors are `[batch, 8, sequence, 64]`.

The cache retains eight KV heads. It does not expand them to 32 query heads. Those cache tensors use one quarter of the elements of the corresponding full-head cache. This describes tensor storage, not total process memory or a measured speedup. MLX's native scaled-dot-product attention consumes the grouped layout directly.

The public `TransformerBlock.attention` property remains a `MultiHeadAttention`. Its direct-call path also handles grouped heads, and existing projection parameter paths remain unchanged. Configuration construction and model construction reject nonpositive head counts and nondivisible head layouts with preconditions.

The `llama1B` and `llama8B` presets now use eight KV heads. Their K/V parameter shapes therefore differ from earlier approximate presets. To retain the old full-head layout, set `numKVHeads` to `nil` before constructing the model. These presets still do not establish full checkpoint compatibility; rotary scaling and tied output embeddings require separate validation.

### RoPE (Rotary Position Embedding) with Dynamic Offset
Applies 2D rotation to Query and Key projections based on position index $m$:
$$Q_{\text{rot}} = \text{RoPE}(Q, m), \quad K_{\text{rot}} = \text{RoPE}(K, m)$$
During incremental autoregressive decoding, `RoPEEmbedding` dynamically incorporates `positionOffset` (corresponding to the number of prior cached tokens), ensuring correct relative attention geometry across arbitrary context lengths without learned position embeddings.

### Llama wavelength-dependent rotary scaling

Llama 3.1 and 3.2 checkpoints can specify `rope_type: "llama3"`. Their scaling leaves short wavelengths unchanged, slows long wavelengths by the configured factor, and blends the intermediate band. A different base or a uniform position scale cannot express this rule.

```swift
let encoding = PositionalEncodingScheme.llama3RoPE(
    base: 500_000,
    scaling: Llama3RoPEScaling(
        factor: 32,
        lowFrequencyFactor: 1,
        highFrequencyFactor: 4,
        originalContextLength: 8192
    )
)
```

These values match the Llama 3.2 1B checkpoint. Supply the values from your own checkpoint for other models. `LLMConfig.llama1B` now selects this encoding. The original Llama 3 8B preset retains ordinary RoPE.

`RoPEEmbedding` precomputes one frequency denominator per rotated pair and passes that array to MLX's native rotary kernel. These constants are excluded from model weights. The kernel supports CPU and Apple GPU execution, partial rotary dimensions, both pair layouts, and position offsets. `scale` remains a uniform position multiplier in addition to the optional wavelength scaling.

Existing `.rope(base:)` calls and unscaled `RoPEEmbedding` construction retain their behavior. Exhaustive switches over `PositionalEncodingScheme` must handle the new `.llama3RoPE` case. The Llama 3.2 preset now produces different rotations by design. Its default sequence allocation remains 8192; matching rotary parameters alone does not establish full checkpoint or long-context generation compatibility.

The frequency rule follows [Meta's Llama reference](https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py) with explicit parameters from the checkpoint. Tests compare rotations with an independent Double calculation, including long offsets. Their error budget accounts for Float32 phase rounding before evaluating sine and cosine.

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

### Tied input and output embeddings

Some checkpoints, including Llama 3.2 1B, set `tie_word_embeddings` to `true`. They use the token embedding matrix for both input lookup and output logits, so the checkpoint can omit `lm_head.weight`.

Set `LLMConfig.tieWordEmbeddings` to `true` to select this layout. The `llama1B` preset enables it. Other presets and custom configurations retain the default `false`.

A tied decoder has no independent output module or output-weight parameter. It calls the current embedding's `asLinear` operation after final normalization. Checkpoint reloads, parameter updates, embedding replacement, and quantized embedding replacement therefore affect both uses. Training accumulates input and output contributions into the single embedding parameter.

`loadWeights` requires `model.embed_tokens.weight` in tied mode and does not report a missing `lm_head.weight`. If both keys are supplied, the embedding is authoritative and the separate output tensor is ignored. An output tensor alone does not substitute for missing embeddings. Untied models continue to require `lm_head.weight`.

`TransformerDecoder.lmHead` is now `Linear?`. Callers that access the independent head directly must unwrap it. A `nil` head selects projection through the embedding; it does not disable output logits. This represents the absence of a separate module and avoids keeping an unused vocabulary-sized matrix merely to preserve the old property type.

```swift
if let independentHead = decoder.lmHead {
    // Inspect or use the independent projection in an untied model.
    print(independentHead.shape)
}
```

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
