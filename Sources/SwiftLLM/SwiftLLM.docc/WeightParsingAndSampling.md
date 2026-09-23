# GGUF/SafeTensors Parsing & Sampling

Parse GGUF and SafeTensors model weights zero-copy into Apple Silicon unified memory and sample tokens with fine-grained logits controls.

## Overview

`SwiftLLM` provides high-throughput binary weight parsers that map model weights directly into unified memory (UMA) without intermediate memory allocations, paired with a robust token sampling pipeline operating strictly on unnormalized logits.

---

## 1. GGUF Zero-Copy Weight Parsing

The `GGUFParser` decodes GGUF header metadata and extracts tensor blocks into semantic `QuantizedTensor` representations without eager dequantization into full floating-point arrays:

```swift
import Foundation
import SwiftLLM

let modelURL = URL(fileURLWithPath: "models/llama-3-8b-instruct.Q4_0.gguf")

// 1. Parse GGUF weights into QuantizedTensor map
let weights: [String: QuantizedTensor] = try GGUFParser.parse(url: modelURL)

// 2. Inspect parsed tensor metadata
if let attnWeight = weights["blk.0.attn_q.weight"] {
    print("Tensor name: \(attnWeight.name)")
    print("Shape: \(attnWeight.shape)")
    print("Scheme: \(String(describing: attnWeight.scheme))") // .q4_0
    print("Block Size: \(attnWeight.layoutMetadata.blockSize)") // 32
}

// 3. Extract separated packed weight and scale buffers for Metal GEMV
if let (packedWeights, scales) = weights["blk.0.attn_q.weight"]?.extractSeparatedBuffers() {
    print("Extracted \(packedWeights.count) weight bytes and \(scales.count) scale bytes.")
}
```

---

## 2. SafeTensors Zero-Copy Weight Parsing

The `SafeTensorsParser` parses JSON header descriptors and maps FP16/FP32/BF16 tensor payloads directly into `MLXArray` instances:

```swift
import Foundation
import SwiftLLM
import MLX

let safeURL = URL(fileURLWithPath: "models/model.safetensors")

// Parse SafeTensors into native MLX tensor dictionary
let tensors: [String: MLXArray] = try SafeTensorsParser.parse(url: safeURL)

for (tensorName, array) in tensors {
    print("Loaded tensor '\(tensorName)': shape=\(array.shape), dtype=\(array.dtype)")
}
```

---

## 3. Token Sampling with SamplingConfiguration

After computing next-token logits from `TransformerDecoder.forward`, token selection is governed by `SamplingConfiguration`:

```swift
import SwiftLLM
import MLX

// 1. Define generation parameters
let sampling = SamplingConfiguration(
    temperature: 0.7,
    topK: 40,
    topP: 0.9,
    repetitionPenalty: 1.1
)

let pastTokens = [128000, 791, 7453, 374]
let nextLogits: MLXArray = ... // Shape [vocab_size] from decoder.forward

// 2. Sample next token ID applying penalty, temperature, top-k, and top-p
let nextToken = Sampler.sample(
    logits: nextLogits,
    config: sampling,
    pastTokens: pastTokens
)

print("Sampled next token ID: \(nextToken)")
```

> [!TIP]
> For detailed mathematical formulations of repetition penalty, temperature scaling, top-k truncation, and nucleus (top-p) masking, see <doc:SamplingAndDecodingStrategies>.

---

## Topics

### Weight Parsers & Representations
- ``GGUFParser``
- ``SafeTensorsParser``
- ``QuantizedTensor``
- ``QuantizedLayoutMetadata``

### Sampling & Decoding
- ``Sampler``
- ``SamplingConfiguration``
- ``LLMOptions``
- <doc:SamplingAndDecodingStrategies>
