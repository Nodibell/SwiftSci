# ``SwiftLLM``

Local Causal Transformer Decoder & Weight Parsers.

## Overview

`SwiftLLM` implements local transformer decoder architectures, MLX GPU integration, and zero-copy model weight parsers.

### Key Capabilities

- **Quantized Metal GPU Execution**: 4-bit / 8-bit quantized linear layers (`QuantizedLinear`) supporting `q4_0`, `q4_1`, `q8_0`, `awq4`, and `awq8` formats with on-the-fly dequantization.
- **Constrained JSON Grammar Decoding**: Token-level state machine (`JSONGrammarDecoder`) enforcing strict `Codable` JSON schema adherence via logit masking.
- **Dynamic Paged KV-Cache**: Unified memory block allocator (`PagedKVCache`) managing physical pages (`pageSize: 16`) to prevent memory fragmentation.
- **Decoder Architecture**: Causal self-attention, Rotary Position Embeddings (RoPE), and SwiGLU activation functions.
- **Weight Parsers**: Zero-copy `GGUFParser` and `SafeTensorsParser` weight loaders.
- **Token Sampler**: Temperature scaling, Top-K, and greedy argmax `Sampler`.
- **GPU Execution**: UMA unified memory tensor acceleration via MLX integration.

### Example Usage

```swift
import SwiftLLM

let config = LLMConfig(vocabSize: 32000, hiddenDim: 4096)
let model = LLMModel(config: config)
let logits = try await model.forward(tokens: inputTokens)
```

## Topics

### Guides & Tutorials
- <doc:TransformerDecoderArchitecture>
- <doc:WeightParsingAndSampling>
