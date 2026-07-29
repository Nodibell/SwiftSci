# ``SwiftLLM``

Local Causal Transformer Decoder & Weight Parsers.

## Overview

`SwiftLLM` implements local transformer decoder architectures, MLX GPU integration, and zero-copy model weight parsers.

### Key Capabilities

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
