# ``SwiftLLM``

Local GPU Large Language Model architecture, weight parsers, and sampling algorithms.

## Overview

`SwiftLLM` enables local inference of causal language models on Apple Silicon GPU.

### Architecture Features
- **Transformer Decoder**: Causal Multi-Head Attention, RoPE (Rotary Position Embeddings), SwiGLU.
- **Weight Parsers**: Zero-copy `GGUFParser` and `SafeTensorsParser`.
- **Sampling Strategies**: Greedy argmax, Temperature, Top-K, Top-P sampling.

## Topics

### Large Language Models
- ``TransformerDecoder``
- ``BPETokenizer``
