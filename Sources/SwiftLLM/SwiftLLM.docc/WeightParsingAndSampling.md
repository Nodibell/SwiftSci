# GGUF/SafeTensors Parsing & Sampling

Parse GGUF and SafeTensors model weights zero-copy and sample tokens using Temperature and Top-K sampling.

## Overview

Load weights straight into unified memory and stream token generation.

### 1. Token Sampling

```swift
import SwiftLLM

let sampler = Sampler(temperature: 0.7, topK: 50)
let nextToken = sampler.sample(logits: logits.last!)
print("Sampled Token ID: \(nextToken)")
```
