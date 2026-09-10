# Native Metal MSL Shaders for Quantized LLMs

Hardware dequantization and SIMD-group matrix multiplication for 4-bit and 8-bit quantized weights on Apple Silicon GPU.

## Overview

Large language models like Llama 3, Mistral, and DeepSeek require massive memory bandwidth. Weight-only quantization (such as GGUF Q4_0 and Q8_0) compresses 16-bit floating-point weights into 4-bit or 8-bit integer blocks with group scaling factors, reducing memory footprint by 50% to 75%.

`SwiftLLM` implements native Metal Shading Language (MSL) SIMD kernels executing hardware dequantization and GEMV / GEMM transformations directly on Apple Silicon GPUs without intermediate CPU memory allocations.

## Architecture

```
Packed Quantized Weights [Q4_0 / Q8_0]   Scale Factors [FP16]   Input Activations [FP32]
                 │                               │                        │
                 └───────────────────────┬───────┴────────────────────────┘
                                         ▼
                   Metal Compute Command Encoder
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     MSL Kernel: gemv_q4_0                           MSL Kernel: gemv_q8_0
   - Unpack 4-bit nibbles (w - 8)                  - Load signed 8-bit integers
   - Multiply by group FP16 scale                  - Multiply by group FP16 scale
   - Accumulate with input vector                  - Accumulate with input vector
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                            Output Activation Vector
                               (Unified Memory)
```

## SIMD-Group Matrix Multiplication

On Apple Silicon (M-series GPUs), threads within a 32-thread SIMD-group collaborate on 8x8 matrix tiles using `metal_simdgroup_matrix`. This eliminates thread divergence and ensures maximal utilization of the GPU arithmetic units:

```metal
#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

kernel void gemm_simdgroup_q4_0(
    device const half* A               [[buffer(0)]],
    device const half* B               [[buffer(1)]],
    device half* C                     [[buffer(2)]],
    constant uint& M                   [[buffer(3)]],
    constant uint& N                   [[buffer(4)]],
    constant uint& K                   [[buffer(5)]],
    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]
) {
    simdgroup_matrix<half, 8, 8> acc;
    acc = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
    // Accumulate across inner K dimension...
}
```

## Swift API Integration

`QuantizedLinear` exposes both MLX tensor inference and direct Metal kernel execution:

```swift
import SwiftLLM

// 1. Create a quantized linear layer (e.g. 4096 in, 4096 out)
let linear = QuantizedLinear(
    inFeatures: 4096,
    outFeatures: 4096,
    scheme: .q4_0,
    weight: mlxWeights,
    scales: mlxScales
)

// 2. Direct Metal shader execution without CPU round-trips
let outputs = try linear.forwardMetal(
    inVector: inputActivations,
    rawWeights: packedQ4Bytes,
    rawScales: fp16Scales
)
```

## Topics

### Quantization & Metal Execution
- ``QuantizedLinear``
- ``QuantizationScheme``
- ``MetalQuantizedEngine``
- ``MetalQuantizedError``
