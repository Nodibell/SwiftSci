# Fixed decoder conformance

This original tensor pack uses the repository MIT license. It tests complete logits from SwiftLLM's public TransformerDecoder with fixed weights. It downloads no models and makes no claim about language-model quality, training, tokenization or generation.

`neural-conformance` contains twelve cases. Both explicitly requested MLX devices run learned-position and rotary-position models using full-sequence and cached execution. A zero-projection case on each device checks the residual and final normalization against a simple analytic answer. Two single-batch GPU RoPE cases distinguish batch-size behavior from token content. `neural-loader-conformance` contains two additional cases through the public `loadWeights` method. It is a diagnostic profile: loader failures must remain visible and cannot produce a passing certificate.

`neural-cpu-conformance` contains the five CPU direct-parameter cases for ordinary macOS CI. Run the complete `neural-conformance` profile on an Apple silicon host with usable Metal GPU access. Do not report unavailable GPU execution as a pass.

## Frozen architecture and inputs

The decoder has vocabulary size 7, hidden width 8, two attention heads, intermediate width 6, one layer, maximum sequence length 8 and RMS normalization epsilon 2^-10. Attention and feed-forward projections have no biases. The output head is independent of the embedding matrix. RoPE uses base 10000, scale 1 and split-half rotation within each four-component head.

Inputs contain all 13 parameter arrays, including the unused learned-position array in RoPE mode. All weights are finite, exactly representable Float32 values with magnitude at most 2. Token IDs are bounded JSON integers in rectangular batches. Validators reject extra parameters, missing weights, wrong shapes, non-Float32 values and invalid execution settings before calling MLX.

Direct cases use public `Module.update` with actual parameter names. Public-loader cases translate the same input tensors to the documented loader names and call `TransformerDecoder.loadWeights`. Both paths must verify every actual parameter name, shape, dtype and value after loading. Missing replacements cannot hide behind random initialization.

## Independent answers

Regenerate with `Benchmarks/.venv-standardized/bin/python Benchmarks/Fixtures/neural/generate.py`. The source lock pins the tensor generator, scalar oracle and reference dependency requirements. Each manifest pins that lock, the input and the reference. The scalar oracle uses mpmath at 80 decimal digits on exact input values. It independently evaluates RMS normalization, projections, causal softmax attention, RoPE, residual additions, SwiGLU and final logits. Cached expected logits come from the full causal computation, so the oracle does not duplicate the cache algorithm.

The runtime comparator behind the `pandas` engine uses NumPy Float32 on CPU. It implements real incremental K/V caches for cached cases. It does not execute MLX or verify Swift's loader. Its CPU result is a mathematical comparison for either requested Swift device, not evidence of GPU execution and not a matched-backend speed comparison.

All logits are checked with absolute and relative tolerances of 2e-5. This budget applies to the bounded Float32 architecture and input range above. It allows accumulated rounding in projection, normalization, transcendental and attention kernels while remaining far below the deliberately changed outputs used by negative controls. It is not a universal error bound for deeper networks or lower precision. Integer output dimensions and cache counts retain their exact intended values.

## Cache, device and timing rules

Cached execution processes a two-token prefix and then two single tokens. It records cache lengths 2, 3 and 4, concatenates complete logits and checks them against the independent full-sequence answers. Every timed invocation creates a fresh model and cache. Multi-token continuation after a populated cache is outside this profile and needs a separate contract.

Swift scopes MLX to the requested CPU or GPU. The adapter has no automatic fallback. It evaluates outputs, synchronizes the selected device's stream, checks Float32 dtype and shape, and extracts every result before stopping the timer. This records requested MLX placement; it does not claim a hardware counter trace of each individual kernel.

Decoding happens outside timing. Model construction, tensor creation, parameter loading/readback, forward execution, synchronization and output extraction are inside. Construction includes initialization that the fixed weights replace. These small end-to-end timings are diagnostic and are unsuitable as a steady-state inference or formal performance baseline.

The pack covers one decoder architecture and four-token sequences. It does not establish pretrained checkpoint compatibility, quantized inference, long-context behavior, compiled generation, training correctness, or all SwiftVision and SwiftLLM APIs. Keep public-loader diagnostic failures separate from direct-parameter arithmetic results.
