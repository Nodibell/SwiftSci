# Decoder shape and context conformance

This original MIT-licensed fixture pack extends the decoder comparison to wider layers and longer contexts. The frozen [four-token pack](../neural/README.md) retains its inputs, answers and source lock.

| Hidden width | Heads | Head width | Intermediate width | Vocabulary | Tokens | Batch | Positions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 2 | 8 | 24 | 11 | 8 | 2 | Learned |
| 32 | 4 | 8 | 48 | 17 | 32 | 1 | Rotary |
| 32 | 2 | 16 | 48 | 17 | 128 | 1 | Rotary |

Each architecture has CPU and GPU cases, each with full-sequence and cached execution. `neural-shaped-conformance` contains all 12 cases; `neural-shaped-cpu-conformance` contains the six CPU cases. A cache run consumes half the sequence as a prefix, then one token per step. It records actual cumulative cache lengths. Every timed call creates a fresh model and cache.

## Mathematical contract

Each model has one pre-normalized decoder layer, RMS epsilon 2^-10, bias-free attention and SwiGLU projections, and an independent output head. Rotary positions use base 10000 and split-half rotation. The learned-position tensor exists in the complete parameter pack even when rotary positions leave it unused.

The generator uses exact dyadic Float32 weights. Projection, embedding and position values have magnitude at most 1/16; normalization scales range from 3/4 to 9/8. This bounded family keeps the initial expansion numerically controlled. It is not representative of all trained weight distributions.

The scalar mpmath oracle evaluates complete causal logits at 80 decimal digits. Cached answers come from full-sequence mathematics, independently of the runtime cache algorithm. The tolerance is 2e-5 absolute plus 2e-5 relative, fixed before execution. This is a contract for these bounded inputs, not a general error bound for arbitrary networks. Tests reject altered logits, dimensions and cache counts and check causality and repeated execution.

The Python and Swift input validators bound vocabulary to 32, hidden width to 32, heads to 4, intermediate width to 64, sequence capacity to 128 and batch to 2. Head width must be even. They reject incomplete weights, non-Float32 values, unsupported loading modes and invalid cache schedules. The checked profiles cover the table above, not every permitted configuration.

## Reproduction and interpretation

From the repository root, regenerate with:

```sh
Benchmarks/.venv-standardized/bin/python Benchmarks/Fixtures/neural-shaped/generate.py
```

The source lock pins the generator, shape validator, scalar oracle and reference requirements. Manifests pin every input and answer. The [MLX comparison guide](../../MLX-COMPARISONS.md) gives installation and execution commands; substitute `neural-shaped-conformance` or its CPU profile.

Input decoding is outside timing. Model construction, parameter creation and readback, forward evaluation, stream synchronization and complete output extraction are inside. Swift initializes the public model before replacing its weights; the Python comparators construct fixed arrays. These end-to-end measurements cannot isolate language overhead or establish steady-state tokens per second.

Python MLX and Swift share a backend family. NumPy is a CPU mathematical comparison even for GPU-requested fixtures. Independent scalar answers remain the numerical reference. Whole-process RSS is not GPU allocation accounting.

Deeper networks, multi-token continuation after a populated cache, contexts beyond 128 tokens, prepared-model throughput, pretrained checkpoints, quantization, tokenizer behavior and generation stopping remain outside this contract.
