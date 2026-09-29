# Measure the cost of a Float32 output projection

The [score diagnostic](SCORES.md) found two next-token decisions where BF16 output scores tied. Full Float32 computation separated the candidates and agreed with llama.cpp. This experiment asks whether widening only the final output projection preserves that distinction while retaining a BF16 transformer and KV cache.

This is benchmark-only inference code. It changes neither SwiftSci production code nor the installed MLX-LM package.

## Three precision policies

[The adapter](../../Python/llama_precision.py) prepares these policies from the same verified checkpoint:

| Policy | Transformer and input embeddings | KV cache | Output projection | Retained parameters for this checkpoint |
| --- | --- | --- | --- | ---: |
| `bf16` | BF16 | BF16 | BF16 | 2,471,628,800 bytes |
| `head-float32` | BF16 | BF16 | Float32 | 3,522,301,952 bytes |
| `float32` | Float32 | Float32 | Float32 | 4,943,257,600 bytes |

The mixed policy passes the original BF16 body's final hidden state to a Float32 matrix multiplication. It promotes both that hidden state and the output weights before multiplication. Promoting already-rounded output scores would not recover the lost distinction.

This Llama checkpoint ties its input embeddings to its output weights. The experimental adapter retains the original BF16 embedding and an exact Float32 copy for output projection. That copy adds 1,050,673,152 bytes. It is prepared once and retained across requests. The table counts parameter-array payloads, not total process memory or allocator reservations.

The adapter requires a tied-embedding Llama model. It is an inference experiment: the copied projection does not track later training updates to the embedding. It is not a new general-purpose model API.

## Run the score check first

Run the [comparison with `--scores`](SCORES.md#run). All three policies must finish, consuming the same recorded prompt and continuation. The output directory must contain `score-request.json` and completed reports under `scores-bf16`, `scores-head-float32` and `scores-float32`.

`score-analysis.json` reports each policy's selected tokens, competing-token margins and retained parameter memory. The original BF16 cross-runtime generation status stays visible even when a wider policy agrees.

## Measure equivalent work

Run from the repository root with the pinned MLX-LM environment and verified checkpoint:

```sh
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/llama_precision_benchmark.py \
  --model "$HOME/Library/Caches/SwiftSci/models/Llama-3.2-1B-Instruct-bf16-863c846a" \
  --score-run Benchmarks/Runs/llama-cpp-01/score-comparison \
  --python Benchmarks/.venv-llm/bin/python \
  --output Benchmarks/Runs/llama-precision-timing-01
```

[The timing controller](../../Tools/llama_precision_benchmark.py) starts one fresh process per policy per block. It runs two blocks with policy order reversed in the second block. The second block also reverses the prompt order. Each process runs two warmups and five measured repetitions per prompt, yielding ten measurements per policy and prompt.

[The worker](../../Python/llama_precision_timing_worker.py) forces the same continuation and leaves only the last token to greedy selection. All policies therefore process the same prompt IDs, continuation IDs and number of decode steps. The complete returned token sequence must match that policy's earlier score probe. Measurements exclude model loading and projection preparation, which are recorded separately.

Each elapsed sample includes cache creation, MLX-LM's normal prefill/decode path, sampling, detokenization and synchronization through host-visible output. It does not retain the full logits for analysis during timing. Cache dtype and repeated token sequences are checked. This is the same measurement boundary for all three policies.

`precision-timing.json` retains every sample, per-block medians, combined medians, elapsed-time ratios to BF16, source/input identities and environment details. Larger ratios mean more time. Raw worker files and logs remain beside it.

## Interpret memory separately

The worker records retained parameter bytes, active MLX device memory after preparation, peak active device memory during each request, cached allocator memory and process peak RSS. Those quantities measure different things and must not be added together. Process peak RSS includes initialization and host-side work; the active-device figures come from MLX's allocator. The parameter byte count is the clearest measure of the extra retained output matrix.

## Limits

The workload covers four short prompts through at most 32 generated positions. It measures a controlled, teacher-forced trajectory, not complete free-running answers or language quality. Two reversed-order blocks reduce ordering bias but do not establish thermal control or a production performance baseline. Long contexts, larger models, batching and memory pressure require separate sweeps.

The wider projection uses a full retained Float32 matrix. Alternative kernels, chunked projections or wider accumulation with narrower stored weights are not evaluated here. The measurements must not be generalized to those designs.

## Observed result on 2026-09-29

On the pinned 1B checkpoint and Apple M4 Max, widening only the output projection selected the llama.cpp token at all eight shared-prefix probes. Both previously tied decisions became distinct. The transformer and cache remained BF16. This does not certify a complete free-running generation or language quality.

All 188 controller tests passed, including the optional Metal checks. A small controlled test also demonstrated a BF16 score tie that the wider projection separates, while preserving the body's original weights.

| Prompt | BF16 median ms | Mixed median ms | Mixed / BF16 | Full Float32 median ms | Float32 / BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| scientific-explanation | 261.88 | 267.60 | 1.022 | 405.33 | 1.548 |
| structured-observations | 141.46 | 152.77 | 1.080 | 241.28 | 1.706 |
| swift-code | 237.36 | 279.69 | 1.178 | 463.55 | 1.953 |
| unicode-whitespace | 126.23 | 138.07 | 1.094 | 197.86 | 1.567 |

Each median combines ten measured repetitions across two fresh-process blocks. The mixed policy took 2% to 18% longer in these combined medians; full Float32 took 55% to 95% longer. Block variation limits those estimates. For example, the scientific prompt's BF16 block medians were 232.95 and 275.23 ms, while the mixed policy measured 266.25 and 274.40 ms. The combined 2% difference for that prompt should not be treated as a precise overhead estimate.

The measured extra output-weight payload was 1.05 GB. This establishes a candidate precision boundary and its current storage cost. It does not show that retaining a second full matrix is the best memory layout.
