# Measure the cost of a Float32 output projection

The [score diagnostic](SCORES.md) found two next-token decisions where BF16 output scores tied. Full Float32 computation separated the candidates and agreed with llama.cpp. This experiment asks whether widening only the final output projection preserves that distinction while retaining a BF16 transformer and KV cache.

This is benchmark-only inference code. It changes neither SwiftSci production code nor the installed MLX-LM package.

## Four precision policies

[The adapter](../../Python/llama_precision.py) prepares these policies from the same verified checkpoint:

| Policy | Transformer / input embedding storage | KV cache | Output projection | Retained parameters for this checkpoint |
| --- | --- | --- | --- | ---: |
| `bf16` | BF16 | BF16 | BF16 | 2,471,628,800 bytes |
| `head-float32` | BF16 | BF16 | Float32 | 3,522,301,952 bytes |
| `shared-float32` | BF16 / Float32, gathered rows cast to BF16 | BF16 | Float32 | 2,996,965,376 bytes |
| `float32` | Float32 | Float32 | Float32 | 4,943,257,600 bytes |

Both mixed policies pass the original BF16 body's final hidden state to a Float32 matrix multiplication. They promote the final hidden state before multiplication and retain Float32 output weights. Promoting already-rounded output scores would not recover the lost distinction.

This Llama checkpoint ties its input embeddings to its output weights. The `head-float32` policy retains the original BF16 embedding and an exact Float32 copy for output projection. That copy adds 1,050,673,152 bytes. It is prepared once and retained across requests. The table counts parameter-array payloads, not total process memory or allocator reservations.

The `shared-float32` policy replaces the BF16 embedding with one exact Float32 matrix. Input lookup gathers rows from that matrix and casts only those rows to BF16 before the transformer. Output projection reads the same stored matrix in Float32. It removes the separate BF16 matrix, saving 525,336,576 bytes compared with `head-float32`. The transformer weights, activations and KV cache remain BF16. Parameter metadata reports transformer weights separately from embedding storage.

The adapter requires a tied-embedding Llama model. These are inference experiments. The copied projection in `head-float32` does not track later training updates to the embedding. The shared policy changes the embedding module inside its private model instance; it has not been validated for training. It is not a new general-purpose model API.

## Run the score check first

Run the [comparison with `--scores`](SCORES.md#run). All four policies must finish, consuming the same recorded prompt and continuation. The output directory must contain `score-request.json` and completed reports under `scores-bf16`, `scores-head-float32`, `scores-shared-float32` and `scores-float32`.

`score-analysis.json` reports each policy's selected tokens, competing-token margins and retained parameter memory. It also compares every vocabulary score between the separate and shared output-weight policies, recording exact equality and the maximum absolute difference at each probe. The original BF16 cross-runtime generation status stays visible even when a wider policy agrees.

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

Each elapsed sample includes cache creation, MLX-LM's normal prefill/decode path, sampling, detokenization and synchronization through host-visible output. It does not retain the full logits for analysis during timing. Cache dtype and repeated token sequences are checked. This is the same measurement boundary for all four policies.

`precision-timing.json` retains every sample, per-block medians, combined medians, elapsed-time ratios to BF16, source/input identities and environment details. Larger ratios mean more time. Raw worker files and logs remain beside it.

## Interpret memory separately

The worker records retained parameter bytes, active MLX device memory after preparation, peak active device memory during each request, cached allocator memory and process peak RSS. Those quantities measure different things and must not be added together. Process peak RSS includes initialization and host-side work; the active-device figures come from MLX's allocator. The parameter byte count is the clearest measure of the extra retained output matrix.

## Limits

The workload covers four short prompts through at most 32 generated positions. It measures a controlled, teacher-forced trajectory, not complete free-running answers or language quality. Two reversed-order blocks reduce ordering bias but do not establish thermal control or a production performance baseline. Long contexts, larger models, batching and memory pressure require separate sweeps.

Both wider-projection policies use a full retained Float32 matrix. Alternative kernels, chunked projections or wider accumulation with narrower stored weights are not evaluated here. The measurements must not be generalized to those designs.

## Earlier separate-matrix result on 2026-09-29

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

## Shared-matrix result on 2026-09-29

The shared matrix produced exactly the same 128,256 vocabulary scores as the separate-matrix policy at each of eight recorded positions. Both policies selected the llama.cpp token at all eight positions and separated the two previously tied decisions. This is evidence for the tested prefixes, not a general equivalence or accuracy certificate.

Retained parameter payload fell from 3,522,301,952 to 2,996,965,376 bytes, a saving of 525,336,576 bytes, or 14.9%. Measured active device memory after preparation also fell by exactly that amount in both timing blocks. Peak active device memory during these requests fell from about 3.61 to 3.63 GB to about 3.09 to 3.10 GB. Values use decimal GB. The shared policy still retains about 525 MB more parameter data than BF16.

| Prompt | BF16 ms | Separate FP32 head ms | Shared FP32 matrix ms | Full FP32 ms | Shared / separate |
| --- | ---: | ---: | ---: | ---: | ---: |
| scientific-explanation | 214.26 | 248.69 | 248.49 | 370.41 | 0.9992 |
| structured-observations | 127.31 | 142.23 | 142.68 | 200.49 | 1.0031 |
| swift-code | 226.16 | 260.59 | 260.92 | 391.35 | 1.0013 |
| unicode-whitespace | 117.24 | 129.79 | 130.10 | 179.41 | 1.0024 |

These medians combine ten measured repetitions per prompt and policy across two fresh-process blocks. Shared-matrix timings were within 0.4% of the separate-matrix timings in this run. The experiment does not establish a speed difference at that scale. Both mixed policies took about 11% to 16% longer than BF16. Timing comparisons use this run's BF16 baseline; the earlier table remains the record of the earlier experiment.

The source-tree SHA-256 for both the score and timing runs was `85a42cc6f83d62bdb45ff971855c19aa578319fb45344977843ce0ccf610dd7a`. Raw results are retained locally under `Benchmarks/Runs/local-llama-shared-precision-20260929` and `Benchmarks/Runs/local-llama-shared-timing-20260929`. The commands above reproduce the protocol with fresh output directories.

Longer prompts, complete free-running generation and batches remain untested for this policy. Native SwiftSci Llama generation remains blocked by the previously documented integration defects. These results validate an MLX-LM benchmark experiment.

Validation: all 189 benchmark-controller tests passed with the optional Metal checks enabled, using an isolated snapshot of the tracked files.
