# Longer prompts and free-running Llama batches

This opt-in suite extends the [precision experiment](PRECISION.md) beyond fixed continuations. It runs the pinned Llama 3.2 1B checkpoint through four MLX-LM precision policies, using actual greedy generation and actual batches. It records differences and task failures without certifying native SwiftSci or changing production code.

## Inputs and checked behavior

The [suite specification](generation-suite.json) fixes the generation limits, repetitions, prefill chunk size and batch membership. The [case builder](../../Tools/llama_generation_cases.py) combines the four original [smoke prompts](prompts.json) with deterministic sensor tables containing 16, 128 and 512 records. The chat template uses the fixed date from the smoke fixture.

Each generated table asks for its middle record as a JSON object with fields explicitly named `sensor` and `value`. The builder computes the expected sensor ID and integer value directly from the table's generation rule. The task check requires exactly those two fields, a correct integer value and valid JSON without markdown. This checks record retrieval and output format. It is not a general language-quality benchmark. The original four smoke prompts remain ungraded. Sensor task checks count repeated executions, not independent questions; the same three tables also appear in batch groups.

The suite includes seven single-request cases and three batch groups:

| Batch | Members | Purpose |
| --- | --- | --- |
| `short-four` | All four original smoke prompts | Four concurrent requests with different outputs and stopping points |
| `mixed-two` | Scientific explanation and 128-record retrieval | Unequal prompt lengths and completion lengths |
| `long-two` | 16-record and 512-record retrieval | Larger padding and cache-length differences |

Token counts are measured with the pinned checkpoint tokenizer and saved with the exact prompt IDs. The suite rejects prompts above 16,384 tokens. Each request stops at EOS or the explicit limit of 256 new tokens. A length-limited answer remains visible as `finish_reason: length`; it is not treated as a completed answer.

## Run

Use the environments and verified checkpoint from the [installation guide](README.md#install-and-download). From the repository root:

```sh
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/llama_generation_benchmark.py \
  --model "$HOME/Library/Caches/SwiftSci/models/Llama-3.2-1B-Instruct-bf16-863c846a" \
  --python Benchmarks/.venv-llm/bin/python \
  --output Benchmarks/Runs/llama-generation-01
```

Choose a fresh output directory. Metal is required. The controller verifies checkpoint files before and after the run and records source, fixture and environment identities. It starts a fresh process for each policy. The worker runs three repetitions per single case and batch, with the first repetition marked as warmup. Every repetition uses a fresh KV cache. No prompt cache is reused between requests.

The worker uses MLX-LM's public `stream_generate` and `BatchGenerator` paths. Batches are simultaneous model requests, not a loop over single-request generation. Completed batch caches and single-request caches must have the policy's expected dtype. The batch generator closes even if a request fails.

## Read the results

`generation.json` contains the combined report. The per-policy JSON files retain all prompt tokens, generated tokens, decoded outputs, stop reasons, task checks, timings and memory readings. Logs and a dependency freeze sit beside them.

The summary keeps three questions separate:

- Does each policy repeat its own token sequence and stopping reason across three runs?
- Does batching change a case's output compared with running it alone under the same policy?
- Do the separate and shared FP32 matrix policies agree for complete bounded generations?

A task check can fail even when two policies agree perfectly. A batch can differ from a single request because execution shape changes floating-point calculations. Such differences remain observations that need investigation. The controller rejects incomplete reports and changed prompt identities; it does not turn output agreement into numerical conformance.

## Timing and memory boundaries

Single-request records include end-to-end elapsed time, time to first token, and MLX-LM's prompt and generation throughput counters. Batch records include aggregate elapsed time, per-request time to first token, and the batch API's separate prompt/generation time and token counters. The library defines those counters differently across its single and batch paths. Preserve their labels when reporting them.

Free-running policies may produce different answers and different token counts. Their total elapsed times therefore do not measure equivalent work. This suite is a functional diagnostic with timing observations, not a production performance baseline. Use the earlier fixed-trajectory experiment for controlled precision-cost comparisons.

Model loading and precision preparation are recorded separately. Peak active device memory resets before each repetition. The retained parameter payload and active device memory after preparation are also recorded. These measures overlap and must not be added together.

## Remaining boundaries

This suite exercises one 1B BF16 checkpoint, one pinned MLX-LM environment, greedy sampling and batches of at most four requests. It does not test concurrent clients, cancellation, recovery after memory exhaustion, training, stochastic sampling or long-context language quality. It does not rerun llama.cpp on the expanded prompts. The earlier [llama.cpp comparison](LLAMA-CPP.md) remains separate evidence.

Native SwiftSci Llama generation still requires the tokenizer, grouped-query attention, rotary-scaling and tied-embedding repairs recorded in the [integration findings](README.md#observed-compatibility-gaps).

## Observed result on 2026-09-30

The clarified-prompt run completed 180 bounded generations across four policies. Each policy ran seven single cases and eight requests across three batch groups, repeated three times. The sensor prompts contained 234, 1,355 and 5,199 tokens after chat formatting.

| Policy | Repeatable trajectories | Batch cases equal to single | Strict sensor checks passed |
| --- | ---: | ---: | ---: |
| `bf16` | 15/15 | 4/8 | 0/18 |
| `head-float32` | 15/15 | 5/8 | 0/18 |
| `shared-float32` | 15/15 | 5/8 | 0/18 |
| `float32` | 15/15 | 8/8 | 0/18 |

All 15 separate/shared FP32 trajectory pairs agreed in every repetition, including the batches and stopping reasons. The comparison covers complete returned token sequences through EOS or the configured token limit. It does not establish full-vocabulary score equality at these longer contexts.

The task failures matter. All four policies added markdown fences despite the instruction to return JSON alone. Their single-request answers contained the correct sensor/value pair for the 16- and 128-record tables, but the 512-record answer returned `603` for `S0256`, whose actual value was `402`. Full Float32 did not correct that error. The strict checks remain failed; the model's repeated answers are not proof of task correctness. Eighteen checks per policy count repetitions and batch appearances of three tables, not eighteen independent questions.

Batching changed some outputs under BF16 and mixed precision. Full Float32 matched its single-request outputs for all eight batch appearances in this run. This suggests a useful follow-up precision investigation, but token agreement alone does not isolate the cause of the lower-precision differences.

Some answers reached the 256-token limit. The mixed policies reached it on the single structured-observation prompt, full Float32 on the single Swift-code prompt, and all policies on the batched Swift-code prompt. Those are incomplete answers, retained as `length`, rather than successful EOS completion.

### Long-prompt timing observations

These figures describe the 5,199-token single request on the Apple M4 Max. Throughputs are medians of two measured repetitions after one warmup. Peak active device memory uses the larger measured sample. GB is decimal.

| Policy | Library prompt tokens/s | Library generation tokens/s | Peak active device GB |
| --- | ---: | ---: | ---: |
| `bf16` | 6327 | 167.4 | 3.172 |
| `head-float32` | 6302 | 142.9 | 4.223 |
| `shared-float32` | 6339 | 145.7 | 3.697 |
| `float32` | 5314 | 88.7 | 6.344 |

The shared matrix saved 525,336,576 bytes of peak active device memory relative to the separate matrix for this request. The two mixed policies had similar throughput in this small sample. These are diagnostic observations from one fixed policy order, not a controlled performance baseline or a guarantee about other prompt lengths.

The source-tree SHA-256 was `62b835c13e01b9b1f60760e8bcfad8251611c7449c882d778539928c2bc8c764`. Raw results remain locally in `Benchmarks/Runs/local-llama-generation-20260930-explicit-schema`. An earlier exploratory run remains in `Benchmarks/Runs/local-llama-generation-20260930`; its prompt did not explicitly name the two JSON fields. The clarified run changed that wording and retained the same strict task criteria. No failure was waived.

Validation: all 195 benchmark-controller tests passed with optional Metal checks enabled, using an isolated snapshot of tracked files.
