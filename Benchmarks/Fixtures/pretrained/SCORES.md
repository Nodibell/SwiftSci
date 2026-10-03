# Diagnose Llama token-choice differences

The [cross-runtime comparison](LLAMA-CPP.md) can capture scores at the first differing content token while preserving the preceding token sequence. This diagnostic distinguishes changes in token ranking from tokenization or prompt differences. It does not declare either runtime a numerical oracle.

## Run

Use the verified checkpoint, GGUF and conversion report from the [comparison setup](LLAMA-CPP.md#convert-and-verify). Add `--scores` and choose a fresh output directory:

```sh
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/llama_cpp_compare.py \
  --model "$model" --gguf "$gguf" \
  --conversion-report "$results/conversion.json" \
  --python Benchmarks/.venv-llm/bin/python \
  --server /opt/homebrew/bin/llama-server \
  --scores --output "$results/score-comparison"
```

The ordinary comparison runs first. llama.cpp additionally returns its top 20 pre-sampling log probabilities at every generated position. These values must repeat exactly across the three runs at each selected probe position. Timing fields from this instrumented run must not be compared with runs that omit probability collection.

## Hold the token history fixed

For each prompt, [the analysis controller](../../Tools/llama_score_analysis.py) selects the first generated token and the first differing content token where both runtimes have generated a token. If the content sequences and stopping reasons agree, it selects the final content token as a control. Positions are zero-based indices into generated tokens, excluding the prompt.

A disagreement where one runtime stops or reaches its limit while the other continues is recorded in `unsupported_cases`. The diagnostic does not substitute an earlier matching token for that decision. It also records empty-content cases as unsupported. If no supported cases remain, `score-analysis.json` has status `unsupported` and zero probes; no score workers run. The ordinary generation comparison still records the differing outputs and stop reasons.

The [MLX score worker](../../Python/llama_score_worker.py) reuses the recorded prompt IDs and forces the shared continuation one token at a time. It preserves MLX-LM's normal prefill/decode partition. A logits processor records the model output and returns it unchanged. A sampler records MLX's native log probabilities before selecting the requested continuation token. At the final probe position it uses ordinary greedy selection.

The controller requires the instrumented BF16 token choices to match the uninstrumented MLX baseline at every probe. It also checks that the model consumed the exact requested prefix. This prevents comparison of scores conditioned on different text.

## Separate the precision questions

The worker runs in separate processes for BF16, both separate-matrix and shared-matrix Float32 output projections, and full Float32. The [mixed-precision experiment](PRECISION.md) checks whether widening only the final matrix multiplication is sufficient. The Float32 run promotes the existing BF16 checkpoint values exactly; it does not recover precision lost when the original checkpoint was created. Model-body, output-score and KV-cache dtypes must match the requested policy. The mixed mode requires BF16 body parameters and cache with Float32 output scores. The checkpoint files remain unchanged.

Each probe records:

- Raw MLX logits and the native MLX log probabilities for every vocabulary token.
- The model-output dtype and cache dtype.
- The winning token before and after MLX normalization, plus the number of tokens tied for the highest raw score.
- Float32 scores rounded back to BF16, their winning token and tie count.
- llama.cpp's top 20 pre-sampling log probabilities and token IDs.

Raw logits can differ by an additive constant without changing predictions. The report therefore compares the margin between the two competing tokens, and separately compares normalized log probabilities. MLX normalization for this cross-runtime comparison uses NumPy Float64 over the full vocabulary. The original native MLX values are also retained.

The llama.cpp API exposes log probabilities, not raw logits. Their pairwise difference represents the corresponding logit margin, subject to the API's Float32 softmax/log rounding. RMSE and maximum difference against llama.cpp apply only to its top 20 candidates. The report must not present them as full-vocabulary error measures. Full-vocabulary centered-logit differences are available only between the two MLX precision runs.

## Read the artifacts

The ordinary comparison files remain present. Score collection adds:

| Artifact | Purpose |
| --- | --- |
| `score-request.json` | Exact prompt IDs, forced continuation and selected positions. |
| `scores-bf16/scores.json` | BF16 model/cache precision, token choices, ties and artifact identities. |
| `scores-head-float32/scores.json` | The same observations with a BF16 body/cache and a Float32 output projection. |
| `scores-shared-float32/scores.json` | The same observations with one shared Float32 embedding/output matrix, BF16 gathered input rows and a BF16 body/cache. |
| `scores-float32/scores.json` | The same observations for Float32 computation. |
| `scores-*/scores.npz` | Full raw logits and native log probabilities as Float32 arrays, loaded without pickle. BF16 values promote exactly. |
| `score-analysis.json` | Competing-token margins, normalization effects, rounding sensitivity and candidate-only error measures. |
| `scores-*.log` | Worker diagnostics. |

A completed score analysis means the supported measurements were collected and their consistency checks passed. Any unsupported cases remain listed separately. It is not a conformance pass. The parent comparison retains `completed-with-differences` when the ordinary BF16 generations differ, even if the Float32 probes agree.

## Observations from the pinned checkpoint

On 2026-09-29, both first divergences occurred at exact ties in MLX's BF16 raw logits. Normalization did not change the winning token in these probes.

| Prompt and generated position | Competing tokens | BF16 scores | Float32 MLX preference | llama.cpp preference |
| --- | --- | --- | --- | --- |
| Scientific explanation, 29 | `,` and ` with` | Both 21.875. | ` with`, margin 0.039171. | ` with`, margin 0.051674. |
| Structured observations, 12 | ` the` and ` only` | Both 27.0. | ` only`, margin 0.045614. | ` only`, margin 0.032124. |

Rounding the Float32 output scores back to BF16 recreated both top-score ties and the original BF16 choices. It also preserved the choices at the other six control probes. This establishes that final-score rounding alone is sufficient to reproduce these two ties from the measured Float32 scores; it does not isolate every source of forward-pass error.

The Float32 model chose the llama.cpp token at all eight selected probes. This supports precision-sensitive token ranking as an explanation for these two examples. It does not establish equivalence across all logits, contexts, models or generated sequences. Float32 score differences remain, and promoting the whole model changes several computations at once.
