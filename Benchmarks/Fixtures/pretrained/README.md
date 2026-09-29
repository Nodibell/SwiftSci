# Real-checkpoint integration diagnostics

This opt-in pack tests Llama 3.2 1B Instruct through a pinned Python MLX-LM reference and SwiftSci's public safetensors parser, decoder configuration and BPE tokenizer. It records integration blockers without changing production code. It does not issue a workload conformance certificate.

## Pinned inputs

[The checkpoint manifest](llama-3.2-1b-instruct-bf16.json) identifies the public [MLX-community BF16 conversion](https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-bf16/tree/863c846a9ac6fad4e49e1743d52984dff262e953), its immutable revision and the size and SHA-256 of every downloaded file. Its upstream is [Meta Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct). Model weights retain the upstream Llama license and acceptable-use terms. They are cached outside this repository and are not redistributed in this branch.

[The prompts](prompts.json) are original MIT-licensed test inputs for scientific explanation, structured observations, Swift code and Unicode/whitespace handling. They exercise integration behavior. They are not a language-quality benchmark, and generated answers are not graded for task correctness.

## Install and download

Use the [standard benchmark environment](../../README.md#run-locally) for the controller and a separate environment for pretrained-model dependencies. The recorded package set was tested with Python 3.14 on Apple silicon. Python MLX 0.31.1 matches the existing comparator's core version; MLX-LM 0.31.2 accepts that version.

```sh
python3 -m venv Benchmarks/.venv-llm
Benchmarks/.venv-llm/bin/python -m pip install -r Benchmarks/Python/requirements-llm.txt
Benchmarks/.venv-llm/bin/python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
snapshot_download(
    'mlx-community/Llama-3.2-1B-Instruct-bf16',
    revision='863c846a9ac6fad4e49e1743d52984dff262e953',
    local_dir=str(Path.home() / 'Library/Caches/SwiftSci/models/Llama-3.2-1B-Instruct-bf16-863c846a'),
    allow_patterns=['*.json', '*.safetensors', 'README.md'],
)
PY
```

The downloaded pack occupies about 2.48 GB. The integration controller verifies it against the manifest before use. No custom model code is trusted or downloaded for execution. See [MLX-LM's documentation](https://github.com/ml-explore/mlx-lm) for its loader and generation APIs.

The separately installed [llama.cpp runtime](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md) requires GGUF weights. This pack uses safetensors and does not execute llama.cpp. A same-source GGUF conversion and conversion-parity check must precede its inclusion in comparisons. Installing its executable alone does not validate model compatibility.

## Run the integration diagnostic

Build after staging new source files, as the build tool copies tracked files to an isolated source snapshot. Choose a fresh result directory.

```sh
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py build
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/pretrained_llama.py \
  --model "$HOME/Library/Caches/SwiftSci/models/Llama-3.2-1B-Instruct-bf16-863c846a" \
  --python Benchmarks/.venv-llm/bin/python \
  --swift-worker "$HOME/Library/Caches/SwiftSci/standardized-benchmarks/derived/Build/Products/Release/SwiftSciBenchmarkWorker" \
  --output Benchmarks/Runs/llama-integration-01
```

Metal is required for the reference run; there is no CPU fallback. The controller checks that the Swift binary matches the source fingerprint. This is opt-in work outside ordinary CPU acceptance and CI; CI does not download the model.

## Interpret the results

`reference.json` records package versions, cold model-load time, checkpoint-tokenizer outputs, chat-template inputs and three greedy generation runs per prompt. The first run per prompt is warmup. Generation stops at EOS or 32 tokens. Token sequences must repeat across runs. A passing smoke result means the reference executed and repeated consistently; it does not establish response quality or numerical agreement with SwiftSci.

`swift.json` records actual parser results, checkpoint/public-model projection shapes and tokenization comparisons. The inspection model uses eight positions to bound the unused learned-position allocation; projection shapes retain the checkpoint's layer widths. The inspector does not install incompatible weights or attempt generation when those checks identify a mismatch.

`integration.json` records checkpoint, prompt, source and binary identities. `blocked` exits with status 1 and retains findings. `failed` indicates a diagnostic execution error. `preflight-passed` still does not mean generation passed: the diagnostic intentionally reports `generation_comparison: not-executed`. Reports must preserve this distinction.

A complete comparison still needs successful public checkpoint loading, matching tokenizer and special-token behavior, correct rotary positions and attention layout, numerical logit checks, generation stopping and cancellation checks, and separately measured preparation, prefill and decoding. Each production defect should be repaired on its own branch with regression coverage; this branch retains the evidence.

## Observed compatibility gaps

The pinned checkpoint run on 2026-09-29 completed all four reference prompts with identical greedy token sequences across three repetitions. SwiftSci parsed 146 BF16 tensors. Native generation remained blocked by these checks:

| Finding | Reproducible evidence | Required follow-up validation |
| --- | --- | --- |
| Grouped-query attention | The checkpoint has 32 query heads and 8 KV heads. Its first key projection is `[512, 2048]`; the public decoder constructs `[2048, 2048]`. | Correct projection shapes, KV-head grouping and cached/full logit agreement. |
| Rotary scaling | The checkpoint specifies `rope_type: llama3`, factor 32, frequency factors 1 and 4, and original context 8192. The public positional configuration exposes only the base. | Checkpoint frequency rules and position-dependent logits at short and extended offsets. |
| Tied output embeddings | `tie_word_embeddings` is true and `lm_head.weight` is absent. The public model has a separate output head and its loader expects that key. | Preserve the shared embedding/output semantics and verify complete parameter loading. |
| Tokenizer parity | All four raw-text cases disagree with the checkpoint tokenizer. Expected/actual token counts are 18/34, 23/42, 18/39 and 19/22. | Checkpoint pre-tokenization, vocabulary/merge behavior, whitespace, Unicode and special tokens. |

The parameter and configuration checks inspect the public [decoder](../../../Sources/SwiftLLM/Core/TransformerDecoder.swift). Token comparisons execute the public [BPETokenizer](../../../Sources/SwiftNLP/Core/BPETokenizer.swift). These are blockers for this checkpoint; they do not invalidate the smaller supported decoder contracts. No production repair or substitute model implementation is included here.
