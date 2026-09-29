# Compare Llama through MLX-LM and llama.cpp

This optional diagnostic runs the [pinned Llama checkpoint](README.md#pinned-inputs) through two reference runtimes on Apple silicon. It verifies the GGUF conversion before comparing tokenization and greedy generation. SwiftSci's real-checkpoint generation remains blocked by the [documented compatibility gaps](README.md#observed-compatibility-gaps).

The comparison produces integration evidence. It does not issue a numerical certificate, grade answer quality or establish a production performance baseline.

## What is controlled

- Both runtimes use the same 146 checkpoint tensors. The independent [conversion verifier](../../Tools/verify_llama_gguf.py) checks every value, reverses GGUF's query/key row permutation, and checks lossless BF16-to-Float32 promotion of normalization weights.
- The verifier checks model dimensions, head counts, rotary base, normalization epsilon, chat template and token metadata. It independently calculates the additional rotary-frequency table with a relative tolerance of `1e-6` for Float32 rounding. Unknown tensors and layouts fail verification.
- [The prompt pack](prompts.json) specifies the template date. The controller verifies raw-text and fully rendered chat token IDs before passing the same token arrays to llama.cpp. It does not let the server insert another template or BOS token.
- Each runtime uses BF16 KV caches. MLX reports the actual cache dtypes. The llama.cpp log must confirm Metal, all 17 offloaded layers, and BF16 keys and values. CPU fallback fails this diagnostic.
- Each prompt runs three times per engine, with the first marked as warmup. Generation is greedy and limited to 32 tokens. Each request starts a fresh cache; cached prompt reuse and context truncation fail the run.
- EOS handling differs between the APIs. Reports retain the original responses and compare content token IDs after removing one terminal EOS token. Stop reasons must also agree before a generation counts as an exact match.

Equal weights and input tokens do not imply bitwise-equal computations across backends. Intermediate precision, kernels and reduction order can differ. Token disagreement is evidence to investigate; this diagnostic does not assign its cause.

## Prepare the converter

First install the model and MLX-LM environment from the [integration guide](README.md#install-and-download). Install llama.cpp separately using its [official installation instructions](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/docs/install.md). This comparison pins version `0.5.0`, build `11146`, commit `7fe450e19`; the controller rejects another runtime version.

[The toolchain manifest](llama-cpp-toolchain.json) records the full converter revision and downloaded archive hash. [The conversion environment](../../Python/requirements-gguf.txt) is separate from the MLX-LM and standard benchmark environments. On the tested Python 3.14 installation, NumPy 2.2.6 built from source successfully.

Run from the repository root:

```sh
set -e
tools="$HOME/Library/Caches/SwiftSci/tools"
mkdir -p "$tools"
curl --fail --location \
  https://api.github.com/repos/ggml-org/llama.cpp/tarball/7fe450e19305b828c199d602c23a8337aaa1f03b \
  --output "$tools/llama-cpp-7fe450e19.tar.gz"
python3 - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('Benchmarks/Fixtures/pretrained/llama-cpp-toolchain.json').read_text())
archive = Path.home() / 'Library/Caches/SwiftSci/tools/llama-cpp-7fe450e19.tar.gz'
with archive.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
if digest != manifest['archive_sha256'] or archive.stat().st_size != manifest['archive_bytes']:
    raise SystemExit('Converter archive differs from recorded input')
PY
mkdir -p "$tools/llama-cpp-7fe450e19"
tar -xzf "$tools/llama-cpp-7fe450e19.tar.gz" \
  -C "$tools/llama-cpp-7fe450e19" --strip-components=1
python3 -m venv "$tools/gguf-venv"
"$tools/gguf-venv/bin/python" -m pip install -r Benchmarks/Python/requirements-gguf.txt
```

The source archive provides the converter and matching `gguf-py` module. The [official converter](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/convert_hf_to_gguf.py) and [Llama conversion rules](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/conversion/llama.py) define the output format. The verifier does not call those transformation functions.

## Convert and verify

Choose new output paths. The BF16 GGUF occupies about 2.48 GB and stays outside Git. The original model license applies to the converted weights.

```sh
model="$HOME/Library/Caches/SwiftSci/models/Llama-3.2-1B-Instruct-bf16-863c846a"
gguf="$model.gguf"
results="Benchmarks/Runs/llama-cpp-01"
mkdir -p "$results"
"$tools/gguf-venv/bin/python" "$tools/llama-cpp-7fe450e19/convert_hf_to_gguf.py" \
  "$model" --outtype bf16 --outfile "$gguf"
PYTHONPATH="$tools/llama-cpp-7fe450e19/gguf-py" \
  "$tools/gguf-venv/bin/python" Benchmarks/Tools/verify_llama_gguf.py \
  --model "$model" --gguf "$gguf" --output "$results/conversion.json"
```

The verifier checks the source checkpoint manifest before and after reading it, and hashes the GGUF before and after verification. The comparison rejects changed weights, manifests or verifier code. The conversion report preserves each tensor's name, shape, stored type and exact-value result.

## Run the comparison

```sh
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/llama_cpp_compare.py \
  --model "$model" --gguf "$gguf" \
  --conversion-report "$results/conversion.json" \
  --python Benchmarks/.venv-llm/bin/python \
  --server /opt/homebrew/bin/llama-server \
  --output "$results/comparison"
```

[The controller](../../Tools/llama_cpp_compare.py) first runs MLX-LM, then starts a temporary llama.cpp server bound to `127.0.0.1`. It uses a generated API key, ignores system proxies for local requests, disables the web UI and network model downloads, and stops the server on success or failure. No persistent service is installed. Endpoint behavior follows the pinned [server API](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/tools/server/README.md).

The fresh output directory contains:

| File | Evidence |
| --- | --- |
| `comparison.json` | Source and artifact identities, hardware, server options, conversion report, token comparisons, full generation responses and observed differences. |
| `reference.json` | MLX-LM inputs, outputs, cache dtypes, timings and repeatability results. |
| `server.log` | Model loading, GPU offload, cache precision and request processing. |
| `reference.log` | Reference worker diagnostics. |
| `python-environment.txt` | Installed MLX-LM environment. |

`matched` means every tested token sequence and stop reason agreed. `completed-with-differences` means the integration checks passed but at least one cross-runtime generation differed. Both return zero because the diagnostic completed; neither means numerical certification. `failed` returns nonzero and preserves the error. Consumers must inspect the named result fields rather than treating process success as numerical conformance.

## Timing limits

These are short integration prompts with two measured repetitions per case. MLX's elapsed time includes cache creation, generation and host-visible readback. llama.cpp request time includes loopback HTTP; its server-provided prefill and decode timings have different boundaries from MLX-LM's rates. Server startup includes initialization and warmup, so it is not directly comparable to MLX model-load time. Verbose per-token debugging is disabled.

Keep the timing fields as diagnostics. Do not compute a general speedup from them, especially when generated token sequences differ. A performance baseline needs longer workloads, more repetitions, matched timing boundaries, controlled thermal conditions and teacher-forced logit comparisons on shared prefixes.

## Local validation on 2026-09-29

On an Apple M4 Max, the conversion preserved all 146 checkpoint tensors exactly. The rotary table's maximum relative difference from the independent calculation was `2.10e-7`, within the stated Float32 tolerance. A one-bit change to an embedding weight caused verification to fail.

All four raw-text and four chat-tokenization checks passed. Both runtimes repeated their own outputs across all three runs. With BF16 caches in both engines:

| Prompt | Cross-runtime content token agreement | Stop reason |
| --- | --- | --- |
| Scientific explanation | First 29 tokens agree, then diverge within 32 generated tokens. | Both reached the length limit. |
| Structured observations | First 12 tokens agree, then diverge within 32 generated tokens. | Both reached the length limit. |
| Swift code | All 32 tokens agree. | Both reached the length limit. |
| Unicode and whitespace | All 11 content tokens agree. | Both stopped at EOS. |

Matching generation does not establish task correctness. For example, both runtimes altered whitespace in the preservation prompt. The controller suite passed 180 tests, including optional Metal cases. No production SwiftSci implementation was changed to obtain these results.
