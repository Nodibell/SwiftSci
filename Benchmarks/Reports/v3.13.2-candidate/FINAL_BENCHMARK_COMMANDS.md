# Final Reproducible Benchmark & Validation Commands (`v3.13.2-candidate`)

All Stage A commands below were executed on `Apple M4 Pro` (`macOS 27.2.1 / 26B131`, AC power) using uninstrumented Release binaries and separated Python environments. Run from repository root:
`/Users/oleksiichumak/Developer/Xcode.projects/SwiftSci/SwiftSci`.

---

## 1. Build Uninstrumented Release Benchmark Worker

```bash
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py build
```

Built binary path used in Stage A runs:
`/Users/oleksiichumak/Library/Caches/SwiftSci/standardized-benchmarks/derived/Build/Products/Release/SwiftSciBenchmarkWorker`

---

## 2. Standardized Multi-Process Benchmark Sweeps & Audits (Stage A Executed)

```bash
OUT_ROOT="Benchmarks/Runs/v3.13.2-candidate"
WORKER="/Users/oleksiichumak/Library/Caches/SwiftSci/standardized-benchmarks/derived/Build/Products/Release/SwiftSciBenchmarkWorker"
PY_STD="Benchmarks/.venv-standardized/bin/python"

"$PY_STD" Benchmarks/Tools/bench.py run --profile standard --engines swiftsci,pandas --mode legacy-capped --swift-worker "$WORKER" --python "$PY_STD" --output "$OUT_ROOT/standard"
"$PY_STD" Benchmarks/Tools/bench.py run --profile standard --engines swiftsci,pandas --mode production-default --swift-worker "$WORKER" --python "$PY_STD" --output "$OUT_ROOT/standard-production-default"
"$PY_STD" Benchmarks/Tools/bench.py run --profile nist --engines swiftsci,pandas --mode legacy-capped --swift-worker "$WORKER" --python "$PY_STD" --output "$OUT_ROOT/nist"
"$PY_STD" Benchmarks/Tools/bench.py run --profile public-data --engines swiftsci,pandas,polars,duckdb --mode legacy-capped --swift-worker "$WORKER" --python "$PY_STD" --output "$OUT_ROOT/public-data"
"$PY_STD" Benchmarks/Tools/bench.py run --profile migration --engines swiftsci,pandas --mode legacy-capped --swift-worker "$WORKER" --python "$PY_STD" --output "$OUT_ROOT/migration"
"$PY_STD" Benchmarks/Tools/bench.py run --profile tabular-migration --engines swiftsci,pandas,polars,duckdb --mode legacy-capped --swift-worker "$WORKER" --python "$PY_STD" --output "$OUT_ROOT/tabular-migration"

for sweep in standard standard-production-default nist public-data migration tabular-migration; do
  "$PY_STD" Benchmarks/Tools/bench.py audit "$OUT_ROOT/$sweep"
  "$PY_STD" Benchmarks/Tools/bench.py report "$OUT_ROOT/$sweep" > "$OUT_ROOT/$sweep.report.json"
done
```

---

## 3. CPU Acceptance Tier & Auxiliary Conformance Controls (Stage A Executed)

```bash
"$PY_STD" Benchmarks/Tools/acceptance.py \
  --tier cpu \
  --check-baseline \
  --swift-worker "$WORKER" \
  --python "$PY_STD" \
  --output "$OUT_ROOT/acceptance-cpu"

"$PY_STD" Benchmarks/Tests/check_pretrained_inspection.py \
  --swift-worker "$WORKER" \
  --output "$OUT_ROOT/pretrained-inspection"

"$PY_STD" Benchmarks/Tests/check_public_workflow_workers.py \
  --run "$OUT_ROOT/acceptance-cpu/scientific-workflow" \
  --output "$OUT_ROOT/workflow-worker-controls"
```

---

## 4. Accuracy-Gated Scientific Validation Suite (`39` Workloads + `3` Rolling-Origin Backtests + G-037)

```bash
PY_SCI="/Users/oleksiichumak/.pyenv/versions/3.11.9/bin/python3.11"

"$PY_STD" Benchmarks/Tools/scientific_validation_suite.py \
  --swift-worker "$WORKER" \
  --python-engine "$PY_SCI" \
  --processes 5 \
  --warmups 2 \
  --samples 5 \
  --output "$OUT_ROOT/scientific_validation.json"
```

---

## 5. Inventory, Python Unit Tests, Swift Release Tests, DocC & Coverage

```bash
"$PY_STD" Benchmarks/Tools/bench.py inventory
"$PY_STD" -m unittest discover -s Benchmarks/Tests -v

swift test -c release

swift package generate-documentation \
  --target SwiftDataFrame \
  --target SwiftStats \
  --target SwiftPreprocessing \
  --target SwiftML \
  --target SwiftCluster \
  --target SwiftNLP \
  --target SwiftOptimize \
  --target SwiftForecast \
  --target SwiftLLM \
  --target SwiftExplain \
  --target SwiftVisualization \
  --target SwiftVision \
  --target SwiftDatabase \
  --target SwiftAgent \
  --warnings-as-errors

python3 scripts/verify_doc_coverage.py
```

---

## 6. Stage B clean-tree release verification

Run after authorization from the clean, locally tagged release commit on Apple Silicon with AC power. Use a fresh output directory and a fresh worker build cache.

```bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
test -z "$(git status --porcelain)"
export RELEASE_SHA="$(git rev-parse 'v3.13.2^{commit}')"
test "$(git rev-parse HEAD)" = "$RELEASE_SHA"
pmset -g batt | grep -q 'AC Power'

PY_STD=Benchmarks/.venv-standardized/bin/python
PY_SCI="$HOME/.pyenv/versions/3.11.9/bin/python3.11"
export OUT_ROOT=Benchmarks/Runs/v3.13.2
test ! -e "$OUT_ROOT"
mkdir -p "$OUT_ROOT"
BUILD_CACHE="$(mktemp -d "${TMPDIR:-/tmp}/swiftsci-v3.13.2.XXXXXX")"
WORKER="$BUILD_CACHE/derived/Build/Products/Release/SwiftSciBenchmarkWorker"
"$PY_STD" Benchmarks/Tools/bench.py build --cache "$BUILD_CACHE" | tee "$OUT_ROOT/build.log"
test -x "$WORKER"

run_sweep() {
  local profile="$1" name="$2" engines="$3" mode="$4"
  "$PY_STD" Benchmarks/Tools/bench.py prepare --profile "$profile" > "$OUT_ROOT/$name.prepare.log"
  "$PY_STD" Benchmarks/Tools/bench.py run --profile "$profile" --engines "$engines" \
    --mode "$mode" --swift-worker "$WORKER" --python "$PY_STD" --output "$OUT_ROOT/$name"
  "$PY_STD" Benchmarks/Tools/bench.py audit "$OUT_ROOT/$name" > "$OUT_ROOT/$name.audit.log"
  "$PY_STD" Benchmarks/Tools/bench.py report "$OUT_ROOT/$name" > "$OUT_ROOT/$name.report.json"
}
run_sweep standard standard swiftsci,pandas legacy-capped
run_sweep standard standard-production-default swiftsci,pandas production-default
run_sweep nist nist swiftsci,pandas legacy-capped
run_sweep public-data public-data swiftsci,pandas,polars,duckdb legacy-capped
run_sweep migration migration swiftsci,pandas legacy-capped
run_sweep tabular-migration tabular-migration swiftsci,pandas,polars,duckdb legacy-capped

"$PY_STD" Benchmarks/Tools/acceptance.py --tier cpu --check-baseline \
  --swift-worker "$WORKER" --python "$PY_STD" --output "$OUT_ROOT/acceptance-cpu"
"$PY_STD" Benchmarks/Tests/check_pretrained_inspection.py \
  --swift-worker "$WORKER" --output "$OUT_ROOT/pretrained-inspection"
"$PY_STD" Benchmarks/Tests/check_public_workflow_workers.py \
  --run "$OUT_ROOT/acceptance-cpu/scientific-workflow" \
  --output "$OUT_ROOT/workflow-worker-controls"
"$PY_STD" Benchmarks/Tools/scientific_validation_suite.py \
  --swift-worker "$WORKER" --python-engine "$PY_SCI" --processes 5 --warmups 2 --samples 5 \
  --output "$OUT_ROOT/scientific_validation.json"

"$PY_STD" Benchmarks/Tools/bench.py inventory > "$OUT_ROOT/inventory.json"
"$PY_STD" -m unittest discover -s Benchmarks/Tests -v 2>&1 | tee "$OUT_ROOT/python-tests.log"
swift test -c release 2>&1 | tee "$OUT_ROOT/swift-release-tests.log"
swift package generate-documentation \
  --target SwiftDataFrame --target SwiftStats --target SwiftPreprocessing \
  --target SwiftML --target SwiftCluster --target SwiftNLP --target SwiftOptimize \
  --target SwiftForecast --target SwiftLLM --target SwiftExplain --target SwiftVisualization \
  --target SwiftVision --target SwiftDatabase --target SwiftAgent --warnings-as-errors \
  2>&1 | tee "$OUT_ROOT/docc.log"
"$PY_STD" scripts/verify_doc_coverage.py | tee "$OUT_ROOT/doc-coverage.log"

"$PY_STD" - <<'PYTHON'
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
release_sha = os.environ["RELEASE_SHA"]

def read(relative):
    return json.loads((root / relative).read_text())

def verify_source(source):
    assert source["commit"] == release_sha
    assert source["dirty"] is False

sweeps = ("standard", "standard-production-default", "nist", "public-data",
          "migration", "tabular-migration")
for sweep in sweeps:
    cert = read(f"{sweep}/certificate.json")
    assert cert["status"] == "passed"
    verify_source(cert["source"])

acceptance = read("acceptance-cpu/acceptance.json")
policy = read("acceptance-cpu/regression-policy.json")
verify_source(acceptance["source"])
verify_source(policy["source"])
assert acceptance["coverage_complete"] and len(acceptance["profiles"]) == 16
assert policy["status"] == "passed" and policy["blocking"] == []
assert len(policy["known_failures"]) == 4
assert all(item["engine"] == "pandas" and item["classification"] == "comparator-accuracy"
           for item in policy["known_failures"])

scientific = read("scientific_validation.json")
verify_source(scientific["source_identity"])
summary = scientific["summary"]
assert summary["total_workloads"] == summary["passed_workloads"] == 39
assert summary["expect_error_verified"] == 2 and summary["all_gates_passed"]
assert len(scientific["rolling_origin_backtests"]) == 3 and summary["rolling_origin_passed"]
fitted_value_checks = [
    w["gate"]["metrics"]["g037"] for w in scientific["workloads"]
    if w["gate"]["metrics"].get("g037", {}).get("applicable")
]
rolling_checks = [
    origin["g037"] for backtest in scientific["rolling_origin_backtests"].values()
    for origin in backtest["origins"] if origin.get("g037", {}).get("applicable")
]
assert len(fitted_value_checks) == 9 and len(rolling_checks) == 6
assert all(check["passed"] for check in fitted_value_checks + rolling_checks)
assert scientific["power_state"]["source"] == "AC"
PYTHON

pmset -g batt | grep -q 'AC Power'
test -z "$(git status --porcelain)"
```

Preserve Stage A artifacts with their original hashes and labels. The scientific runner currently writes a fixed candidate label in its raw JSON. Use the verified clean commit, source fingerprint, and gate results to identify Stage B evidence. Prepare final release reports from the Stage B outputs outside the measured commit and attach them when publication is authorized.
