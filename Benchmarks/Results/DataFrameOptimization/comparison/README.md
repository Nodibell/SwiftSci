# Reproduce the dataframe comparison

The recorded run uses SwiftSci `4c5bb953547ba5d84c7fd86bf868b1680c5f1811` and Kiraa Swift Pandas `fc5d957401f70e9d3b6b07270f7f300d488fd39f`. The Swift workers and `python_bench.py` are unchanged from that run. Only runner paths and metadata capture have been adapted for distribution. `generate-fixtures.py` recreates and checks the original CSV bytes.

Use an Apple silicon Mac with Xcode. Clone Kiraa into a directory named `kiraa-swift-pandas`, check out the recorded revision, and set its absolute path:

```sh
export KIRAA_REPO=/absolute/path/to/kiraa-swift-pandas
cd Benchmarks/Results/DataFrameOptimization/comparison
python3 -m venv .venv
.venv/bin/python -m pip install -r ../production/requirements.txt
.venv/bin/python generate-fixtures.py
```

The requirements file records the original environment. Changes to Python/library/toolchain versions should be reported as a new environment. The normal comparison checks that both library checkouts have no tracked modifications and records their current commits.

Build both workers with the source implementation, outside Documents if file-provider metadata causes signing failures:

```sh
export SWIFTPANDAS_USE_BINARY=0
export BENCH_BUILD_DIR="$HOME/Library/Caches/SwiftSci/dataframe-comparison"
swift build -c release --arch arm64 -j 8 \
  --scratch-path "$BENCH_BUILD_DIR" --force-resolved-versions
export BENCH_BIN_DIR="$(swift build -c release --arch arm64 \
  --scratch-path "$BENCH_BUILD_DIR" --show-bin-path)"
.venv/bin/python run.py
```

The original binary build used the local Xcode build environment. The commands above are the portable SwiftPM route, not a claim that compiler-driver differences are irrelevant. The runner records executable and compiler hashes for every new run. Do not compare stale binaries against a different recorded source commit.

`run.py` uses three rotating serial batches, two warmups and five timed samples per worker. It writes results beneath `results/` and exits 2 if any comparison fails. At the recorded Kiraa revision, native integer filtering is a known parity failure and must remain excluded from performance rankings. Exit 2 is not a successful all-library parity result.

For a short smoke check, use `BENCH_SIZES=100000 BENCH_REPS=1 BENCH_BATCHES=1`. This is insufficient for a performance claim. `BENCH_OPS` accepts a comma-separated subset. All dataframe outputs are compared after timing; grouped results are aligned by keys.

## Full SwiftSci tests

From the SwiftSci repository root, run Debug and Release sequentially:

```sh
xcodebuild test -scheme SwiftSci-Package -configuration Debug \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$HOME/Library/Caches/SwiftSci/full-tests" \
  -onlyUsePackageVersionsFromResolvedFile \
  -parallel-testing-enabled NO ENABLE_TESTABILITY=YES

xcodebuild test -scheme SwiftSci-Package -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$HOME/Library/Caches/SwiftSci/full-tests" \
  -onlyUsePackageVersionsFromResolvedFile \
  -parallel-testing-enabled NO ENABLE_TESTABILITY=YES
```

The recorded runs used an invocation-scoped package-plugin validation exception after reviewing pinned MLX revision `0bb916c67f4b9e5c682cbe02a42c701c93ab5021`. Review and approve that package locally if Xcode requests it; do not disable global package validation. The native runner was needed to locate MLX's Metal test resources.

The committed test summaries describe the tested implementation commit. Evidence-only additions do not establish new test results for a later production change or upstream merge.
