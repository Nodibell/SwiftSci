# Version 1.5: DataFrame Engine Overhaul, Parallel I/O & Accelerate-Native Algorithms

## Background

Benchmark results from v1.4 reveal **9 informational gaps** where Python/pandas outperforms SwiftSci.
All 9 share a common root cause: single-threaded row-by-row processing, `String`-heap allocation
during parsing/filtering, and scalar loops where SIMD or Accelerate equivalents exist.

This plan targets **all 9 benchmark gaps** with surgical, algorithm-level fixes and introduces
three new high-value features. It also adds a **DocC documentation sprint** to close the 60%
undocumented public API gap discovered by the v1.4 audit.

---

## Benchmark Gap Analysis

| Benchmark | Swift | Python | Speedup | Root Cause |
|---|---|---|---|---|
| CSV Read (100k rows) | 71.5 ms | 18.8 ms | 0.26× | String materialisation per cell during type inference |
| CSV Stream Read (chunk=10k) | 276.8 ms | 21.7 ms | 0.08× | `FileHandle.read` loop + UTF-8 decode per chunk |
| CSV Stream + Filter | 279.1 ms | 22.8 ms | 0.08× | Same + row-by-row filter after materialisation |
| CSV Stream + GroupBy | 277.9 ms | 25.7 ms | 0.09× | Same + GroupBy on String-backed cells |
| Filter rows (100k) | 27.2 ms | 0.57 ms | 0.02× | `applyMask([Bool])` allocates full mask; non-Double columns box to `Any?` |
| SortBy double column | 73.5 ms | 6.5 ms | 0.09× | `gathered(at:)` is O(n × numCols) serial index copies |
| KMeans (10k×4, 3 clusters) | 44.7 ms | 11.7 ms | 0.26× | GPU routing overhead on small n; scalar distance loop |
| PCA SVD (1k×100 → 10 comps) | 1.98 ms | 0.75 ms | 0.38× | Full `dgesvd_`; covariance-based `dsyevd_` is faster for p < n |
| TS Decomposition (1k pts) | 0.45 ms | 0.09 ms | 0.21× | Scalar seasonal/residual loops; no Accelerate |

---

## Part A — Performance Fixes

---

### A1. `SwiftDataFrame` — Column-Parallel CSV Build

**Root cause**: after `SystemsCSVParser` produces offsets, the code materialises *all* cells as
`[String?]` in a serial loop, then type-infers by scanning those `String` copies a second time.

**Fix**: two-phase parallel column builder.

#### [MODIFY] `Sources/SwiftDataFrame/IO/CSVReader.swift`

Replace the `rawCells` block in `read(url:options:)`:
- Group `[[CSVFieldOffset]]` by column index after parsing.
- Spawn one `withTaskGroup` child task per column.
- Each task calls `VectorizedByteParsers.parseDouble` / `parseInt` / `parseBool` directly on the
  `UnsafeBufferPointer<UInt8>` — no intermediate `String` allocation.
- String columns: `String(decoding: UnsafeBufferPointer(start: base+offset, count: length), as: UTF8.self)` — single allocation per cell, no `trimmingCharacters` copy.

**Target**: CSV Read < 25 ms  (from 71.5 ms)

---

### A2. `SwiftDataFrame` — Memory-Mapped Streaming CSV

**Root cause**: `readStream` uses `FileHandle.read` → `Data` → `String(data:encoding:)` →
`splitLines` — three full passes over every 256 KB buffer chunk.

**Fix**: mmap-backed streaming cursor.

#### [MODIFY] `Sources/SwiftDataFrame/IO/CSVReader.swift` — `readStream`

- Map the entire file once with `Data(contentsOf:options:.alwaysMapped)`.
- Use `SystemsCSVParser` to find newline byte positions across the whole buffer — O(n) single pass.
- Partition the resulting row-index array into chunks of `chunkSize` rows.
- Yield each chunk's `DataFrame` directly from the raw buffer using the column-parallel builder
  from A1; never materialise a `String` line array.

**Target**: CSV Stream Read < 35 ms; Stream+Filter and Stream+GroupBy < 40 ms  (from ~278 ms)

---

### A3. `SwiftDataFrame` — Bitmap-Free Filter (`filterRows`)

**Root cause**: `filter(column:where:)` for non-`Double` columns calls `col.value(at:i)` (boxes
to `Any?` via protocol dispatch) then allocates a full `[Bool]` mask, then `applyMask` scans it
again to build indices.

**Fix**: return `[Int]` indices directly from typed column fast-paths.

#### [MODIFY] `Sources/SwiftDataFrame/Columns/TypedColumn.swift`

Add `func filteredIndices(matching condition: FilterCondition) -> [Int]` specialisations:
- `Double`: existing `mask(matching:)` converted to index list (no second scan).
- `Int64` / `Int`: `SIMD8<Int64>` lane comparisons to build index list without boxing.
- `String`: direct `values[i]` access; no `value(at:)` protocol call.
- All others: fall back to current path.

#### [MODIFY] `Sources/SwiftDataFrame/Core/DataFrame.swift`

- `filter(column:where:)` calls `filteredIndices` → `gathered(at:)` directly, skipping `applyMask`.

**Target**: Filter rows < 3 ms  (from 27.2 ms)

---

### A4. `SwiftDataFrame` — Parallel Gather + `vDSP_vgathrD` Sort

**Root cause**: `gathered(at:indices)` rebuilds every column serially with a scalar loop —
O(n × numCols) cache-unfriendly index operations.

**Fix**: parallel gather with SIMD gather for Double columns.

#### [MODIFY] `Sources/SwiftDataFrame/Core/DataFrame.swift`

Replace `gathered(at:)` with `parallelGathered(at:)`:
- `withTaskGroup(of: (Int, any AnyColumn).self)` — one task per column.
- For `TypedColumn<Double>`: use `vDSP_vgathrD` (Accelerate indexed gather) instead of scalar loop.
- For `TypedColumn<Int64>` / `String`: `withUnsafeMutableBufferPointer` + direct index copy.

#### [MODIFY] `Sources/SwiftDataFrame/Columns/TypedColumn.swift`

Add internal `func vGather(at indices: [Int]) -> TypedColumn<T>` for the parallel gather tasks.

**Target**: SortBy double column < 15 ms  (from 73.5 ms)

---

### A5. `SwiftCluster` — KMeans CPU Fast-Path & Adaptive Routing

**Root cause**: `HardwareRouter` routes 10k×4 to GPU (MLX), incurring device init + memory
transfer overhead. CPU `fitCPU` also has a scalar squared-distance inner loop.

**Fix**: vDSP distance kernel + lower GPU threshold.

#### [MODIFY] `Sources/SwiftCluster/Core/KMeans.swift`

- Lower GPU threshold in `HardwareRouter` for KMeans: use CPU when `nSamples × nFeatures < 500_000`.
- In `fitCPU`: replace scalar distance loop with `vDSP_distancesqD` per centroid.
- Centroid accumulation: `vDSP_vaddD` for sums, `vDSP_vsdivD` for mean.
- KMeans++ init: parallel distance minimisation via `withTaskGroup` instead of O(k·n) serial loop.

**Target**: KMeans < 15 ms  (from 44.7 ms)

---

### A6. `SwiftCluster` — PCA Covariance-Based Eigen-Decomposition

**Root cause**: always calls `dgesvd_` (full SVD on the data matrix). For p < n, computing the
p×p covariance matrix and calling `dsyevd_` (symmetric eigensolver) is O(n·p² + p³) vs O(m·n·p).

**Fix**: branch on `nFeatures < nSamples`.

#### [MODIFY] `Sources/SwiftCluster/Core/PCA.swift`

New `fitCPUCov` path when `nFeatures ≤ nSamples`:
1. Mean-centre `X` using `vDSP_meanvD` (already exists).
2. `C = XᵀX / (n-1)` via `cblas_dsyrk` (symmetric rank-k, O(n·p²)).
3. `dsyevd_` divide-and-conquer eigensolver on p×p `C` (O(p³), much smaller constant).
4. Top `nComponents` eigenvectors → principal components.

Keep `dgesvd_` path when `nFeatures > nSamples`.

**Target**: PCA SVD < 1 ms  (from 1.98 ms)

---

### A7. `SwiftForecast` — Vectorised Time Series Decomposition

**Root cause**: seasonal and residual components are built with scalar loops; moving-average
trend uses a scalar rolling sum.

**Fix**: replace all loops with `vDSP` vector arithmetic.

#### [MODIFY] Decomposition.swift in `Sources/SwiftForecast/Core/`

- Trend (moving average): `vDSP_convD` boxcar kernel instead of scalar rolling sum.
- Detrended: `vDSP_vsubD(x, 1, trend, 1, detrended, 1, n)`.
- Seasonal per-bin average: `vDSP_meanvD` per seasonal index bin instead of Dictionary accumulator.
- Seasonal broadcast back: fill seasonal array with bin means using pointer arithmetic.
- Residual: `vDSP_vsubD(detrended, 1, seasonal, 1, residual, 1, n)`.

**Target**: TS Decomposition < 0.12 ms  (from 0.45 ms)

---

## Part B — New Features

---

### B1. NEW: `SwiftDataFrame` — `DataFrame.join` (Hash Join)

No `merge` / `join` equivalent exists in SwiftSci today. This is the single most-requested
DataFrame operation in data pipelines.

#### [NEW] `Sources/SwiftDataFrame/Core/DataFrame+Join.swift`

```swift
public enum JoinKind { case inner, left, right, outer }

extension DataFrame {
    public func join(
        _ other: DataFrame,
        on key: String,
        how: JoinKind = .inner
    ) throws -> DataFrame
}
```

Implementation:
- Build hash table `[AnyHashable: [Int]]` on the smaller DataFrame's key column.
- Probe loop on the larger DataFrame; collect (left, right) index pairs.
- Assemble result columns using `parallelGathered(at:)` from A4.
- Null keys never match (standard SQL semantics).
- Disambiguate duplicate column names with `_x` / `_y` suffix.

---

### B2. NEW: `SwiftML` — `MLPClassifier` / `MLPRegressor`

Pure Accelerate-backed MLP — closes the `LinearRegression` benchmark gap and enables
simple deep-learning style workflows without requiring MLX.

#### [NEW] `Sources/SwiftML/Core/MLP.swift`

```swift
public final class MLPClassifier {
    public init(
        hiddenLayerSizes: [Int],
        activation: ActivationFunction = .relu,
        solver: MLPSolver = .adam,
        maxIter: Int = 200,
        learningRate: Double = 1e-3,
        seed: Int = 42
    )
    public func fit(features: [[Double]], targets: [Double]) throws
    public func predict(features: [[Double]]) throws -> [Double]
    public func predictProbability(features: [[Double]]) throws -> [[Double]]
}
public final class MLPRegressor { /* same init */ }
```

- Forward pass: `cblas_dgemm` per layer; `vDSP_vaddD` for bias.
- Activations: ReLU via `vDSP_vthresD`; sigmoid/tanh via `vvexp`.
- Backward pass: chain-rule gradient via `vDSP` ops; Adam/SGD optimiser.
- `Codable` persistence via `MLPModelState` (pattern from v1.4 `DecisionTree`).

---

### B3. NEW: `SwiftDataFrame` — `pivot` & `melt`

Two fundamental reshaping operations needed for analytics and reporting pipelines.

#### [NEW] `Sources/SwiftDataFrame/Core/DataFrame+Pivot.swift`

```swift
extension DataFrame {
    public func pivot(
        index: String,
        columns: String,
        values: String,
        aggFunc: Aggregation = .mean
    ) throws -> DataFrame

    public func melt(
        idVars: [String],
        valueVars: [String],
        varName: String = "variable",
        valueName: String = "value"
    ) throws -> DataFrame
}
```

`melt` is the inverse of `pivot`; `pivot(index:columns:values:).melt(...)` round-trips correctly.

---

## Part C — DocC Documentation Sprint

The v1.4 audit found **60% of 674 public symbols have no doc comment** and **0 DocC articles exist**.

### C1. Fix per-module coverage to ≥ 80%

Priority order (worst first):

| Module | Current | Gap | Comments needed |
|---|---|---|---|
| SwiftOptimize | 20% | 60% | ~37 |
| SwiftForecast | 21% | 59% | ~43 |
| SwiftML | 25% | 55% | ~91 |
| SwiftLLM | 33% | 47% | ~17 |
| SwiftStats | 41% | 39% | ~21 |
| SwiftPreprocessing | 48% | 32% | ~48 |
| SwiftCluster | 51% | 29% | ~15 |
| SwiftNLP | 57% | 23% | ~11 |
| SwiftDataFrame | 58% | 22% | ~26 |

For every public method add at minimum:
```swift
/// One-sentence summary.
///
/// - Parameter name: Description.
/// - Returns: Description.
/// - Throws: `DataFrameError.columnNotFound` if …
```

### C2. Write 3 DocC Articles

Create `Sources/SwiftDataFrame/Documentation.docc/`:

- `GettingStarted.md` — load CSV → filter → groupBy → export (the Titanic workflow)
- `MachineLearningWorkflow.md` — StandardScaler → train/test split → RandomForest → metrics
- `TimeSeriesForecasting.md` — ARIMA, Holt-Winters, decomposition

### C3. Fix Landing Page

Create `Sources/SwiftSci.docc/SwiftSci.md` with `@Links` to all 10 modules so the
landing page is an ecosystem overview, not a forced redirect to SwiftDataFrame.

### C4. Mark Low-Level Internals

Add `/// - Warning: Low-level API. Prefer \`DataFrame(csv:)\`.` to `VectorizedByteParsers`
and `SystemsCSVParser`, or make them `internal` (preferred).

---

## Benchmark Targets for v1.5

| Benchmark | v1.4 Swift | v1.4 Python | v1.5 Target | CI Gate |
|---|---|---|---|---|
| CSV Read (100k rows) | 71.5 ms | 18.8 ms | **< 25 ms** | info |
| CSV Stream Read | 276.8 ms | 21.7 ms | **< 35 ms** | info |
| CSV Stream + Filter | 279.1 ms | 22.8 ms | **< 40 ms** | info |
| CSV Stream + GroupBy | 277.9 ms | 25.7 ms | **< 40 ms** | info |
| Filter rows (100k) | 27.2 ms | 0.57 ms | **< 3 ms** | **CI** |
| SortBy double column | 73.5 ms | 6.5 ms | **< 15 ms** | info → CI |
| KMeans (10k×4, 3 clusters) | 44.7 ms | 11.7 ms | **< 15 ms** | info |
| PCA SVD (1k×100 → 10 comps) | 1.98 ms | 0.75 ms | **< 1 ms** | info |
| TS Decomposition (1k pts) | 0.45 ms | 0.09 ms | **< 0.12 ms** | info |

> `Filter rows` is the only new CI gate — 0.02× is the starkest gap and the fix is mechanical.

---

## Verification Plan

### Automated Tests

```bash
# Full test suite (release build)
swift test --configuration release -Xswiftc -O

# Benchmark suite
swift run --configuration release BenchmarkRunner
```

### CI Gate Updates

Update `.github/workflows/ci.yml` benchmark step:
- Add `Filter rows (100k)` gate: fail if Swift speedup < 1.5×
- Add `CSV Read (100k rows)` gate: fail if Swift speedup < 0.8× (parity first)

### Manual Verification

- `DataFrame.join`: inner/left/outer on Titanic (`PassengerId` key) with null key rows.
- `MLPClassifier`: > 90% accuracy on `makeClassification(n=1000, features=10, seed=42)`.
- `pivot` / `melt` round-trip: `df.melt(…).pivot(…)` reproduces original shape and values.
- DocC site: open each new article link, verify no 404s.

---

## Open Questions

1. **Filter rows CI gate**: promote to gated in v1.5, or informational until nullable edge cases
   (mixed-type columns, all-null columns) are verified?

2. **MLP backend**: Accelerate-only (CPU, no MLX dependency, no CI Metal issue) or also MLX GPU
   path for large hidden layers? Recommendation: CPU-only for v1.5, add MLX path in v1.6.

3. **`DataFrame.join` key types**: support any `Hashable SupportedType`, or restrict to
   `String` and `Int` / `Int64` (most common join keys) for v1.5?

4. **DocC `VectorizedByteParsers` / `SystemsCSVParser` visibility**: make `internal` (breaking
   for any downstream code using them directly) or keep `public` with Warning note?
