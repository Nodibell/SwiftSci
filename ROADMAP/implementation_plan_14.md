# Implementation Plan 14 — Full Audit & v1.4.0 Improvements

> Ground truth: read every source file. No assumptions from previous plans.
> Status: **July 2026**

---

## 0. Inventory — What Already Exists

| Module | Key Files |
|---|---|
| **SwiftDataFrame** | `DataFrame.swift`, `TypedColumn.swift`, `GroupedDataFrame.swift`, `CSVReader.swift`, `CSVWriter.swift`, `JSONReader.swift` |
| **SwiftStats** | `Stats+Descriptive.swift`, `Stats+HypothesisTesting.swift`, `Stats+Correlation.swift`, `Stats+LinearAlgebra.swift` |
| **SwiftPreprocessing** | 24 files: StandardScaler, MinMaxScaler, RobustScaler, Normalizer, Imputer, OneHotEncoder, LabelEncoder, OrdinalEncoder, KBinsDiscretizer, PolynomialFeatures, PowerTransformer, MissingValueIndicator, FeatureEngineering, ColumnTransformer, SelectKBest, VarianceThreshold, SMOTE, RandomUndersampler, TrainTestSplit, Pipeline, WiredMemoryManager, HardwareRouter |
| **SwiftML** | DecisionTree (Clf+Reg), RandomForest (Clf+Reg), LogisticRegression, LinearRegression, GradientBoostedTreesRegressor, CalibratedClassifier, EstimatorPipeline (Classification+Regression), DatasetUtilities |
| **SwiftCluster** | KMeans, PCA, DBSCAN, OutlierDetection (`IsolationForest`, `LocalOutlierFactor`) |
| **SwiftNLP** | BPETokenizer, CountVectorizer, TFIDFVectorizer, WordEmbeddings |
| **SwiftOptimize** | GridSearchCV, RandomizedSearchCV, KFold, Metrics (full: accuracy, precision, recall, F1, MCC, Cohen's Kappa, logLoss, brierScore, ROC/AUC, PR curves, MSE/RMSE/MAE/R2) |
| **SwiftForecast** | ARIMA, SARIMA, ExponentialSmoothing, GARCHModel, KalmanFilter, TimeSeriesDecomposition, TimeSeriesTransformers |
| **SwiftLLM** | GGUFParser, SafeTensorsParser, TransformerDecoder, Sampler |
| **SwiftExplain** | KernelSHAP |

---

## 1. Confirmed Bugs

### 1.1 `GradientBoostedTreesRegressor` — commented-out guards (CRITICAL)

**File**: `Sources/SwiftML/Core/GradientBoosting.swift`, lines 23–24

```swift
// guard nEstimators > 0 else { throw MLError.invalidParameter("nEstimators must be > 0") }
// guard learningRate > 0 else { throw MLError.invalidParameter("learningRate must be > 0") }
```

Guards are commented out. `GradientBoostedTreesRegressor(nEstimators: 0)` succeeds silently and returns an empty tree forest. Fix: uncomment both guards.

---

### 1.2 Test suite `ParserTests.swift` — compile errors blocking CI (HIGH)

**File**: `Tests/SwiftLLMTests/ParserTests.swift`

From `build_errors.txt`:
- `var version: UInt32 = 3.littleEndian` — integer literal `3` has type `Int`, not `UInt32`. Fix: `UInt32(3).littleEndian`
- `var tensorCount: UInt64 = 1.littleEndian` — same pattern. Fix: `UInt64(1).littleEndian`
- `var metadataCount: UInt64 = 0.littleEndian` — same. Fix: `UInt64(0).littleEndian`
- Multiple `UnsafeBufferPointer(start: &val, count: n)` creating dangling pointers — fix with `withUnsafePointer`/`withUnsafeBufferPointer` scoped calls.

This is why `SwiftLLMTests` is `--skip`-ped in CI. Once fixed, remove the skip.

---

### 1.3 `RandomForestClassifier.predict` — missing `async` on actor method (MEDIUM)

**File**: `Sources/SwiftML/Core/RandomForest.swift`, lines 90 and 102

```swift
public func predict(features: [[Double]]) throws -> [Int] {   // no async
public func predictProbability(features: [[Double]]) throws -> [[Double]] {  // no async
```

These are actor methods. The protocol `ClassifierEstimator` declares `predict` as `async throws`, but the actor conforms with a `throws`-only version. This works currently but is inconsistent and may break under stricter checking. Fix: add `async` to match the protocol exactly.

Same issue in `RandomForestRegressor.predict` (line 234) and `GradientBoostedTreesRegressor.fit`/`predict` (lines 32 and 71).

---

### 1.4 `KMeans` — naive centroid initialisation (poor convergence quality)

**File**: `Sources/SwiftCluster/Core/KMeans.swift`, lines 167 and 203

Both CPU and GPU paths initialize centroids as `X[0..<nClusters]` (first K rows). This is equivalent to "first-K" initialisation, which converges poorly on sorted or clustered data. No `kmeans++` or random initialisation.

---

### 1.5 `SelectKBest` — ignores `targets`, scores by variance only

**File**: `Sources/SwiftPreprocessing/Core/FeatureSelection.swift`, lines 63–68

```swift
public func fit(features: [[Double]], targets: [Double]?) throws {
    // targets accepted but never used
    let variance = col.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / numSamples
    scores.append((f, variance))
}
```

`targets` parameter is accepted but never used. True `SelectKBest` scores features using ANOVA F-value (supervised). The current implementation is equivalent to `VarianceThreshold` with K output.

---

### 1.6 `SMOTE` — uses `Int.random` not seeded RNG

**File**: `Sources/SwiftPreprocessing/Core/ImbalancedLearning.swift`, lines 55 and 70

```swift
let idx = Int.random(in: 0..<count)      // not seeded
let neighborIdx = distances[Int.random(in: 0..<k)].index  // not seeded
let gap = Double.random(in: 0.0...1.0)   // not seeded
```

`self.seed` is stored but never used. Every call produces different results. Fix: use `SeededRandom` with `self.seed`.

---

### 1.7 `RandomUndersampler` — `seed` stored but `shuffled()` is uncontrolled

**File**: `Sources/SwiftPreprocessing/Core/ImbalancedLearning.swift`, line 114

```swift
let shuffled = indices.shuffled()   // not seeded
```

Same pattern as 1.6 — seed stored but not used.

---

### 1.8 `TransformerDecoder` — single attention layer, no feed-forward network

**File**: `Sources/SwiftLLM/Core/TransformerDecoder.swift`

The architecture is: `Embedding + PositionEmbedding → MultiHeadAttention → RMSNorm → Linear`. There is no FFN sublayer, no second LayerNorm, no depth parameter. Architecture label is misleading for production use.

---

### 1.9 `GGUFParser` — silently corrupts quantized tensor data

**File**: `Sources/SwiftLLM/Core/GGUFParser.swift`, lines 161–168

```swift
default:
    // Q4_0, Q4_1, Q8_0 etc. treated as Float32
    array = MLXArray(tensorData, info.shape, dtype: .float32)
```

GGUF files in the wild are predominantly 4-bit or 8-bit quantized. The parser reads wrong byte counts for Q4 tensors (uses `elementCount * 4` bytes instead of `elementCount * 0.5`), producing corrupt arrays. Should throw `unsupportedQuantization` or implement dequantization.

---

### 1.10 `DataFrame.gathered(at:)` — `public` leaks internal implementation detail

**File**: `Sources/SwiftDataFrame/Core/DataFrame.swift`, line 378

```swift
public func gathered(at indices: [Int]) -> DataFrame {
```

Low-level row-gather primitive used internally by `filter`, `sample`, `head`, etc. Should be `internal`.

---

### 1.11 `HardwareRouter` — CPU/GPU threshold hardcoded, not configurable

**File**: `Sources/SwiftPreprocessing/Core/HardwareRouter.swift`

Fixed sample-count thresholds for routing. Users on M1 Pro vs. M3 Max have very different optimal crossover points with no override mechanism.

---

## 2. Architecture / Design Problems

### 2.1 Dual `predict` signatures on actors — sync vs. async inconsistency

`RandomForestClassifier`, `RandomForestRegressor`, `GradientBoostedTreesRegressor` implement actor methods as `throws`-only (no `async`), while protocols require `async throws`. `LogisticRegression` and `LinearRegression` correctly declare `async throws`. Cannot mix model types in generic code.

**Fix**: all actor protocol `predict`/`fit` overrides must be `async throws`.

---

### 2.2 `ClassificationPipeline` / `RegressionPipeline` — `@unchecked Sendable`

**File**: `Sources/SwiftML/Core/EstimatorPipeline.swift`, lines 6 and 43

`@unchecked Sendable` bypasses Swift 6 concurrency checks. Both classes hold `any PreprocessingTransformer` (class-based existential with mutable state). Risk of data race if a transformer modifies state from multiple threads.

**Fix**: require `PreprocessingTransformer: Sendable` or copy state on fit.

---

### 2.3 `CalibratedClassifier` — `@unchecked Sendable` on `final class` with mutable state

**File**: `Sources/SwiftML/Core/CalibratedClassifier.swift`, line 5

Stores mutable `a: Double` and `b: Double`. As `final class` (not `actor`), concurrent access after `fit` is a data race. Should be `actor` or use a lock.

---

### 2.4 `WiredMemoryManager` — `maxConcurrentTasks: 2` hardcoded default

**File**: `Sources/SwiftPreprocessing/Core/WiredMemoryManager.swift`, line 6

```swift
public static let shared = WiredMemoryManager(maxConcurrentTasks: 2)
```

Too conservative for M3 Max (128 GB, 40 GPU cores). Should default to `ProcessInfo.processInfo.processorCount / 2`.

---

### 2.5 `DataFrame` — dual column storage (dict + order array) adds sync complexity

**File**: `Sources/SwiftDataFrame/Core/DataFrame.swift`, lines 11–12

```swift
internal let _columns: [String: any AnyColumn]
internal let _columnOrder: [String]
```

Both must be kept in sync across every transformation. Consider migrating to Swift Collections `OrderedDictionary` (P3, breaking API change).

---

### 2.6 `SwiftAnalyticsMacros` — directory exists but not wired into `Package.swift`

`Package.swift` has no `.macro(...)` target for `SwiftAnalyticsMacros`. Either wire properly or remove the directory.

---

### 2.7 `ARIMAModel.fit` — actor method declared `throws` not `async throws`

**File**: `Sources/SwiftForecast/Core/ARIMA.swift`, line 33

`ARIMAModel` is an `actor` but `fit` is `throws`-only. Same inconsistency as 2.1 across the forecasting module.

---

## 3. Missing Features (Prioritised)

### P0 — Blocking or correctness-critical

| # | Feature | Module | Notes |
|---|---|---|---|
| 3.1 | Fix GBDT commented-out guards (bug 1.1) | SwiftML | 2-line fix |
| 3.2 | Fix ParserTests compile errors (bug 1.2) | SwiftLLMTests | Unblocks CI |
| 3.3 | Fix SMOTE / RandomUndersampler seeded RNG (bugs 1.6, 1.7) | SwiftPreprocessing | Reproducibility |
| 3.4 | Fix SelectKBest ANOVA F-value scoring (bug 1.5) | SwiftPreprocessing | Correctness |
| 3.5 | Add `async` to RF/GBDT actor protocol overrides (issue 2.1) | SwiftML | Protocol conformance |

### P1 — High value, low risk

| # | Feature | Module | Notes |
|---|---|---|---|
| 3.6 | KMeans++ initialisation | SwiftCluster | Fixes convergence on sorted data |
| 3.7 | GGUFParser: throw on unsupported quantization | SwiftLLM | Data safety |
| 3.8 | Model persistence (Codable save/load) | SwiftML | LinearReg, LogisticReg, DT, RF |
| 3.9 | Feature importances on DecisionTree/RandomForest | SwiftML | Needed for RFE and SHAP |
| 3.10 | `RecursiveFeatureElimination` | SwiftPreprocessing | Depends on 3.9 |

### P2 — Medium priority

| # | Feature | Module | Notes |
|---|---|---|---|
| 3.11 | `NGramTokenizer` + `HashingVectorizer` + `StopWords` | SwiftNLP | Text pipeline completion |
| 3.12 | `InteractionFeatures` + `DateFeatures` + `FeatureHasher` | SwiftPreprocessing | Feature engineering |
| 3.13 | `LagTransformer` + `RollingWindow` + `ExpandingWindow` | SwiftForecast | Time series features |
| 3.14 | `EllipticEnvelope` outlier detection | SwiftCluster | Robust covariance via MCD |
| 3.15 | `ADASYN` oversampler | SwiftPreprocessing | Complement to SMOTE |
| 3.16 | Multi-layer TransformerDecoder (depth + FFN sublayer) | SwiftLLM | Architectural correctness |

### P3 — Lower priority / design decisions needed

| # | Feature | Module | Notes |
|---|---|---|---|
| 3.17 | GGUF Q4/Q8 dequantization | SwiftLLM | Needs block layout spec |
| 3.18 | HardwareRouter configurable thresholds | SwiftPreprocessing | Ergonomics |
| 3.19 | DataFrame → OrderedDictionary migration | SwiftDataFrame | Breaking API change |
| 3.20 | SwiftAnalyticsMacros target wiring | Package.swift | Needs design doc |
| 3.21 | Dataset generators (makeRegression, makeMoons, etc.) | Test utilities target | Useful for tests |
| 3.22 | Stemmer / Lemmatizer | SwiftNLP | Language choice required first |
| 3.23 | HolidayEncoder | SwiftForecast | Calendar/locale scope decision needed |

---

## 4. Specific Code Fixes

### Fix 1 — `GradientBoostedTreesRegressor` guards

**File**: `Sources/SwiftML/Core/GradientBoosting.swift`

```diff
- // guard nEstimators > 0 else { throw MLError.invalidParameter("nEstimators must be > 0") }
- // guard learningRate > 0 else { throw MLError.invalidParameter("learningRate must be > 0") }
+ guard nEstimators > 0 else { throw MLError.invalidParameter("nEstimators must be > 0") }
+ guard learningRate > 0 else { throw MLError.invalidParameter("learningRate must be > 0") }
```

---

### Fix 2 — ParserTests integer type literals

**File**: `Tests/SwiftLLMTests/ParserTests.swift`

```diff
- var version: UInt32 = 3.littleEndian
+ var version: UInt32 = UInt32(3).littleEndian

- var tensorCount: UInt64 = 1.littleEndian
+ var tensorCount: UInt64 = UInt64(1).littleEndian

- var metadataCount: UInt64 = 0.littleEndian
+ var metadataCount: UInt64 = UInt64(0).littleEndian
```

All `UnsafeBufferPointer(start: &val, count: n)` patterns must be wrapped in `withUnsafePointer(to:)` / `withUnsafeBufferPointer` scoped calls to eliminate dangling pointer warnings.

---

### Fix 3 — SMOTE seeded RNG

**File**: `Sources/SwiftPreprocessing/Core/ImbalancedLearning.swift`

Replace `Int.random` and `Double.random` calls with `SeededRandom(seed: Int(self.seed))`:

```swift
var rng = SeededRandom(seed: Int(self.seed))
let idx = rng.nextInt(upperBound: count)
let neighborIdx = distances[rng.nextInt(upperBound: k)].index
// Add nextDouble() to SeededRandom via xorshift normalised to [0,1)
let gap = Double(rng.next()) / Double(UInt64.max)
```

Same fix for `RandomUndersampler` — replace `indices.shuffled()` with a seeded Fisher-Yates shuffle using `SeededRandom`.

---

### Fix 4 — SelectKBest ANOVA F-value scoring

**File**: `Sources/SwiftPreprocessing/Core/FeatureSelection.swift`

When `targets != nil`, compute ANOVA F-statistic per feature using `SwiftStats`:

```swift
public func fit(features: [[Double]], targets: [Double]?) throws {
    if let targets {
        // Per-feature ANOVA F-value using Stats.oneWayANOVA
        // Group samples by target class, compute F-statistic
        // Requires adding SwiftStats as dependency of SwiftPreprocessing in Package.swift
    } else {
        // Existing variance-based scoring (unsupervised fallback)
    }
}
```

> **Note**: Adding `SwiftPreprocessing → SwiftStats` is safe (no cycle): `SwiftStats → SwiftDataFrame`, `SwiftPreprocessing → SwiftDataFrame`. No cycle introduced.

---

### Fix 5 — KMeans++ Initialisation

**File**: `Sources/SwiftCluster/Core/KMeans.swift`

Replace `X[0..<nClusters]` (CPU) and `X[0..<nClusters]` (GPU) with KMeans++ sampling:

```swift
// 1. Choose first centroid uniformly at random from X
// 2. For each subsequent centroid k:
//    a. D(x)² = min squared distance from x to any already-chosen centroid
//    b. Sample next centroid proportional to D(x)²
```

Apply to both `fitCPU` and `runFitGPU` paths.

---

### Fix 6 — RF/GBDT `async throws` alignment

**Files**: `RandomForest.swift`, `GradientBoosting.swift`

```diff
- public func predict(features: [[Double]]) throws -> [Int] {
+ public func predict(features: [[Double]]) async throws -> [Int] {

- public func predictProbability(features: [[Double]]) throws -> [[Double]] {
+ public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
```

Update all callers from `try model.predict(...)` to `try await model.predict(...)`.

---

## 5. CI / Build Improvements

### 5.1 Remove test skip list once ParserTests is fixed

**File**: `.github/workflows/ci.yml`

```diff
  swift test --enable-code-coverage -v \
-   --skip SwiftLLMTests \
-   --skip SwiftMLTests \
```

`SwiftMLTests` skip also needs investigation — if GPU-only, add a capability flag rather than a blanket skip.

### 5.2 Add release build step

```yaml
- run: swift build -c release
```

Release builds with whole-module optimisation catch additional warnings and potential performance regressions.

### 5.3 Linux CI target (optional)

`SwiftDataFrame`, `SwiftStats`, `SwiftNLP` have no Apple-specific dependencies. Add an optional Linux runner that builds only platform-independent targets.

---

## 6. Dependency Analysis

```
SwiftDataFrame          <- Arrow (apache/arrow-swift, from: 21.0.0)
SwiftStats              <- SwiftDataFrame + Accelerate
SwiftPreprocessing      <- SwiftDataFrame + MLX
SwiftML                 <- SwiftPreprocessing + MLX + MLXNN
SwiftCluster            <- SwiftDataFrame + SwiftStats + SwiftPreprocessing + MLX
SwiftNLP                <- SwiftDataFrame
SwiftOptimize           <- SwiftDataFrame + SwiftML
SwiftForecast           <- SwiftDataFrame + SwiftStats + Accelerate
SwiftLLM                <- SwiftNLP + MLX + MLXNN
SwiftExplain            <- SwiftML + SwiftStats + SwiftDataFrame + SwiftPreprocessing
```

**Observations**:
- `SwiftNLP` does not depend on `MLX` — good, NLP preprocessing stays compute-device agnostic.
- `SwiftExplain` pulls in 4 modules — widest dependency set. Breaking changes in core propagate here immediately.
- `SwiftOptimize` depends on `SwiftML` via `KFold.swift`. This couples validation to models. Consider making `KFold` depend on protocol types only.
- `mlx-swift` is pinned to `exact: "0.31.6"` — requires manual bumps for every patch. Consider `upToNextMinor` if API has been stable.
- `arrow-swift` pinned to `from: "21.0.0"` — reasonable, but verify Arrow 22.x does not introduce breaking changes when it releases.

---

## 7. Code Quality Issues

### 7.1 Non-English comments in production source

**Files**: `GradientBoosting.swift` line 13, `RandomForest.swift` line 31

```swift
// Ліс тепер зберігається як масив плоских дерев
// DOD Архітектура: Ліс — це масив масивів плоских вузлів
```

Translate to English for open-source contributor accessibility.

---

### 7.2 `bootstrapSample` — unnamespaced module-level function

**File**: `Sources/SwiftML/Core/RandomForest.swift`, line 5

```swift
func bootstrapSample(features: [[Double]], targets: [Double], seed: Int) -> ([[Double]], [Double]) {
```

Free function with no access modifier pollutes the internal namespace. Move to `private static func` inside `RandomForestClassifier` or into a shared `TreeUtils.swift` with `internal` access.

---

### 7.3 `giniImpurity`, `entropy`, `mseImpurity`, `bestSplit` — module-level free functions

**File**: `Sources/SwiftML/Core/DecisionTree.swift`, lines 35–113

Same issue as 7.2. Namespace under `internal enum DecisionTreeHelpers` or mark `internal`.

---

### 7.4 `DataFrame.shape` uses `_columns.first` with undefined dictionary order

**File**: `Sources/SwiftDataFrame/Core/DataFrame.swift`, line 76

```swift
(rows: _columns.first.map { $0.value.count } ?? 0, ...)
```

`Dictionary.first` order is undefined. Safe (all columns have same count) but semantically odd. Consider caching `_rowCount` as `internal let` at init time.

---

### 7.5 `KernelSHAP` — custom O(M^3) Gauss-Jordan solver

**File**: `Sources/SwiftExplain/Core/KernelSHAP.swift`, lines 160–202

The custom `solveLinearSystem` is dense Gauss-Jordan elimination. For M > 100 features this bottlenecks. Replace with LAPACK `dposv_` (symmetric positive definite solver) via Accelerate — 10–50x faster and numerically more stable on the ZTWZ matrix which is always symmetric positive definite.

---

### 7.6 `GGUFParser.readInt` — slow `copyBytes` for every integer read

**File**: `Sources/SwiftLLM/Core/GGUFParser.swift`, line 18

```swift
_ = data.copyBytes(to: valPtr, from: offset..<(offset + size))
```

`copyBytes` is byte-by-byte. For large GGUF files (LLaMA-3 has 30+ metadata keys), this is unnecessarily slow. Replace with:

```swift
data.withUnsafeBytes { ptr in
    ptr.load(fromByteOffset: offset, as: T.self)
}
```

Zero-copy read from memory-mapped data.

---

## 8. Release Notes — v1.4.0

```markdown
## v1.4.0

### Bug Fixes
- `GradientBoostedTreesRegressor`: Restored input validation guards that were accidentally commented out.
- `SMOTE` / `RandomUndersampler`: `seed` parameter is now honoured; oversampling is reproducible.
- `SelectKBest`: When `targets` are provided, now uses ANOVA F-value scoring instead of variance.
- `RandomForestClassifier` / `RandomForestRegressor` / `GradientBoostedTreesRegressor`: `predict` and `predictProbability` now declared `async throws` to match protocol requirements.
- `Tests/SwiftLLMTests/ParserTests.swift`: Fixed integer literal type errors and dangling pointer warnings; `SwiftLLMTests` is no longer skipped in CI.

### Improvements
- `KMeans`: Replaced first-K centroid initialisation with KMeans++ for robust convergence.
- `KernelSHAP`: ZTWZ linear system now solved via LAPACK `dposv_` instead of custom Gauss-Jordan.
- `GGUFParser`: Throws `SwiftMLError.unsupportedQuantization` for Q4/Q8/Q5 tensor types instead of silently reading wrong byte counts.
- `GGUFParser.readInt`: Replaced `copyBytes` with zero-copy `withUnsafeBytes` load.
- `WiredMemoryManager.shared`: `maxConcurrentTasks` now defaults to `ProcessInfo.processorCount / 2` (minimum 2).
- `bootstrapSample`, `giniImpurity`, `entropy`, `mseImpurity`, `bestSplit` namespaced to avoid internal namespace pollution.
- Non-English inline comments translated to English.

### New Features
- `SelectKBest`: Supervised ANOVA F-value scoring when `targets` are provided.
- `FeatureImportances`: `DecisionTreeClassifier` and `RandomForestClassifier` expose `featureImportances: [Double]?` (Gini importance).
- `RecursiveFeatureElimination`: Feature elimination using model feature importances.
- `NGramTokenizer`: N-gram token extraction in `SwiftNLP`.
- `HashingVectorizer`: Feature hashing for text in `SwiftNLP`.
- `InteractionFeatures`: Pairwise feature interactions in `SwiftPreprocessing`.
- `DateFeatures` / `TimeFeatures`: Cyclical date/time component extraction in `SwiftPreprocessing`.
- `LagTransformer` / `RollingWindow` / `ExpandingWindow`: Time series feature transformers in `SwiftForecast`.
- Model persistence: Codable-based `save(to:)` / `load(from:)` for `LinearRegression`, `LogisticRegression`, `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`.
- CI: `SwiftLLMTests` no longer skipped; release build step added.
```

---

## 9. Execution Order

Phase 1 — Bug fixes (P0) [COMPLETED]
  1. Fix ParserTests.swift compile errors → remove SwiftLLMTests skip [DONE]
  2. Uncomment GradientBoostedTreesRegressor guards [DONE]
  3. Fix SMOTE / RandomUndersampler seeded RNG [DONE]
  4. Fix SelectKBest ANOVA scoring (+ add SwiftStats dependency to SwiftPreprocessing in Package.swift) [DONE]
  5. Add `async` to RF/GBDT predict/fit actor protocol overrides [DONE]

Phase 2 — Performance & quality [COMPLETED]
  6. SystemsCSVParser zero-copy RFC 4180 DFA + memory-mapped I/O [DONE]
  7. VectorizedByteParsers zero-allocation ASCII parsers [DONE]
  8. vDSP mean, variance, and stdDev reductions [DONE]
  9. DataFrame filterRows(by:) & argsort [DONE]
  10. KMeans++ initialisation (CPU + GPU paths) [DONE]
  11. GGUFParser: loadUnaligned fix [DONE]
  12. KernelSHAP: async prediction closure + KernelSHAP actor explain [DONE]

Phase 3 — New features [COMPLETED]
  13. Feature importances on DecisionTree / RandomForest [DONE]
  14. RecursiveFeatureElimination [DONE]
  15. Model persistence (Codable save/load for Linear/Logistic/DecisionTree/RandomForest) [DONE]
  16. NGramTokenizer + HashingVectorizer [DONE]
  17. InteractionFeatures + DateFeatures + TimeFeatures [DONE]
  18. LagTransformer + RollingWindow + ExpandingWindow [DONE]

Phase 4 — Architecture [COMPLETED]
  19. WiredMemoryManager default limit → dynamic (ProcessInfo) [DONE]
  20. CalibratedClassifier → converted to actor [DONE]
  21. DataFrame OrderedDictionary migration [EVALUATED]
  22. SwiftAnalyticsMacros target wiring [DONE]
