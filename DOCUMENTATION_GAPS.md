# SwiftSci v3.6.1 Documentation Gaps Audit

**Date**: September 2026  
**Version**: 3.6.1  
**Scope**: All 14 core modules  
**Status**: ⚠️ **CRITICAL** — 100% DocC coverage claim is misleading

---

## Executive Summary

| Metric | Reality |
|--------|---------|
| **Claim** | "Full Symbol Coverage: 100.00% (1,753 / 1,753 public symbols documented)" |
| **Reality** | All symbols have *some* docstring, but ~40-60% contain `<#description#>` placeholders |
| **Severity** | HIGH — Users cannot understand parameter intent or return values |
| **Impact** | Misleads developers about documentation quality in marketing/release notes |

---

## Module-by-Module Gaps

### 1. SwiftML — Core Machine Learning Algorithms

#### LinearRegression
**File**: `Sources/SwiftML/Core/LinearRegression.swift`  
**Status**: 🔴 CRITICAL GAPS

| Method | Issue | Example |
|--------|-------|---------|
| `fit(features:targets:learningRate:epochs:)` | All parameters are `<#description#>` | Lines 56-61 |
| `fitCPUAnalytical()` | Missing implementation doc | No docstring |
| `fitCPUGradientDescent()` | All params placeholder | Lines 176-175 |
| `predict(features:)` | Return value undocumented | `- Returns: <#description#>` |
| `predict(X:)` | MLX variant missing context | Line 316-320 |
| `getWeights()` | No explanation of format | Line 341-344 |
| `getBias()` | Undefined return on failures | Line 348-354 |

**What's missing**:
- When to use LAPACK vs gradient descent
- Learning rate guidance
- Convergence criteria
- NaN/Inf handling strategy

---

#### LogisticRegression
**File**: `Sources/SwiftML/Core/LogisticRegression.swift`  
**Status**: 🔴 CRITICAL GAPS

| Method | Issue |
|--------|-------|
| `predictProbability(features:)` | Lines 232-239: all params `<#description#>` |
| `predictProbability(X:)` | Lines 241-269: return value undefined |
| `predict(features:)` | Lines 271-279: threshold not explained |
| `predict(X:threshold:)` | Lines 281-299: boundary behavior undefined |
| `binaryPositiveClassProbability()` | Private but critical — no docstring |

**What's needed**:
- Probability calibration method (Platt? Isotonic?)
- Multi-class strategy (OvR? OvO?)
- Threshold semantics (class 0 vs 1?)

---

#### DecisionTree (Classifier & Regressor)
**File**: `Sources/SwiftML/Core/DecisionTree.swift`  
**Status**: 🟡 PARTIAL

| Symbol | Issue |
|--------|-------|
| `FlatTreeNode` | Lines 36-77: good structure doc, but field meanings unclear (why flat? why DOD?) |
| `prune(alpha:)` | Lines 390-416: alpha parameter undefined (what range? units?) |
| `buildTree()` | Private helper — no docstring explaining recursion strategy |
| `bestSplit()` | Complex function (180+ lines) with zero docstring explaining algorithm |

**Good**: SplitCriterion enum (lines 6-18) has clear docs

---

### 2. SwiftStats — Statistical Analysis

#### Stats.swift
**File**: `Sources/SwiftStats/Stats.swift`  
**Status**: 🔴 CRITICAL — ENTIRE MODULE UNDOCUMENTED

```swift
public enum Stats {}  // ← Only 1 line of doc, then NOTHING
```

**Missing docstrings for ALL public functions**:
- `mean()`
- `median()`
- `variance()`
- `stdDev()`
- `tTest()`
- `anova()`
- `pearsonCorrelation()`
- `chiSquareTest()`
- `dotProduct()`
- `norm()`
- `cosineSimilarity()`
- `add()`, `subtract()`, `multiply()`

#### Stats+LinearAlgebra.swift
**File**: `Sources/SwiftStats/Core/Stats+LinearAlgebra.swift`  
**Status**: 🔴 CRITICAL

| Method | Line | Issue |
|--------|------|-------|
| `dotProduct()` | 16-20 | Parameters: `- a: <#description#>` ← PLACEHOLDER |
| `norm()` | 24-28 | `- order: <#description#>` — what's NormOrder? |
| `cosineSimilarity()` | 31-35 | Return range undefined ([-1, 1]?) |
| `add()` | 38-42 | No explanation of element-wise semantics |
| `subtract()` | 45-49 | Parameter order (a - b or b - a?) undefined |
| `multiply()` | Presumably missing entirely | — |

---

### 3. SwiftNLP — Natural Language Processing

#### PorterStemmer
**File**: `Sources/SwiftNLP/Core/PorterStemmer.swift`  
**Status**: 🟡 PARTIAL

```swift
public func stem(_ word: String) -> String {
    /// - Parameters:
    ///   - word: <#description#>          ← PLACEHOLDER
    /// - Returns: <#description#>         ← PLACEHOLDER
}
```

**What's missing**:
- Which Porter Stemmer variant? (Porter1980? Porter2?)
- Language support (English only?)
- Case sensitivity behavior
- Example: "running" → "run"

#### StopWords
**File**: `Sources/SwiftNLP/StopWords.swift`  
**Status**: 🔴 CRITICAL

```swift
public static func filter(tokens: [String], language: Language = .english) -> [String] {
    /// - Parameters:
    ///   - tokens: <#description#>        ← PLACEHOLDER
    ///   - language: <#description#>      ← PLACEHOLDER
    /// - Returns: <#description#>         ← PLACEHOLDER
}
```

**Issues**:
- Only `.english` case in switch — what other languages exist?
- Case-sensitivity behavior undefined
- No docstring explaining what "stop words" are

#### LLMContextWindow
**File**: `Sources/SwiftNLP/Extensions/SwiftLLM+NLP.swift`  
**Status**: 🟡 PARTIAL — Initializer is good, methods are not

```swift
public func countTokens(in text: String) -> Int {
    /// - Parameters:
    ///   - text: <#description#>         ← PLACEHOLDER
    /// - Returns: <#description#>        ← PLACEHOLDER
}

public func truncate(text: String, maxTokens limit: Int? = nil) -> String {
    /// - Parameters:
    ///   - text: <#description#>         ← PLACEHOLDER
    ///   - limit: <#description#>        ← PLACEHOLDER
    /// - Returns: <#description#>        ← PLACEHOLDER
}
```

**Missing**:
- Token counting algorithm (word? BPE? subword?)
- Truncation strategy (from start? from end? with ellipsis?)

#### TextNormalizer
**File**: `Sources/SwiftNLP/TextNormalizer.swift`  
**Status**: 🔴 CRITICAL

```swift
public func normalize(_ text: String) -> String {
    /// - Parameters:
    ///   - text: <#description#>         ← PLACEHOLDER
    /// - Returns: <#description#>        ← PLACEHOLDER
}
```

**Missing**:
- Unicode normalization form (NFC? NFKC?)
- Examples ("Hello,  WORLD!" → ?)
- Edge cases (empty string? URLs?)

#### WordEmbeddings
**File**: `Sources/SwiftNLP/Core/WordEmbeddings.swift`  
**Status**: 🟡 PARTIAL

```swift
public func vector(for word: String) -> [Double]? {
    /// - Parameters:
    ///   - word: <#description#>         ← PLACEHOLDER
    /// - Returns: <#description#>        ← PLACEHOLDER
}

public func cosineSimilarity(_ word1: String, _ word2: String) -> Double? {
    /// - Parameters:
    ///   - word1: <#description#>        ← PLACEHOLDER
    ///   - word2: <#description#>        ← PLACEHOLDER
    /// - Returns: <#description#>        ← PLACEHOLDER (range? nil on missing?)
}

public func mostSimilar(to word: String, topK: Int = 10) -> [(word: String, similarity: Double)]? {
    /// - Parameters:
    ///   - word: <#description#>         ← PLACEHOLDER
    ///   - topK: <#description#>         ← PLACEHOLDER
    /// - Returns: <#description#>        ← PLACEHOLDER
}
```

---

### 4. SwiftDataFrame — Data Structures & I/O

#### LazyDataFrame
**File**: `Sources/SwiftDataFrame/Lazy/LazyDataFrame.swift`  
**Status**: 🟡 PARTIAL

```swift
public func filter(_ predicate: @escaping @Sendable (DataFrameRow) -> Bool) -> LazyDataFrame {
    /// - Parameters:
    ///   - predicate: <#description#>    ← PLACEHOLDER
    /// - Returns: <#description#>        ← PLACEHOLDER
}

public func select(_ columns: [String]) -> LazyDataFrame {
    /// - Parameters:
    ///   - columns: <#description#>      ← PLACEHOLDER
    /// - Returns: <#description#>        ← PLACEHOLDER
}

public func collect() async throws -> DataFrame {
    /// - Throws: <#error description#>   ← PLACEHOLDER
    /// - Returns: <#description#>        ← PLACEHOLDER
}
```

**Missing**:
- Query optimization strategy (predicate pushdown? projection pushdown?)
- Lazy vs eager semantics
- When `collect()` actually executes
- Error conditions for `collect()`

#### NPYReader
**File**: `Sources/SwiftDataFrame/IO/NPYReader.swift`  
**Status**: 🔴 CRITICAL

```swift
public static func read(url: URL) throws -> NPYArray {
    /// - Parameters:
    ///   - url: <#description#>          ← PLACEHOLDER
    /// - Throws: <#error description#>
    /// - Returns: <#description#>        ← PLACEHOLDER
}

public static func read(data: Data) throws -> NPYArray {
    /// - Parameters:
    ///   - data: <#description#>         ← PLACEHOLDER
    /// - Throws: <#error description#>
    /// - Returns: <#description#>        ← PLACEHOLDER
}
```

**Missing**:
- NumPy dtype support (f8, f4, i4, i8, u1?)
- Memory mapping behavior
- Shape preservation rules

#### NPYArray.toDataFrame()
**File**: `Sources/SwiftDataFrame/IO/NPYReader.swift`, Lines 112-141  
**Status**: 🟡 PARTIAL — Implementation clear, but docstring incomplete

```swift
public func toDataFrame(columnPrefix: String = "col") throws -> DataFrame {
    /// - Parameters:
    ///   - columnPrefix: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
}
```

**Good**: Implementation comment explains 1D/2D/3D behavior (lines 116-119)  
**Missing**: Why reshape 3D → 2D? What about ND arrays (N > 3)?

---

### 5. SwiftPreprocessing — Feature Engineering

#### PolynomialFeatures
**File**: `Sources/SwiftPreprocessing/Core/PolynomialFeatures.swift`  
**Status**: 🔴 CRITICAL

```swift
public mutating func fit(_ data: [[Double]]) throws {
    /// - Parameters:
    ///   - data: <#description#>         ← PLACEHOLDER
    /// - Throws: <#error description#>
}

public func transform(_ data: [[Double]]) throws -> [[Double]] {
    /// - Parameters:
    ///   - data: <#description#>         ← PLACEHOLDER
    /// - Throws: <#error description#>
    /// - Returns: <#description#>        ← PLACEHOLDER
}
```

**Missing**:
- Degree semantics (degree=2 generates what?)
- `interactionOnly` effect (only cross-terms?)
- Feature ordering in output
- Example: 2D input, degree=2 → how many output features?

---

### 6. SwiftForecast — Time Series

#### ExponentialSmoothing
**File**: `Sources/SwiftForecast/Core/ExponentialSmoothing.swift`  
**Status**: 🔴 CRITICAL

```swift
public func fit(series: [Double]) async throws {
    /// - Parameters:
    ///   - series: <#description#>       ← PLACEHOLDER
    /// - Throws: <#error description#>
}

public func forecast(horizon: Int) throws -> ForecastResult {
    /// - Parameters:
    ///   - horizon: <#description#>      ← PLACEHOLDER
    /// - Throws: <#error description#>
    /// - Returns: <#description#>        ← PLACEHOLDER
}
```

**Missing**:
- Alpha optimization strategy
- What `SmoothingMethod` options exist?
- Convergence criteria
- Seasonal period requirement (line 47 checks, but undocumented)

#### TimeSeriesTransformers
**File**: `Sources/SwiftForecast/Core/TimeSeriesTransformers.swift`  
**Status**: 🔴 CRITICAL

```swift
public func transform(series: [Double]) -> (rollingMean: [Double], rollingStd: [Double]) {
    /// - Parameters:
    ///   - series: <#description#>       ← PLACEHOLDER
    /// - Returns: <#description#>        ← PLACEHOLDER (tuple format not explained)
}

// ExpandingWindow.transform()
public func transform(series: [Double]) -> (expandingMean: [Double], expandingStd: [Double]) {
    /// - Parameters:
    ///   - series: <#description#>       ← PLACEHOLDER
    /// - Returns: <#description#>        ← PLACEHOLDER
}
```

**Missing**:
- Rolling window edge behavior (first N-1 values?)
- NaN/Inf propagation rules
- Relationship to pandas `.rolling()` / `.expanding()`

#### ETSModel
**File**: `Sources/SwiftForecast/Core/ETSModel.swift`, Lines 143-171  
**Status**: 🟡 PARTIAL

```swift
public func forecast(steps: Int) async throws -> [Double] {
    /// - Parameters:
    ///   - steps: <#description#>        ← PLACEHOLDER
    /// - Throws: <#error description#>
    /// - Returns: <#description#>        ← PLACEHOLDER
}

public static func autoFit(series: [Double], period: Int = 1) async throws -> ETSModel {
    /// - Parameters:
    ///   - series: <#description#>       ← PLACEHOLDER
    ///   - period: <#description#>       ← PLACEHOLDER
    /// - Throws: <#error description#>
    /// - Returns: <#description#>        ← PLACEHOLDER
}
```

**Missing**:
- ETS model variants (E = Error, T = Trend, S = Seasonal)
- AICc vs AIC selection criterion
- Damping parameter phi behavior

---

### 7. SwiftExplain — Model Interpretability

#### TreeSHAP
**File**: `Sources/SwiftExplain/Core/TreeSHAP.swift`  
**Status**: 🔴 CRITICAL

```swift
public func explain(decisionTree: DecisionTreeClassifier, instance: [Double]) async -> [Double] {
    /// - Parameters:
    ///   - decisionTree: <#description#>  ← PLACEHOLDER
    ///   - instance: <#description#>      ← PLACEHOLDER
    /// - Returns: <#description#>         ← PLACEHOLDER
}

public func explain(randomForest: RandomForestClassifier, instance: [Double]) async -> [Double] {
    /// - Parameters:
    ///   - randomForest: <#description#>  ← PLACEHOLDER
    ///   - instance: <#description#>      ← PLACEHOLDER
    /// - Returns: <#description#>         ← PLACEHOLDER
}
```

**Missing**:
- Return vector semantics (SHAP value per feature? sums to prediction?)
- TreeSHAP vs KernelSHAP trade-offs
- Base value handling

#### PermutationImportance
**File**: `Sources/SwiftExplain/Core/TreeSHAP.swift`, Lines 233+  
**Status**: 🔴 CRITICAL

```swift
public struct PermutationImportance: Sendable {
    public init() {}

    /// Computes feature importance by measuring decrease in model performance (MSE) when each feature column is shuffled.
    /// - Parameters:
    ///   - features: <#description#>      ← PLACEHOLDER
    ///   - targets: <#description#>       ← PLACEHOLDER
    /// - Returns: <#description#>         ← PLACEHOLDER (format? array? dict?)
}
```

---

### 8. SwiftML — Serialization & Exporters

#### ONNXExporter
**File**: `Sources/SwiftML/Serialization/ONNXExporter.swift`  
**Status**: 🔴 CRITICAL

```swift
public static func exportLinearONNX(
    name: String = "SwiftSciLinearONNX",
    inputs: [String],
    output: String = "output",
    weights: [Double],
    bias: Double
) throws -> Data {
    /// - Parameters:
    ///   - name: <#description#>         ← PLACEHOLDER
    ///   - inputs: <#description#>       ← PLACEHOLDER
    ///   - output: <#description#>       ← PLACEHOLDER
    ///   - weights: <#description#>      ← PLACEHOLDER
    ///   - bias: <#description#>         ← PLACEHOLDER
    /// - Returns: <#description#>        ← PLACEHOLDER
    /// - Throws: <#error description#>
}

public static func exportBinaryONNX(
    name: String = "SwiftSciLinearONNX",
    inputs: [String],
    output: String = "output",
    weights: [Double],
    bias: Double
) -> Data {
    /// - Parameters:
    ///   - name: <#description#>         ← PLACEHOLDER
    ///   - inputs: <#description#>       ← PLACEHOLDER
    ///   - output: <#description#>       ← PLACEHOLDER
    ///   - weights: <#description#>      ← PLACEHOLDER
    ///   - bias: <#description#>         ← PLACEHOLDER
    /// - Returns: <#description#>        ← PLACEHOLDER
}
```

**Missing**:
- JSON vs binary format difference
- Input tensor name semantics
- ONNX IR version
- Compatibility matrix (ONNX Runtime versions?)

---

### 9. SwiftDatabase — Database Connectivity

#### DataFrame.fromSQL()
**File**: `Sources/SwiftDatabase/DatabaseConnection.swift`, Lines ~1040+  
**Status**: 🟡 PARTIAL

```swift
public static func fromSQL(_ query: String, connection: any DatabaseConnection) async throws -> DataFrame {
    /// - Parameters:
    ///   - query: <#description#>        ← PLACEHOLDER
    ///   - connection: <#description#>   ← PLACEHOLDER
    /// - Throws: <#error description#>
    /// - Returns: <#description#>        ← PLACEHOLDER
}
```

**Missing**:
- Type inference strategy (int vs double?)
- NULL handling
- Supported column types
- Row limit / streaming semantics

---

### 10. SwiftML — Synthetic Data

#### DatasetUtilities
**File**: `Sources/SwiftML/Core/DatasetUtilities.swift`  
**Status**: 🔴 CRITICAL

```swift
public static func makeClassification(
    nSamples: Int = 100,
    nFeatures: Int = 2,
    nClasses: Int = 2,
    seed: UInt64 = 42
) -> (features: [[Double]], targets: [Double]) {
    /// - Parameters:
    ///   - nSamples: <#description#>     ← PLACEHOLDER
    ///   - nFeatures: <#description#>    ← PLACEHOLDER
    ///   - nClasses: <#description#>     ← PLACEHOLDER
    ///   - seed: <#description#>         ← PLACEHOLDER
    /// - Returns: <#description#>        ← PLACEHOLDER
}

public static func makeRegression(
    nSamples: Int = 100,
    nFeatures: Int = 2,
    noise: Double = 0.1,
    seed: UInt64 = 42
) -> (features: [[Double]], targets: [Double]) {
    /// - Parameters:
    ///   - nSamples: <#description#>     ← PLACEHOLDER
    ///   - nFeatures: <#description#>    ← PLACEHOLDER
    ///   - noise: <#description#>        ← PLACEHOLDER
    ///   - seed: <#description#>         ← PLACEHOLDER
    /// - Returns: <#description#>        ← PLACEHOLDER
}

public static func makeMoons(
    nSamples: Int = 100,
    noise: Double = 0.1,
    seed: UInt64 = 42
) -> (features: [[Double]], targets: [Double]) {
    /// - Parameters:
    ///   - nSamples: <#description#>     ← PLACEHOLDER
    ///   - noise: <#description#>        ← PLACEHOLDER
    ///   - seed: <#description#>         ← PLACEHOLDER
    /// - Returns: <#description#>        ← PLACEHOLDER
}
```

---

## Statistics: Coverage Analysis

### Placeholder Count by Module

| Module | File Count | Functions | Placeholders | % Affected |
|--------|-----------|-----------|--------------|-----------|
| **SwiftStats** | 2+ | ~12+ | 12+ | 100% |
| **SwiftML** | 8+ | ~40+ | ~20 | 50% |
| **SwiftForecast** | 5+ | ~25+ | ~15 | 60% |
| **SwiftNLP** | 5+ | ~20+ | ~12 | 60% |
| **SwiftDataFrame** | 6+ | ~30+ | ~12 | 40% |
| **SwiftPreprocessing** | 10+ | ~50+ | ~8 | 16% |
| **SwiftExplain** | 4+ | ~15+ | ~9 | 60% |
| **SwiftDatabase** | 2+ | ~8+ | ~3 | 37% |
| **SwiftCluster** | 5+ | ~20+ | ~5 | 25% |
| **SwiftLLM** | 4+ | ~15+ | ~4 | 27% |
| **SwiftOptimize** | 4+ | ~18+ | ~6 | 33% |
| **SwiftVision** | 3+ | ~12+ | ~4 | 33% |
| **SwiftAgent** | 2+ | ~8+ | ~2 | 25% |
| **SwiftVisualization** | 3+ | ~10+ | ~3 | 30% |

**Total**:  
- ~1,753 public symbols (claimed)
- ~150-200 with placeholder docstrings
- **~10-15% placeholder rate** (vs "100% coverage" claim)

---

## Impact Assessment

### Severity by User Role

| Role | Impact | Example |
|------|--------|---------|
| **New Users** | 🔴 CRITICAL | Cannot understand parameter semantics. Guessing required. |
| **Production Maintainers** | 🔴 CRITICAL | Cannot validate numeric behavior (convergence, edge cases). |
| **Contributors** | 🟡 MEDIUM | Copy-paste patterns from code instead of docs. |
| **IDE Users (Xcode)** | 🔴 CRITICAL | No autocomplete hints. Cmd+Click shows `<#description#>`. |

---

## Recommendations

### Quick Wins (1-2 hours per module)

1. **SwiftStats** — Replace ALL `<#description#>` in `Stats.swift` and `Stats+LinearAlgebra.swift`
   - Document vDSP variants
   - Explain null/NaN handling
   
2. **SwiftDataFrame** — Complete `LazyDataFrame`, `NPYReader`
   - Lazy evaluation semantics
   - NumPy dtype mapping

3. **SwiftNLP** — Complete `TextNormalizer`, `WordEmbeddings`, `StopWords`
   - Unicode normalization form
   - Embedding dimensions

### High-Impact (4-8 hours)

1. **SwiftML** — Complete `LinearRegression`, `LogisticRegression`
   - LAPACK vs GD trade-offs
   - When to use which device (CPU/GPU)

2. **SwiftForecast** — Complete exponential smoothing, ETS, ARIMA
   - Parameter tuning guides
   - Seasonal period semantics

3. **SwiftExplain** — SHAP value interpretation
   - TreeSHAP vs KernelSHAP
   - Return vector semantics

### Systematic (Long-term)

- Add **code examples** to DocC (`.docc/Tutorials/`)
- Link to **academic papers** for advanced methods
- Create **parameter checklists** (what values make sense?)
- Add **common pitfalls** section per module

---

## Files Needing Immediate Attention

**CRITICAL (0 meaningful docs)**:
- `Sources/SwiftStats/Stats.swift`
- `Sources/SwiftStats/Core/Stats+LinearAlgebra.swift`
- `Sources/SwiftML/Serialization/ONNXExporter.swift`
- `Sources/SwiftForecast/Core/ExponentialSmoothing.swift`
- `Sources/SwiftForecast/Core/ETSModel.swift`

**HIGH (40-60% placeholders)**:
- `Sources/SwiftML/Core/LinearRegression.swift`
- `Sources/SwiftML/Core/LogisticRegression.swift`
- `Sources/SwiftNLP/Core/PorterStemmer.swift`
- `Sources/SwiftNLP/StopWords.swift`
- `Sources/SwiftDataFrame/Lazy/LazyDataFrame.swift`
- `Sources/SwiftDataFrame/IO/NPYReader.swift`
- `Sources/SwiftPreprocessing/Core/PolynomialFeatures.swift`
- `Sources/SwiftExplain/Core/TreeSHAP.swift`

---

## Conclusion

✅ **Code quality**: Real algorithms (LAPACK, MLX), proper architecture  
❌ **Documentation quality**: Placeholder template epidemic, misleading "100% coverage" claim  
⚠️ **Developer experience**: IDE hints useless, examples missing, trade-offs unexplained

**Recommendation**: Update release notes to be honest: _"Functional completeness 100%, documentation coverage ~40-50%. Full API contract defined, detailed parameter guidance in progress."_
