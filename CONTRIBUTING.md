# Contributing to SwiftSci

Thank you for your interest in contributing to **SwiftSci**! SwiftSci is a high-performance, modular Swift 6 data science and machine learning framework designed for Apple Silicon.

## Code of Conduct

Please review and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) in all community interactions.

## Getting Started

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/Nodibell/SwiftSci.git
   cd SwiftSci
   ```

2. **Build and Test:**
   ```bash
   swift build -c release
   swift test
   ```

3. **Check Swift 6 Concurrency:**
   Ensure all target modifications adhere strictly to Swift 6 strict concurrency checks without warnings or data races (`-strict-concurrency=complete`, `@Sendable`, `actor`, thread-safe data structures with `NSLock`).

## Development Guidelines

- **Architecture:** Keep target responsibilities clean across the 14 core modules:
  `SwiftDataFrame`, `SwiftStats`, `SwiftPreprocessing`, `SwiftML`, `SwiftCluster`, `SwiftNLP`, `SwiftOptimize`, `SwiftForecast`, `SwiftLLM`, `SwiftExplain`, `SwiftVisualization`, `SwiftVision`, `SwiftDatabase`, and `SwiftAgent`.
- **Performance:** Leverage Apple Accelerate (`vDSP`, `LAPACK`, `BLAS`), `MLX`, or Apple Silicon Metal GPU shaders where appropriate for vectorized compute.
- **Zero-Copy & Memory Safety:** Use `PreparedNumericMatrix` and consuming transformations (`transform(consuming:)`) for tabular-to-matrix pipelines to eliminate intermediate heap allocations.
- **Documentation:** Every public API method, struct, enum, and class must be documented using Swift DocC comments (`///`).
- **Tests:** Add unit tests under `Tests/<Target>Tests` for every new algorithm or feature.
- **Benchmarks & Validation:** For performance or numerical changes, run standardized checks in `Benchmarks/` using `bench.py` to ensure numerical parity against reference libraries (NumPy, Scikit-Learn, SciPy).

### Swift type checking in tests

Keep generated numeric fixtures in small, explicitly typed expressions. Give nested
collection closures concrete return types, bind arithmetic to typed intermediate
values, and build arrays before passing them to overloaded initializers. Reuse a
typed fixture helper when tests need the same data. Do not reduce fixture sizes or
change numeric precision to work around a compiler diagnostic.

Compile test targets as part of local validation. `swift build` does not compile
them, and `swift test --skip` still compiles skipped suites. A successful build on
a newer Swift compiler does not establish compatibility with the CI compiler.

## Architectural & Concurrency Design Criteria

To maintain consistency in SwiftSci 3.11.0, all new types must adhere to these architectural criteria:

### 1. Concurrency Tier Decision Matrix

| Tier | Type Construct | Criteria & Use Cases | Examples |
|---|---|---|---|
| **Tier A** | `actor` | Stateful ML & Forecast models maintaining internal mutable arrays or multi-pass optimization state across concurrent tasks. | `KMeans`, `DBSCAN`, `ExponentialSmoothing`, `ARIMA`, `AutoARIMA` |
| **Tier B** | `struct` (`Sendable`) | Pure value-semantics preprocessing transformers, data containers, value-typed columns, and configuration objects. Value semantics guarantee data-race freedom. | `TypedColumn`, `DataFrame`, `PreparedNumericMatrix`, `MinMaxScaler`, `StandardScaler`, `RobustScaler`, `Normalizer`, `VarianceThreshold`, `SelectKBest`, `Imputer`, `KBinsDiscretizer`, `KNNImputer`, `PowerTransformer`, `TargetEncoder` |
| **Tier C** | `final class` (`@unchecked Sendable` with `NSLock`) | Reference-type container pipelines, streaming readers, or stateful tokenizers/vectorizers requiring internal thread synchronization. | `CountVectorizer`, `TFIDFVectorizer`, `Pipeline`, `ColumnTransformer`, memory buffer wrappers |

### 2. Model Hierarchy Guidelines: `Estimator` vs `PreprocessingTransformer`

- **`Estimator` Protocol**: Used for predictive machine learning & time-series models (`SwiftML`, `SwiftCluster`, `SwiftForecast`).
  - Core contract: `fit(data:)`, `predict(data:)`.
  - Typical construct: `actor` (stateful fitting) or immutable evaluated models.
- **`PreprocessingTransformer` & `ConsumingPreprocessingTransformer`**: Used for feature transformation and data pipeline operations (`SwiftPreprocessing`, `SwiftDataFrame`).
  - Core contract: `fit(data:)`, `transform(data:)`, `transform(consuming:)`, `fitTransform(data:)`.
  - Typical construct: `struct` (immutable value-type transformations producing new `DataFrame` or in-place `PreparedNumericMatrix` mutations).

## Submitting Pull Requests

1. Create a feature branch (`git checkout -b feature/my-new-algorithm`).
2. Verify all unit tests pass with zero warnings (`swift test`).
3. If modifying performance-critical or mathematical paths, verify with `Benchmarks/Tools/bench.py`.
4. Commit changes with clear, descriptive commit messages adhering to Conventional Commits (`feat:`, `fix:`, `perf:`, `test:`).
5. Push to your branch and open a Pull Request targeting `main`.

Thank you for helping make Swift data science robust, fast, and native!
