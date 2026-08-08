# Contributing to SwiftSci

Thank you for your interest in contributing to **SwiftSci**! SwiftSci is a high-performance, modular Swift 6 data science and machine learning framework designed for Apple Silicon.

## Code of Conduct

Please review and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) in all community interactions.

## Getting Started

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/swiftsci/SwiftSci.git
   cd SwiftSci
   ```

2. **Build and Test:**
   ```bash
   swift build
   swift test
   ```

3. **Check Swift 6 Concurrency:**
   Ensure all target modifications adhere strictly to Swift 6 strict concurrency checks without warnings or data races (`@Sendable`, `actor`, thread-safe data structures).

## Development Guidelines

- **Architecture:** Keep target responsibilities clean across `SwiftDataFrame`, `SwiftStats`, `SwiftML`, `SwiftCluster`, `SwiftNLP`, `SwiftForecast`, `SwiftLLM`, etc.
- **Performance:** Leverage Apple Accelerate (`vDSP`, `LAPACK`) or `MLX` where appropriate for vectorized compute.
- **Documentation:** Every public API method, struct, enum, and class must be documented using Swift DocC comments (`///`).
- **Tests:** Add unit tests under `Tests/<Target>Tests` for every new algorithm or feature.

## Architectural & Concurrency Design Criteria

To maintain consistency before freezing the SwiftSci 3.0 API, all new types must adhere to these architectural criteria:

### 1. Concurrency Tier Decision Matrix

| Tier | Type Construct | Criteria & Use Cases | Examples |
|---|---|---|---|
| **Tier A** | `actor` | Stateful ML & Forecast models maintaining internal mutable arrays or multi-pass optimization state across concurrent tasks. | `KMeans`, `DBSCAN`, `ExponentialSmoothing`, `ARIMA` |
| **Tier B** | `struct` (`Sendable`) | Pure value-semantics preprocessing transformers, data containers, value-typed columns, and configuration objects. Value semantics guarantee data-race freedom. | `TypedColumn`, `DataFrame`, `MinMaxScaler`, `StandardScaler`, `RobustScaler`, `Normalizer` |
| **Tier C** | `final class` (`@unchecked Sendable`) | Reference-type container pipelines (`Pipeline`, `ColumnTransformer`) or wrappers for C-pointers/system APIs. Requires internal lock/synchronization or explicit design rationale. | `Pipeline`, `ColumnTransformer`, memory buffer wrappers |

### 2. Model Hierarchy Guidelines: `Estimator` vs `PreprocessingTransformer`

- **`Estimator` Protocol**: Used for predictive machine learning & time-series models (`SwiftML`, `SwiftCluster`, `SwiftForecast`).
  - Core contract: `fit(data:)`, `predict(data:)`.
  - Typical construct: `actor` (stateful fitting).
- **`PreprocessingTransformer` Protocol**: Used for feature transformation and data pipeline operations (`SwiftPreprocessing`, `SwiftDataFrame`).
  - Core contract: `fit(data:)`, `transform(data:)`, `fitTransform(data:)`.
  - Typical construct: `struct` (immutable value-type transformations producing new `DataFrame` instances).

## Submitting Pull Requests

1. Create a feature branch (`git checkout -b feature/my-new-algorithm`).
2. Verify all unit tests pass (`swift test`).
3. Commit changes with clear, descriptive commit messages.
4. Push to your branch and open a Pull Request targeting `main`.

Thank you for helping make Swift data science robust, fast, and native!
