# SwiftSci 2.0 — Implementation Plan

## Executive Summary 

SwiftSci v1.7 already provides a comprehensive *modular* data science library: `SwiftDataFrame` (zero-copy columnar data on Arrow), `SwiftStats` (vectorized stats with Accelerate), `SwiftML` (regressions, trees, neural nets), `SwiftNLP`, `SwiftForecast`, `SwiftExplain` (KernelSHAP), `SwiftVisualization`, etc.  Version 2.0 will **solidify the architecture** (stable public APIs, formal error and concurrency models, extension/plugin interfaces) and **fill in missing features** (lazy/eager DataFrame 2.0, advanced I/O formats, workflow/AutoML, hyperparameter engines, explainability tools, model persistence, hardware scheduling, NLP/forecasting enhancements, validation, reporting, etc.).  

This plan lays out: the **architecture and design specification** (modules, dependencies, protocols, types, error/memory/concurrency models, coding and performance standards); a **detailed feature roadmap** (with priorities, deliverables, timelines, effort estimates, code examples); **data schemas and file format specifications**; backward-compatibility and deprecation policies; **testing and benchmarking** strategies; **security and risk** considerations; **milestones** (alpha/beta/RC/release) with clear deliverables; and a **maintenance checklist**.  Wherever possible we draw on primary sources (the SwiftSci repo, Apple’s docs on Accelerate/Metal/MLX/CoreML, Apache Arrow/Parquet/Feather specs, ONNX spec) to guide our design and cite factual assertions.  

SwiftSci is built to leverage Apple Silicon’s unified-memory and strict-concurrency model, combining CPU vectorization (Accelerate) with GPU/ANE (via MLX/Metal).  Its v1.x highlights include **data-oriented design** for tree ensembles and **device routing** (CPU for branch-heavy tasks, GPU for large-tensor tasks).  We will carry these principles forward and extend them.  The 2.0 release criteria include API stability, full test coverage, performance parity, documentation, and community uptake. A short **maintainer checklist** is provided at the end to track progress across modules and quality gates.  

## Architecture & Design Specification

We begin by **freezing the architecture** of SwiftSci to avoid accumulating technical debt.  The library will be organized into clearly defined modules with no circular dependencies, using protocol-oriented design and Swift 6 concurrency.  We will formalize all public interfaces and provide documentation.  

### Module Responsibilities 

SwiftSci 2.0 will consist of these core modules (some of which already exist in 1.7): 

- **`SwiftCore`** (proposed): Common low-level utilities, numeric primitives, random number generators, linear algebra kernels (wrapping Accelerate), memory/buffer types, and core protocols.  
- **`SwiftDataFrame`**: Columnar data frames built on Apache Arrow. Provides in-memory tables with *zero-copy* slicing and streaming CSV/JSON readers. Supports joins, pivots, melts, and conversion to feature/target matrices. (v1.x already had “built on Arrow” zero-copy DataFrame).  
- **`SwiftStats`**: Vectorized descriptive stats (mean, var, distribution fitting, tests) powered by Accelerate (vDSP/BLAS).  
- **`SwiftPreprocessing`**: Feature transformers: scaling, encoders (`OneHotEncoder`, `OrdinalEncoder`, `TargetEncoder`, `FrequencyEncoder`), imputers (`Imputer`, `KNNImputer`), selectors, and data pipelines (`Pipeline`, `ColumnTransformer`).  
- **`SwiftML`**: Supervised learning algorithms: Linear/Logistic Regression, Decision Trees, Random Forests, Gradient-Boosted Trees, Multilayer Perceptrons, support for generating synthetic data, and tracking feature importances (Gini, etc). Models implement `Codable` for (legacy) persistence.  
- **`SwiftCluster`**: Unsupervised learning: PCA (SVD-based), DBSCAN, Isolation Forest, Local Outlier Factor, K-Means (with K-Means++).  
- **`SwiftOptimize`**: Model selection and evaluation: Cross-validation (`KFold`), metrics (ROC-AUC, precision, recall, F1, etc), `GridSearchCV`, `RandomizedSearchCV`, and (2.0) new hyperparameter search strategies (Bayesian, Hyperband).  
- **`SwiftForecast`**: Time-series forecasting: Holt-Winters, ARIMA/SARIMA, GARCH, Kalman Filters, expanding window aggregations, seasonal decomposition (additive/multiplicative).  
- **`SwiftNLP`**: Text processing: tokenization, embeddings, vectorizers. Includes stop-word lists, text normalization, Byte-Pair Encoding tokenizers, N-gram tokenizers, `HashingVectorizer`, `TFIDFVectorizer`, and pretrained embeddings support.  
- **`SwiftExplain`**: Model explainability: black-box explainers (currently KernelSHAP) and related tools. (We will add Permutation Importance, Partial Dependence, ICE plots, and automatic reports.).  
- **`SwiftLLM`**: Local large language model interface: GPU-accelerated transformer generation (causal decoder), with support for SafeTensors and GGUF formats.  
- **`SwiftVisualization`**: Reporting and plotting: Export charts (HTML/JS or image) for correlation heatmaps, ROC curves, feature importance, confusion matrices, etc.  
- **`SwiftVision`** (proposed for Saura migration): Computer vision pipeline for image classification, segmentation (U-Net with Dice/IoU metrics), object detection (YOLO/CoreML wrappers), CNN feature extractors, and `.npz`/binary array dataset loading.  
- **`SwiftDatabase`** (proposed for Saura migration): Direct relational database connectors (SQLite, PostgreSQL, MySQL) enabling SQL query execution straight into zero-copy `SwiftDataFrame` instances without intermediate files.  
- **`SwiftAgent`** (proposed for Saura migration): Safe Swift dynamic execution context & REPL sandbox for local LLMs, replacing Python-based agentic code execution loops (`<execute_python>`).  
- **Third-party adapters (plugins)**: e.g. `SwiftONNX`, `SwiftCoreML`, `SwiftArrow`, etc, will be optional modules.  

Each module’s responsibilities will be documented in an **architecture overview**. For example, `SwiftDataFrame` is responsible for tabular data I/O and manipulation (relying on Arrow), while `SwiftML` implements algorithms that consume feature matrices and target vectors.  We will create **ARCHITECTURE.md** detailing each module’s role, its public API, and allowed dependencies.  

### Dependency Graph 

Modules must form a DAG. No circular imports. In general higher-level modules depend on lower-level:

```
SwiftCore
│
├─ SwiftDataFrame      (depends on SwiftCore)
│    ├─ (formats: CSV/Arrow/Parquet, etc)
│    └─ SwiftDatabase  (depends on SwiftDataFrame)
├─ SwiftStats          (depends on SwiftCore)
├─ SwiftPreprocessing  (depends on SwiftDataFrame, SwiftStats)
├─ SwiftML             (depends on SwiftDataFrame, SwiftStats, SwiftPreprocessing)
├─ SwiftCluster        (depends on SwiftDataFrame, SwiftStats)
├─ SwiftOptimize       (depends on SwiftML, SwiftStats)
├─ SwiftForecast       (depends on SwiftDataFrame, SwiftStats)
├─ SwiftNLP            (depends on SwiftDataFrame, SwiftStats)
├─ SwiftExplain        (depends on SwiftML, SwiftStats)
├─ SwiftLLM            (depends on SwiftNLP, SwiftML)
├─ SwiftVision         (depends on SwiftCore, SwiftML)
├─ SwiftAgent          (depends on SwiftLLM, SwiftDataFrame)
└─ SwiftVisualization  (depends on SwiftDataFrame, SwiftML)
```

```mermaid
graph TD
    SwiftCore --> SwiftDataFrame
    SwiftCore --> SwiftStats
    SwiftDataFrame --> SwiftDatabase
    SwiftDataFrame --> SwiftPreprocessing
    SwiftStats --> SwiftPreprocessing
    SwiftDataFrame --> SwiftCluster
    SwiftCore --> SwiftCluster
    SwiftStats --> SwiftCluster
    SwiftDataFrame --> SwiftML
    SwiftStats --> SwiftML
    SwiftPreprocessing --> SwiftML
    SwiftML --> SwiftOptimize
    SwiftStats --> SwiftOptimize
    SwiftDataFrame --> SwiftOptimize
    SwiftDataFrame --> SwiftForecast
    SwiftStats --> SwiftForecast
    SwiftDataFrame --> SwiftNLP
    SwiftStats --> SwiftNLP
    SwiftNLP --> SwiftLLM
    SwiftML --> SwiftExplain
    SwiftStats --> SwiftExplain
    SwiftCore --> SwiftVision
    SwiftML --> SwiftVision
    SwiftLLM --> SwiftAgent
    SwiftDataFrame --> SwiftAgent
    SwiftDataFrame --> SwiftVisualization
    SwiftML --> SwiftVisualization
```

In this graph, `SwiftCore` is the lowest level. UI frameworks (AppKit/UIKit) are explicitly **not** allowed as dependencies; only Foundation, Swift System, and Apple framework libraries (Accelerate, Metal, etc.) are used. We will produce a *module dependency diagram* (e.g. in `docs/ARCHITECTURE.md`) and a textual list of dependencies for each module.  

### Public Protocols & APIs 

We will define a set of **core Swift protocols** as interfaces that algorithms and utilities conform to, ensuring consistency. Example protocols include: 

- `Estimator`: objects that can be **trained** on data (with methods like `fit(features:target:)`) and **predict**.  
- `Predictor`: objects that can produce predictions from features (with `predict(features:)`).  
- `Transformer`: data transformers with `fitTransform` / `transform`.  
- `Clusterer`, `Optimizer`, `Metric`, `Explainer`, `ForecastModel`, `Tokenizer`, `Vectorizer`, `Dataset`, `PipelineStage`, etc.  

Each algorithm (regressor, classifier, transformer, etc.) will implement the appropriate protocol(s).  For example, a `LinearRegression` struct will conform to `Estimator` and `Predictor`, while a `OneHotEncoder` conforms to `Transformer`.  We will document these protocols (names, methods, semantics) in the design spec (`docs/ARCHITECTURE.md`).  

### Shared Data Types 

We will standardize commonly used data structures across modules. Examples include: 

- **`PredictionResult`** / **`RegressionPrediction`** / **`ClassificationPrediction`**: structs to hold model output (predicted value or class label, probability distribution if any).  
- **`ProbabilityDistribution`**: represents outputs like softmax probabilities.  
- **`FeatureImportance`, `SHAPValues`**: types for explainability outputs.  
- **`EvaluationReport`**: holds metrics and confusion matrix for model evaluation.  
- **`TrainingReport`**: summarizes training info (loss curves, runtime).  
- **`DatasetMetadata` / `ModelMetadata`**: for data schema, column types, target info, model hyperparameters, training date, author, etc.  
- **`FeatureSchema`**: describes feature names, types (numerical, categorical), and encoders.  
- **`PredictionResult`**: final prediction combined with metadata.  

These types ensure that all modules can interoperate. For instance, `SwiftDataFrame.toFeatureMatrix()` might return a standardized `Matrix` and column schema object.  We will create a section (and likely a Swift file, e.g. `SwiftCore/Interfaces.swift`) documenting these types.  

### Error Model 

All modules will use a unified error hierarchy rooted at `SwiftSciError` (or similar). Subclasses may include `DataError`, `TrainingError`, `PredictionError`, `ValidationError`, `IOError`, `HardwareError`, `PipelineError`, etc. No module should throw raw `String` errors or define independent error enums.  This ensures calling code can catch `SwiftSciError` and inspect the code. We will define the error types and ensure that every thrown or fatal error uses them.  

### Memory Model 

SwiftSci will leverage **copy-on-write (COW)** and Arrow’s zero-copy buffers. For example, `DataFrame` operations should share memory with Arrow arrays whenever possible (no unnecessary copies). We will document conventions: 

- **Immutable core data**: once read into a `DataFrame`, the underlying Arrow buffers are immutable; transformations produce new views or buffers.  
- **Zero-copy APIs**: CSV/JSON/Parquet readers use memory-mapped or streaming I/O (as already in 1.x) to avoid extra copies.  
- **Mutable Builders**: the only time data is copied is during explicit `.toArrow()` or when modifying schema.  
- **Memory mapping**: support mmapped CSV/Arrow, similar to `SystemsCSVParser` from v1.x for huge files.  

This will be detailed in `docs/MEMORY_MODEL.md`.  

### Concurrency Model 

SwiftSci will be fully **Swift Concurrency**-compliant (as noted in v1.x). We will adopt Swift 6’s strict concurrency rules: 

- All public types (models, DataFrame, etc.) will be `Sendable` where possible.  
- Internal shared state will be avoided; mutable state will be confined to actors or thread-local structures.  
- Parallelism will use `Task` and structured concurrency for async I/O and training.  
- We may introduce actors for global schedulers or resource managers.  
- No use of unsafe threading primitives; no shared mutable global caches.  

In short, only `async/await`, `Task`, and actors will be used for concurrency. This aligns with the existing design (“strict concurrency requirements”) and ensures safety on Apple Silicon.

### Extension and Plugin Architecture 

SwiftSci will allow **extension** by third parties without modifying core code. This means: 

- **Custom Estimators/Transformers**: Users can implement our protocols in their own code; modules should accept any conforming types (e.g. a function `fitPipeline(_ stages: [Estimator])`).  
- **Plugin System**: We will define a plugin API so that external modules (in separate packages) can register new algorithms, file formats, data sources, etc. For example, a user could write a `SwiftPlugin` package that adds an `XGBoostClassifier` or a `ParquetDataset`. The plugin API might use Swift’s new plugin/package manifest hooks or a dynamic registry in code.  
- **Optional Dependencies**: Integrations like ONNX, CoreML, or cloud I/O will be optional Swift packages that can depend on `SwiftSci` but not vice versa.  

We will provide guidance in `docs/EXTENSION.md` on how to add functionality via plugins or package dependencies, ensuring that core remains modular.

### Coding Standards 

We will document style and API guidelines in `docs/API_GUIDELINES.md`. Key rules include: 

- **Naming**: Swift API design guidelines (PascalCase types, camelCase methods, no Hungarian).  
- **Documentation**: All public APIs must have DocC documentation comments.  
- **Generics**: Use generic constraints for algorithms (e.g. data types conforming to `Numeric`).  
- **Error Handling**: Use `throws` with our unified `SwiftSciError` types; avoid crashing for user errors.  
- **Thread Safety**: Annotate `Sendable`, no unsafely shared state.  
- **Performance Comments**: For heavy algorithms, note Big-O and use vectorized calls.  

We will also adopt an API versioning policy: stable APIs (e.g. training loops, estimator interfaces) may only change with semver major bump (2.0). Deprecations get warnings in v1.x, then removal in 2.x.  A `CHANGELOG.md` will document breaking changes.

### Performance Principles 

Every new algorithm or data structure must meet performance standards: 

- **SIMD/Vectorization**: Prefer Accelerate/vDSP or Swift SIMD over manual loops.  
- **Zero-copy**: Avoid unnecessary allocations (use Arrow buffers, memory-map files).  
- **MLX/Metal**: For large matrix/tensor ops, route to Apple GPU (via MLX) or neural engines.  
- **Benchmark-first**: Benchmarks for new features must exist (see `Benchmarks/`). Each feature has target throughput or latency (e.g. “CSV read of 1M rows in <50ms”, “RandomForest training on 10k×100 in <100ms”).  
- **Memory Complexity**: Document memory use (e.g. O(N) vs O(N²)).  

For example, SwiftSci v1.x already switched to data-oriented trees (storing tree nodes in flat arrays) to improve cache locality, and uses an optional `checkNaN` bypass to eliminate overhead in hot loops. We’ll enforce similar innovations: e.g. using Metal Performance Shaders or MLX for GPU-accelerated math. 

All these design principles will be codified in `docs/PERFORMANCE.md`, and a **design review checklist** will require answers to questions like “Is this algorithm `Sendable`? Is it benchmarked?” etc.

## Feature Roadmap

We prioritize features by production impact. The initial v2.0 focus is on API stability and core completeness (DataFrame enhancements, storage, workflow, explainability, persistence), with secondary emphasis on advanced engines (AutoML, hardware scheduling, NLP/forecasting). We break down each major feature area with its priority, milestones, and tasks. Where helpful, we provide Swift API sketches, data schema outlines, and format specs. 

### Stable API (Priority: Critical) 

**Goal:** Finalize a stable public API interface for all core modules. 

- **Tasks:** 
  - Audit all public types/methods and decide which are stable vs experimental. Deprecate or remove unstable APIs from v1.x.  
  - Define final signatures for key protocols (`Estimator`, `Transformer`, etc.) and data structures (`DataFrame`, `Model`, `Dataset`, etc.).  
  - Ensure naming consistency and clarity (per API guidelines doc).  
  - Write `docs/API_GUIDELINES.md` describing design patterns.  
  - Freeze version 2.0 semantic versioning: new features only added behind clearly marked extensions; incompatible changes only at major bump.  
- **API Sketch:** For example, an API for training might look like:  
  ```swift
  public protocol Estimator: Sendable {
      associatedtype ModelType: Predictor
      func fit(features: DataFrame, target: String) async throws -> ModelType
  }
  public struct LinearRegression: Estimator {
      public func fit(features: DataFrame, target: String) async throws -> RegressionModel { … }
  }
  ```  
- **Backward Compatibility:** We will provide deprecation warnings in 1.x and compatibility shims (e.g. alias old method names to new) where possible, to ease migration. A compatibility guide will be included.  
- **Testing:** Compile tests ensuring old v1.7 code still builds against v2.0 (migration tests), plus unit tests for new protocol conformance.  
- **Benchmarks:** None specifically (API stability is about design, not performance).  
- **Security/Privacy:** Not directly applicable.  
- **Risk/Mitigation:** Scope creep: enforce that this stage is API/design-only (no heavy new features) to avoid delays.  

### DataFrame 2.0 (Priority: High) 

**Goal:** Implement advanced DataFrame functionality: lazy evaluation, expression API, query optimizer, window functions, SQL-like API.  

- **Tasks:**  
  - Add a lazy `DataFrame` mode: operations like filter/group/join can build a query plan without immediate execution. This enables optimizations (predicate pushdown, column pruning).  
  - Implement an **Expression DSL** for filters and computed columns, allowing SQL-like syntax: e.g. `df.filter($"age" > 30 && $"country" == "US")`.  
  - Introduce **window functions**: rolling mean, cumulative sum, etc.  
  - Improve `DataFrame.join` with index/hash join strategies.  
  - Extend SQL/JSON: consider a `DataFrame.sql(query: String)` interface for SQL queries.  
  - Convert existing eager ops to use the new lazy engine under the hood.  
- **API Sketch:**  
  ```swift
  let df = try await DataFrame(csv: "data.csv")
  let df2 = df.lazy
      .filter(Expression.column("age") > 30 && Expression.column("income") < 100_000)
      .groupBy("country")
      .agg(mean: "income", count: "*")
      .orderBy("mean_income", descending: true)
      .execute()
  ```  
  or using a QueryBuilder:  
  ```swift
  let query = DataFrameQuery()
      .from("data.csv")
      .select(["age", "income", "country"])
      .where(Col("age") > 30 && Col("income") < 100_000)
      .groupBy("country").agg(mean: "income", count: true)
  let resultDF = try query.run()
  ```  
- **Data Schema:** A lazy query has a schema (columns/types) determined by the plan. Final schema after aggregation must be well-defined. We will formalize a schema type to track names and Arrow data types through the plan.  
- **File Format:** Input formats handled by `SwiftDataFrame` (see Storage section). Lazy mode may involve an IR for execution but no new on-disk format.  
- **Backward Compatibility:** Existing code using eager APIs (`df.filter`, `df.groupBy`) should still work; under the hood they will call the new engine or fallback to eager execution. We may mark new functions as `@available(iOS, deprecated:)` to signal the change.  
- **Testing:** Unit tests comparing results of lazy vs eager execution on sample datasets (groupBy, join, window). Validate correctness on edge cases (nulls, empty groups).  
- **Benchmarks:** Compare execution time and memory for standard operations on large tables (e.g. 1M rows). Ensure lazy queries can match or beat the old eager version by combining filters/join effectively.  
- **Security/Privacy:** None specific (data remains in-memory). Just ensure no arbitrary code execution via expression API (expressions must only refer to columns/operators, no string interpolation).  
- **Risks/Mitigation:** Complexity of query optimizer (risk of bugs, slow planning). Mitigation: start with simple pushdown and rule-based optimizations, avoid over-generality in v2.0. Possibly leverage existing libraries (e.g. reuse some logic from DuckDB or use [Apache Calcite ideas](https://calcite.apache.org/) as inspiration).  

### Storage Layer (Priority: High) 

**Goal:** Support a rich set of storage backends: CSV/JSON (existing), plus **Apache Arrow (Feather)**, **Parquet**, and streaming I/O. Ensure zero-copy reads where possible and efficient writes with compression.  

- **Tasks:**  
  - **CSV/JSON:** Improve existing parsers. Ensure memory mapping and streaming (already in v1.x for CSV). Add JSON parsing into `DataFrame` with schema inference.  
  - **Arrow IPC / Feather**: Add reader/writer for Arrow file format (`.arrow` or `.feather`). These are in-memory columnar formats. Use Apache Arrow C++ library via Swift (perhaps via [C Arrow bindings](https://arrow.apache.org/docs/swift/index.html)). Arrow IPC is essentially “Feather V2”, offering very fast I/O.  
  - **Parquet**: Add Parquet read/write, using Arrow’s Parquet library. Parquet is a compressed columnar format ideal for large datasets.  
  - **Streaming/Chunked I/O**: For large datasets, support reading in chunks (`for batch in DataFrame.readCSVStreaming(...) { ... }`). Ensure groupBy and similar ops can process in stream-friendly ways.  
  - **Zero-copy**: Where possible (e.g. memory-mapped Arrow), avoid copying data into new buffers.  
  - **Format Abstraction**: Implement a `DataFrameReader` and `DataFrameWriter` interface so adding new formats is easy.  
- **API Sketch:**  
  ```swift
  // Read Arrow/Feather file
  let df = try await DataFrame(arrowFile: "data.arrow")
  // Or from Parquet
  let df = try await DataFrame(parquet: "data.parquet")
  // Or write DataFrame to Parquet
  try await df.toParquet("output.parquet", compression: .gzip)
  ```  
- **File Format Specs:**  
  - *CSV/JSON*: as before (comma-separated, UTF-8, JSON-lines or array).  
  - *Arrow/Feather*: per [Apache Arrow IPC spec] – essentially raw Arrow memory with optional LZ4/ZSTD compression. All Arrow data types supported.  
  - *Parquet*: per [Apache Parquet spec] – columnar, typed with encodings (dictionary, RLE, etc) and compression (Snappy, Gzip, etc).  
- **Backward Compatibility:** Existing `DataFrame(csv:)` and `df.toCSV()` remain. The new methods are additive. We will deprecate any old ad-hoc CSV functions (if any). Old code must still build.  
- **Testing:**  
  - **Round-trip tests**: Write a DataFrame to each format and read it back, checking all values and schema match.  
  - **Large file tests**: Read a multi-gigabyte CSV/Parquet in streaming mode, ensure memory stays bounded.  
  - **Edge cases**: missing values, mixed types, nested JSON.  
- **Benchmarks:**  
  - Compare read/write speeds: e.g. 1M rows, 10 cols. Expect Arrow/Feather to be fastest (almost pure memory copy), Parquet slower but smaller size. As one benchmark found, Feather (zstd) can be ~20× faster to write than CSV and ~6× faster to read, while Parquet(gzip) yields ~2× faster read and ~78% smaller size. (CSV should still be supported for legacy use, though slow.)  
  - Document target: e.g. Parquet read <5s for 10M rows, CSV read <30s for 1M rows on M-series (actual targets to be measured).  
- **Security/Privacy:** Ensure Parquet/Arrow reading respects file permissions; validate untrusted input (no code execution). If supporting Parquet encryption, note it for future (Parquet has optional encryption, not initial requirement).  
- **Risks:**  
  - **Library integration**: Linking Arrow/Parquet C++ libs into Swift can be complex (build/cross-compile issues). Mitigation: use SwiftPM binary targets or encourage pre-built libs.  
  - **Version compatibility**: Different Parquet versions might break; write robust open/readers.  
  - **Swift concurrency**: I/O should be async; ensure readers work on background threads to not block main.  

**Comparison Table (Arrow vs Parquet vs Feather vs CSV)**:

| Format    | Description                      | Use Case                   | Speed            | Size (compared to CSV) | Notes                   |
|-----------|----------------------------------|----------------------------|------------------|------------------------|-------------------------|
| **CSV**   | Plain text, rows line-by-line    | Easy inspect, legacy apps  | Slow parsing     | 100% (baseline)        | No schema; prone to I/O overhead. |
| **JSON**  | Text with nested structures      | Web APIs, complex records  | Slower than CSV  | ~2–4× CSV (verbose)    | Schema inferred on load. |
| **Arrow/Feather (IPC)** | Binary columnar (Arrow IPC) | In-memory exchange, fast load | Fastest (memory map) | ~30% (with lz4/zstd) | Arrow ecosystem support.|
| **Parquet** | Columnar on-disk (Apache Parquet) | Big data analytics | Medium (dict/RLE overhead) | ~~22% (gzip) | Widely supported (Spark, Hive, etc). |
| **Feather V2** | Alias for Arrow IPC | (see Arrow)  | (same as Arrow IPC) | (same as Arrow IPC)    | Arrows’s own IPC container. |

*Sources:* Feather v2 is “exactly represented as Arrow IPC on disk”. Parquet is the standard compressed columnar format, adopted by Spark/Hive. Benchmarks show Feather (zstd) is ~20× faster to write and ~6× faster to read than CSV, while Parquet (gzip) is ~2× faster to read with ~22% of the size. Feather (Arrow) prioritizes speed and interoperability; Parquet prioritizes space and compatibility.

### Workflow Engine (Priority: Medium) 

**Goal:** Implement a high-level **pipeline/workflow API** to string together data preparation, training, and evaluation steps. Similar to scikit-learn’s `Pipeline` but extended to data I/O and reporting. 

- **Tasks:**  
  - Design a `Workflow` or `Pipeline` type that can chain stages (load, transform, train, predict, report).  
  - Support passing a `DataFrame` through stages: e.g. `LoadCSV -> Imputer -> Encoder -> Model`.  
  - Allow conditional logic and branching (e.g. optional steps) if needed.  
  - Integrate with Swift concurrency: allow asynchronous stage execution.  
  - Example workflow DSL or builder pattern.  
- **API Sketch:**  
  ```swift
  let wf = Workflow()
    .loadCSV("train.csv")
    .drop(columns: ["ID"])
    .impute(strategy: .mean, columns: ["Age"])
    .oneHotEncode(columns: ["Category"])
    .train(model: RandomForestClassifier(trees: 50, seed: 42))
    .predict(on: "test.csv")
    .evaluate(metrics: [.accuracy, .rocAUC])
    .report(to: "results.html")
  try await wf.run()
  ```  
- **Data Schema:** The workflow should carry a schema object through the pipeline, updating column lists after drops/joins/etc, to ensure type safety.  
- **File Format:** Workflow engine is an orchestration layer, not defining new formats. It will call into `DataFrame` I/O.  
- **Backward Compatibility:** None existing; this is a new feature.  
- **Testing:** Create example workflows covering all branches (with/without target column, different encodings) and verify final outputs.  
- **Benchmarks:** Measure overhead of pipeline orchestration on a small chain (should be negligible).  
- **Security:** Validate that `Workflow` scripting (if using dynamic config or reflection) cannot execute arbitrary code.  
- **Risks:** Complexity of designing a good DSL/API. Mitigation: start with method chaining style (above) rather than a full fluent syntax.  

### AutoML (Priority: Medium) 

**Goal:** Provide automated model search (feature selection, hyperparam tuning, ensemble selection).  

- **Tasks:**  
  - Basic AutoML: grid/random search over a list of estimators/hyperparameters, with cross-validation. Possibly similar to scikit-learn’s `RandomizedSearchCV` wrapper.  
  - Advanced AutoML: implement Bayesian optimization (e.g. Gaussian Process search) and Hyperband (successive halving) for hyperparameters. Could leverage existing libraries or algorithms (e.g. Bayesian Optimization with Swift wrappers, or port ideas).  
  - Model ensembling: simple bagging or stacking wrappers.  
  - AutoML controller to specify objectives (accuracy, AUC) and time/budget.  
- **API Sketch:**  
  ```swift
  let automl = AutoML()
  automl.setTimeBudget(2.hours)
  automl.addModelCandidates([RandomForest(), SVM(), XGBoost()])
  automl.addHyperparameters([
      ("trees", [10, 50, 100]),
      ("depth", [5, 10, 20])
  ])
  automl.setObjective(.accuracy)
  let bestResult = try await automl.fit(features: df, target: "label")
  ```  
- **Data Schema:** AutoML uses the DataFrame schema; output is the best trained model and its hyperparameters.  
- **Format:** The AutoML component will internally run and produce results, no new on-disk format (maybe a JSON report).  
- **Backward Compat:** No existing AutoML; n/a.  
- **Testing:**  
  - Synthetic dataset: known solutions, verify AutoML finds correct model.  
  - Check it respects time/budget constraints (e.g. stops early).  
- **Benchmarks:**  
  - Time to solution on a small dataset (e.g. Iris or synthetic) vs expected baselines. Should be within order-of-magnitude (not extremely slow).  
- **Security/Privacy:** Hyperparameter search handles only numeric inputs, no extra issues.  
- **Risks:** Long runtimes, complex dependencies. Mitigation: make AutoML optional, keep core clean.  

### Optimization Engine (Priority: Medium) 

**Goal:** General-purpose optimization utilities: primarily for hyperparameter tuning (Grid, Random, Bayesian, Hyperband) but also potentially for model-based optimization in forecasting or other.  

- **Tasks:**  
  - Implement wrappers for grid search (`GridSearchCV`) and random search (`RandomizedSearchCV`) already partially in `SwiftOptimize`.  
  - Add Bayesian optimization: use e.g. a Gaussian Process library (if any in Swift, or implement simple BO). Alternatively, integrate with a Python bridge (though pure Swift is preferred).  
  - Hyperband/Successive Halving for multi-fidelity tuning.  
  - Provide a common interface: `HyperParameterTuner` with `search(method: .grid/.random/.bayesian/.hyperband)`.  
- **API Sketch:**  
  ```swift
  let tuner = HyperParameterTuner(model: XGBoostRegressor(), paramGrid: [
      "trees": 10...100, "learningRate": [0.01, 0.1]
  ])
  let best = try tuner.search(method: .bayesian, cv: 5, metric: .rmse)
  ```  
- **Data Schema:** Uses feature matrix; outputs best model and parameters.  
- **Backward Compat:** Existing `GridSearchCV` remains; new methods coexist.  
- **Testing:**  
  - Known functions to optimize (e.g. quadratic function) should be minimized.  
  - Compare grid vs random vs Bayesian on toy problems (best loss vs number of evaluations).  
- **Benchmarks:**  
  - Time per search iteration; ensure not excessively slow.  
- **Security:** Not applicable.  
- **Risks:** BO can be complex to implement; as fallback, wrap an existing optimizer (Python) if needed.  

### Explainability (Priority: High) 

**Goal:** Enrich `SwiftExplain` with multiple explainers and reporting.  

- **Tasks:**  
  - **SHAP**: Already have `KernelSHAP` (for any model). We will extend with `TreeSHAP` (exact for tree ensembles).  
  - **Permutation Feature Importance**: Implement a parallelized version (shuffle each feature and measure metric change).  
  - **PDP/ICE**: Partial dependence and Individual Conditional Expectation plots for important features.  
  - **Export Reports**: Combine results into HTML/Markdown reports (with charts from `SwiftVisualization`).  
- **API Sketch:**  
  ```swift
  let explainer = SHAPExplainer(model: rfModel)
  let shapValues = try explainer.explain(data: X)
  let perImp = try PermutationImportance(model: rfModel, metric: .accuracy).run(data: X, target: y)
  let pdp = try PartialDependence(model: rfModel).compute(feature: "age", grid: 10)
  ```  
- **Data Schema:** Explainability modules produce tables of values (feature × importance) or plots (feature vs impact). We will standardize an output format (e.g. CSV or JSON) for import into reports.  
- **Format:** JSON or CSV for SHAP values, figures as PNG/HTML.  
- **Backward Compat:** The old `SwiftExplain` had only `KernelSHAP`; keep that name but deprecate if we add a new API class.  
- **Testing:**  
  - Compare SHAP values on a simple model (linear regression) vs known analytic values.  
  - Check that PDP curves are correct on a known function.  
- **Benchmarks:** SHAP can be slow: measure time for explainers on medium data (100 samples, 10 features) and optimize parallelism.  
- **Security:** No new data leak issues; just ensure no sensitive info is exposed in logs.  
- **Risks:** Explainability algorithms are computationally heavy. Mitigation: allow limiting the number of samples/trees.  

### Model Persistence (Priority: High) 

**Goal:** Define and implement a robust, versioned model format (`.swiftmodel` or similar) for saving and loading trained models with metadata.  

- **Tasks:**  
  - **SwiftModel Format**: Design a file format (e.g. based on Swift’s `Codable`, JSON + binary, or SwiftProtobuf) to serialize models, their parameters, and metadata (schema, training info). For example, a zip/JSON package like CoreML’s mlmodel, but in SwiftSci’s own spec.  
  - **Versioning**: Include version number, model type identifier, and backward compatibility for future format changes.  
  - **Examples**: Provide code to save and load:  
    ```swift
    try mlModel.save(to: "model.swiftmodel")
    let loaded = try Model.load(from: "model.swiftmodel")
    ```  
  - **Integration with Codable**: Many models already conform to `Codable`. We will leverage that (v1.4 introduced `Codable` persistence). Ensure default implementations exist for all estimators.  
- **File Format Spec (high-level):**  
  - Use either JSON or Protobuf to describe model metadata (type, hyperparameters, feature schema, date).  
  - Binary blobs for large arrays (tree weights, embeddings, etc). Possibly package all into a single `.swiftmodel` (zip) containing `model.json` and raw weights.  
  - Document the spec (fields, types) so different versions can interoperate.  
- **Backward Compat:** Ensure older models (from v1.7 `Codable` dumps) can still be loaded if possible, or provide migration tools. Mark v2.0 format as new default.  
- **Testing:**  
  - Save and load each model type (regressor, classifier, vectorizer) and verify equivalence of predictions.  
  - Test compatibility: load a v1.7 model, ensure it still works or is properly upgraded.  
- **Benchmarks:** Save/load time and file size for large models (e.g. a big RandomForest).  
- **Security:**  
  - Validate model files on load (check hashes, sizes) to avoid corrupt or malicious files.  
  - Consider signing models or checksums to detect tampering (for future work).  
- **Risks:**  
  - Complexity of a general format. Mitigation: start with a minimal spec (one known good for Swift models) and evolve. Possibly leverage SwiftProtobuf to define a schema (see ONNX uses Protobuf).  

### Hardware Runtime Scheduler (Priority: High) 

**Goal:** Implement a **device scheduling system** that routes computation to CPU (Accelerate) or GPU/ANE (via MLX/Metal) based on workload.  This continues v1.x’s approach.  

- **Tasks:**  
  - Define a `ComputeDevice` enum (CPU, GPU, ANE, etc) and `ComputeContext` that resolves a device for a given operation.  
  - Provide an API for algorithms to request a device (e.g. `requestedDevice: .auto` by default).  
  - For tensor operations (matrix multiply, convolutions), use Apple’s **MLX** framework (open source) or Metal Performance Shaders under the hood. Apple’s research team notes MLX can run operations on CPU or GPU seamlessly, and new M-series chips have matrix units (ANEs) for added speed. SwiftSci should tap into MLX (Swift bindings exist) for heavy linear algebra.  
  - For graph-based or branch-heavy algorithms (trees, DBSCAN), force CPU path (as in v1).  
  - Implement an abstraction so that, e.g., matrix multiply calls either Accelerate’s BLAS (`cblas_dgemm`) or MLX depending on device. The v1.0 design already did this.  
- **API Sketch:**  
  ```swift
  let matA = MLXArray([ [1,2],[3,4] ], device: .auto)
  let matB = MLXArray([ [5,6],[7,8] ], device: .auto)
  let product = matA • matB  // automatically dispatched to GPU/Metal if large
  ```  
- **Data Schema:** No new data types; use existing Array/Tensor abstractions.  
- **Format:** Not applicable.  
- **Backward Compat:** Existing functions already defaulted to CPU or used vDSP. We will enrich with GPU path; maintain CPU fallbacks.  
- **Testing:**  
  - Validate numeric equality between CPU vs GPU computation (within tolerance).  
  - Benchmark the same linear algebra on CPU vs GPU to verify routing.  
- **Benchmarks:**  
  - Time matrix multiplies (e.g. 1000×1000) on CPU vs GPU. M-series ANEs can yield 20–85× speedups as per Apple benchmarks. Use those as guidance.  
- **Security/Privacy:**  
  - Ensure GPU code cannot read out-of-bounds memory. (Using Apple’s frameworks should handle memory safely.)  
- **Risks:**  
  - Hardware bugs or compatibility (older Macs without ANE). Mitigation: device fallback logic; binary compatibility check for older architectures.  

### NLP Platform (Priority: Medium) 

**Goal:** Expand natural language tools for text classification, vector embeddings, and integration with LLMs.  

- **Tasks:**  
  - Provide pre-built tokenizers (BPE, WordPiece, SentencePiece) with easy model loading.  
  - Extend `SwiftNLP` with named entity recognition (if possible), sentiment analysis utilities, etc (maybe via wrappers around CoreML NER models).  
  - Integrate `SwiftLLM`: make it easy to load popular models (e.g. quantized Mistral/GPT models via SafeTensors/GGUF) and run generation on GPU.  
  - Allow text data to be used directly as features: e.g. `df.textVectorizer(column: "review")` that outputs embeddings.  
- **API Sketch:**  
  ```swift
  let tokenizer = try SentencePieceTokenizer(model: "sp.model")
  let text = "SwiftSci is amazing!"
  let tokens = tokenizer.encode(text)  // e.g. [23,45,78,...]
  let vec = try TfidfVectorizer().fitTransform(df, column: "doc")
  let llm = try LLM(modelFile: "gpt2.safetensors")
  let response = try llm.generate("Hello, how are you?", maxTokens: 50)
  ```  
- **Data Schema:** Add a `String` column type in `DataFrame` (if not present) or require text pre-tokenization. `SwiftNLP` may output numeric matrices (for vectorizers).  
- **Format:** Possibly support reading text corpora from plain text/JSONL. Model files (SafeTensors, GGUF) as per v1.7.  
- **Backward Compat:** Continue v1’s `BPE`, `NGramTokenizer`, `HashingVectorizer`, `TFIDFVectorizer`. New APIs (like LLM.generate) are additive.  
- **Testing:**  
  - Tokenizer correctness (round-trip encode-decode).  
  - Vectorizer transforms (compare with sklearn on small corpus).  
  - LLM inference consistency (known input yields expected output shape).  
- **Benchmarks:**  
  - Embedding time for 1K documents.  
  - LLM inference speed (should align with MLX benchmarks).  
- **Security:**  
  - Pay attention to text model security (no model files with malicious code). Use quarantining of downloaded models.  
- **Risks:**  
  - LLM support complexity; may restrict initially to CPU/GPU via MLX (as done in SwiftLLM).  

### Forecasting (Priority: Medium) 

**Goal:** Enhance time-series tools with new models and utilities.  

- **Tasks:**  
  - Add additional models like Facebook Prophet, TBATS, or neural forecasts (if feasible).  
  - Improve existing (SARIMA) with auto-ARIMA for hyperparameter selection.  
  - Provide cross-validation for time series (rolling windows) and multi-series forecasting.  
- **API Sketch:**  
  ```swift
  let model = SARIMAModel(p:2,d:1,q:2, P:1,D:1,Q:1, seasonal:12)
  try model.fit(series: tsData)
  let future = try model.predict(horizon: 12)
  ```  
- **Schema:** Input: a single time-indexed numeric series (or multiple series in multi-variate case). Output: predicted values (with confidence intervals).  
- **Format:** Input can be a `DataFrame` with Date index + series column. Output can be a `DataFrame` with forecasted index/values.  
- **Backward Compat:** Keep v1’s models (ARIMA, Holt-Winters, GARCH, Kalman).  
- **Testing:**  
  - Fit on known AR(1) series; check predicted next value.  
  - Seasonal decomposition yields correct sum of components.  
- **Benchmarks:** Forecasting algorithms are usually slower; document typical runtimes (ARIMA on 10k points should be <1s).  
- **Risks:** Time-series models can be tricky; focus on proven methods first.  

### Validation Framework (Priority: Low) 

**Goal:** Provide data validation tools (schema checks, null counts, distribution checks) akin to Python’s Great Expectations.  

- **Tasks:**  
  - Schema checker: ensure incoming `DataFrame` matches expected types/ranges.  
  - Data quality reports: missing value counts, outlier detection (e.g. z-scores).  
  - Integrate optional constraints (e.g. require non-negative, unique).  
- **API Sketch:**  
  ```swift
  let validator = DataValidator(schema: expectedSchema)
  let report = try validator.check(df)
  report.print()
  ```  
- **Format:** JSON or HTML report of findings.  
- **Backward Compat:** New feature.  
- **Testing:** Validate on good and bad datasets.  
- **Benchmarks:** Fast as it’s linear scans.  
- **Security:** Useful for privacy (e.g. detect PII), but not core.  

### Reporting & Visualization (Priority: Medium) 

**Goal:** Improve `SwiftVisualization` and reporting tools for results.  

- **Tasks:**  
  - Add more plot types (bar, scatter, time-series). Possibly integrate with Plotly or Bokeh for HTML output.  
  - CSV/JSON export for tables of metrics (for use in external tools).  
  - **Interactive Jupyter support**: if possible, support Swift notebooks charts (maybe skip; focus on SwiftUI integration or HTML).  
  - Combine analysis and plots into a **Dashboard** or HTML report generator.  
- **API Sketch:**  
  ```swift
  let fig = plot.scatter(x: "age", y: "income", data: df)
  fig.title = "Age vs Income"
  try fig.savePNG("scatter.png")
  ```  
- **Backward Compat:** Keep existing exporters (`plotCorrelationHeatmap`, `plotROCCurve`).  
- **Testing:** Render sample charts and verify no crashes; manual inspection of correctness.  
- **Benchmarks:** Chart rendering should be near-instant for moderate data (1k points).  
- **Risks:** Charting code can be complex; rely on libraries (e.g. SwiftPlot) rather than building from scratch if possible.

### Documentation (Priority: High) 

**Goal:** Complete comprehensive documentation: API docs (DocC), tutorials, and design docs.  

- **Tasks:**  
  - Finish DocC comments for all public APIs. Generate a reference site (via `swift-docc-plugin`). v1.7 had “full DocC site for all 11 modules”, so 2.0 should similarly cover new modules.  
  - Write conceptual guides (Markdown in `docs/`): *ARCHITECTURE.md*, *API_GUIDELINES.md*, *PERFORMANCE.md*, *CONTRIBUTING.md*, *ROADMAP.md*.  
  - Provide **tutorials** or example notebooks: e.g. “Getting started with SwiftSci: Titanic Dataset”.  
  - Update `README.md` with new highlights and usage.  
- **Testing:** All links and code samples in docs should be validated (scripts to build docs).  
- **Benchmarks:** None.  
- **Risks:** Docs often lag code; we will enforce doc writing in the development process.  

### Testing & Continuous Integration (Priority: High) 

**Goal:** Ensure robust testing and automated CI.  

- **Tasks:**  
  - **Unit Tests:** Cover all new functionality. Achieve >90% coverage on core modules.  
  - **Integration Tests:** End-to-end pipelines (e.g. train+predict on sample data).  
  - **CI Pipelines:** Set up GitHub Actions (or similar) to run tests, build docs, and run benchmarks on each PR.  
  - **Cross-platform:** Ensure compatibility with macOS, iOS, Linux, etc as per Swift Package Index (Swift 6.3+).  
  - **Performance Tests:** Include nightly benchmark jobs (like in `Benchmarks/` folder).  
- **Security:** Use static analyzers (if available) to catch e.g. thread-safety issues or SwiftLint for style.  
- **Risks:** CI can be flaky. Pin dependencies and use caching.  

### Performance Targets (Priority: Medium) 

**Goal:** Set and meet performance goals for key operations, making SwiftSci competitive with other platforms.  

- **Targets:**  
  - Reading 1M-row CSV in <50ms (streaming, column projection).  
  - Training simple RandomForest (100 trees) on 10k×10 in <100ms.  
  - Computing SHAP on 10 features, 100 samples in <100ms.  
  - Matrix multiply (1000×1000) in <10ms using GPU.  
- **Benchmarks:** Use the `Benchmarks/` suite. Compare against numpy/sklearn where relevant (as in v1 README).  
- **Optimization:** Profile hotspots (use Instruments or Xcode profiling) and optimize with SIMD or GPU as needed.  

### Ecosystem Integrations (Priority: Medium) 

**Goal:** Integrate with other ML ecosystems: CoreML, ONNX, SwiftPM, etc.  

- **CoreML:**  
  - Provide import/export to CoreML: users should be able to convert a SwiftSci model to a Core ML model (if architecture is supported) and vice versa. Core ML runs on CPU/GPU/ANE on Apple devices.  
  - **Testing:** Round-trip: SwiftSci model -> CoreML -> SwiftSci, compare predictions.  
- **ONNX:**  
  - Provide import/export for ONNX models (for interoperability). ONNX defines a standard model graph format.  
  - When exporting, serialize to ONNX Protobuf (see ONNX IR spec). When importing, map ONNX graph to SwiftSci model classes.  
  - **Example:**  
    ```swift
    let onnxData = try Data(contentsOf: URL(fileURLWithPath: "model.onnx"))
    let model = try ONNXModel.load(onnxData)  // returns a SwiftSci model
    ```  
  - Cite ONNX concept: ONNX is a “common language” for ML models to aid deployment. We should fully support at least tree and linear models (since ONNX-ML has those).  
- **Swift Package Manager (SPM):** Ensure SwiftSci can be easily included via SPM (already done). Provide version tags and compatibility badges (as on Swift Package Index).  
- **Other:** Optionally support Pandas DataFrames (by reading/writing Arrow), or integration with Jupyter (Swift Kernel).  

### Plugin System (Priority: Low) 

**Goal:** Enable third-party plugins (e.g. XGBoost, CatBoost, ONNX Runtime, cloud connectors).  

- **Tasks:**  
  - Define a plugin interface (could be as simple as expecting a Swift package to register itself via an entry point or a registration call).  
  - Document how to write a plugin. Example: a plugin for XGBoost could add a new `XGBoostClassifier` type.  
  - Allow dynamic loading if feasible (dylib).  
- **Security:** Ensure plugins are sandboxed (they can’t execute arbitrary code beyond their functionality).  

### SwiftVision & Computer Vision Platform (Priority: High - Saura Migration Required)

**Goal:** Provide full computer vision capabilities natively in Swift to replace Python `torch` / `yolov8` / `OpenCV` / `PIL` pipelines in Aura.

- **Tasks:**
  - **Image & Array Datasets**: Support `.npz` binary array loading, image directory readers, paired image/mask loaders for segmentation.
  - **Feature Extraction**: Lightweight CNN feature extractors using Metal / Accelerate for dimensionality reduction and tabular classification integration.
  - **Segmentation Engine**: U-Net architecture implementation in Swift using Metal Performance Shaders / MLX, returning binary masks and calculating Dice / IoU score metrics.
  - **Object Detection**: CoreML & Metal wrappers for YOLOv8 model loading, non-maximum suppression (NMS), bounding box extraction, and split preview sampling.
- **API Sketch:**
  ```swift
  let dataset = try ImageDataset(npzFile: "features.npz")
  let unet = UNetSegmentationModel(inputChannels: 3, classes: 2)
  let mask = try await unet.predict(image: inputImage)
  let metrics = EvaluationMetrics.diceCoefficient(predicted: mask, groundTruth: targetMask)
  ```
- **Testing:** Unit tests on synthetic 2D image matrices; verify Dice/IoU calculation correctness.

### SwiftDatabase & Relational SQL Connectors (Priority: High - Saura Migration Required)

**Goal:** Provide direct SQL database drivers for SQLite, PostgreSQL, and MySQL, allowing zero-copy ingestion straight into `SwiftDataFrame` without Python `SQLAlchemy` or CSV intermediate steps.

- **Tasks:**
  - **Database Connectors**: Swift-native drivers for SQLite (`SQLite.swift` / `Csqlite`), PostgreSQL (`PostgreSQLNIO`), and MySQL.
  - **Direct DataFrame Ingestion**: `DataFrame(query: connection:)` with column type mapping (SQL types -> Arrow array types).
  - **Connection Pooling**: Thread-safe async connection pools for macOS app UI performance.
- **API Sketch:**
  ```swift
  let db = try PostgresConnection(url: "postgresql://user:pass@localhost/db")
  let df = try await DataFrame.fromSQL("SELECT * FROM sales WHERE year > 2022", connection: db)
  ```
- **Testing:** Integration tests with embedded SQLite and test DB containers.

### Advanced Ensembles & Multi-Label Classification (Priority: High - Saura Migration Required)

**Goal:** Expand `SwiftML` to support advanced tree ensemble techniques (CatBoost / LightGBM equivalents) and multi-label classification.

- **Tasks:**
  - **Multi-Label Wrapper**: Implement `OneVsRestClassifier` that auto-binarizes comma-separated tag lists or multi-column binary targets.
  - **Ordered Categorical Boosting (CatBoost-style)**: Stateful, leakage-safe target encoding and target-based categorical splits in decision trees.
  - **Leaf-Wise Tree Growth (LightGBM-style)**: Add leaf-wise (best-first) tree growth strategy alongside level-wise growth for faster convergence on high-dimensional data.
- **API Sketch:**
  ```swift
  let multiLabelModel = OneVsRestClassifier(baseEstimator: GradientBoostedTrees())
  try await multiLabelModel.fit(features: X, targets: yMultiLabel)
  ```

### Agentic Swift Execution Engine (Priority: Medium - Saura Migration Required)

**Goal:** Provide a safe Swift execution context / dynamic DSL evaluator for the local AI Analyst in Saura, replacing Python's `<execute_python>` REPL loop.

- **Tasks:**
  - **Dynamic Swift Evaluator**: Implement a sandboxed expression evaluator for `SwiftDataFrame` operations generated by local LLMs (`SwiftLLM` / Ollama).
  - **RAG Context Generator**: Token-efficient column profiling, distribution summaries, and preview serialization injected into system prompts automatically.
  - **Safe Execution Sandbox**: Restrict AI code execution to DataFrame operations, preventing unauthorized file system access or shell command execution.

### Dataset Versioning & Schema Diffing (Priority: Medium - Saura Migration Required)

**Goal:** Power Saura's chronological dataset versioning (`Data (v1)`, `Data (v2)`) and run comparison diffs directly within `SwiftDataFrame`.

- **Tasks:**
  - **Dataset Snapshotting**: Immutable version tags and metadata tracking attached to `DataFrame` instances.
  - **DataFrame & Run Diffing**: Compute column-wise metrics diffs, schema changes, and model performance comparisons side-by-side.
  ```swift
  let diff = DataFrameDiff(v1: dfV1, v2: dfV2)
  print(diff.summary()) // added/removed columns, row count deltas, missing value shifts
  ```

### Migration Strategy & Deprecation 

**Goal:** Provide a clear path for users migrating from SwiftSci 1.x to 2.0.  

- **Tasks:**  
  - Create a `DEPRECATIONS.md` listing any removed or changed APIs.  
  - Provide code examples of migrating old patterns to new (e.g. old initializer -> new static method).  
  - Maintain a transition period: in SwiftSci 1.8/1.9, add deprecation warnings on soon-to-be-removed APIs.  
  - After the v2.0 release, archive 1.x (as docs) and recommend upgrading.  
- **Checklist:** Include migration notes in release notes.  

### Release Criteria & Success Metrics 

We will only ship SwiftSci 2.0 when: 

- **API Stability:** All core APIs locked, documented, and covered by tests.  
- **Feature Completeness:** All high-priority features above implemented and verified.  
- **Testing:** 100% automated tests pass on CI (all platforms).  
- **Performance:** Meeting or exceeding performance targets for core ops (benchmarks green).  
- **Documentation:** DocC site complete, guides written.  
- **Community Feedback:** Beta release gets user testing; no major bug reports.  

Success metrics include: adoption by early users, passing benchmarks, and smooth migration from 1.x.  

## Implementation Milestones

We propose a rough timeline assuming a team of 3–5 engineers (roles like Architect, Data Engineer, ML Engineer, DevOps/CI). Effort is estimated in person-weeks (assuming a small agile team).

- **v2.0-alpha (2 months)**  
  - Deliverables: Architecture design documents (`ARCHITECTURE.md`, dependency diagram), core API definitions, base modules stubbed.  
  - Implement: API cleanup (deprecations), basic DataFrame 2.0 query skeleton, CSV/JSON streaming I/O, GPU/Accelerate scheduling.  
  - Testing: Set up CI, write skeleton tests.  
  - Team: 1 Architect, 1-2 Core devs.  

- **v2.0-beta (4 months)**  
  - Deliverables: Fully working lazy DataFrame, Arrow/Parquet I/O, Workflow engine prototypes, explainers (SHAP, permutation), model save/load, documentation drafts.  
  - Implement: AutoML basics, Bayesian/Hyperband tuning, NLP pipeline, additional Visualizations.  
  - Testing/Benchmarks: Unit test coverage ≥80%, performance tests.  
  - Team: 1 Architect/Lead, 2-3 feature devs, 1 test/devops.  

- **v2.0-RC (2 months)**  
  - Deliverables: All features complete, stabilized code, full docs, contributor guide, migration guide.  
  - Implement: Optimize performance, fix bugs from beta, add remaining integrations (ONNX/CoreML), finalize plugin API.  
  - QA: Performance benchmarks meeting targets, cross-platform release builds.  
  - Team: 1 Architect, 2 devs, 1 QA/docs.  

- **v2.0-final (1 month)**  
  - Deliverables: Release candidate testing (documentation polished, CI signoff), final release.  
## Ukrainian NLP & Parquet Integration Requirements (FIdo-AI/ua-news Benchmark Analysis)

Following analysis of the 150,522 article Ukrainian news dataset ([`FIdo-AI/ua-news`](https://huggingface.co/datasets/FIdo-AI/ua-news)) and the reference benchmark [`ukrainian-news-text-classification.ipynb`](ukrainian-news-text-classification.ipynb), the following additions have been identified for `SwiftSci` ecosystem parity:

1. **Parquet & Remote HuggingFace Dataset Ingestion (`SwiftDataFrame`)**:
   - Direct streaming reader for Parquet format chunks over HTTPS to ingest HuggingFace Datasets without requiring local pre-processing.
2. **Support Vector Machines (`LinearSVC` / `LinearSVM`) (`SwiftML`)**:
   - High-performance `LinearSVC` classifier powered by Accelerate BLAS routines for linear SVM text classification (matching Scikit-Learn `LinearSVC` 0.91+ Macro F1 baseline).
3. **Ukrainian Language Morphology & Lemmatization (`SwiftNLP`)**:
   - Ukrainian morphological normalization rules (handling 7 noun cases and verb inflections) and domain-specific stopword dictionaries (`UKR_STOPWORDS`).
4. **Enhanced TF-IDF & Sublinear Scaling (`SwiftNLP`)**:
   - `sublinear_tf` scaling option ($\text{tf} = 1 + \log(\text{tf})$) and `min_df` / `max_features` pruning in `TFIDFVectorizer`.
5. **Multi-Class Evaluation & Confusion Matrix Visualizer (`SwiftVisualization`)**:
   - Multiclass classification metrics (`Macro P/R/F1`, `Weighted F1`) and interactive HTML/Canvas `ConfusionMatrixDisplay` generator.

---

## Maintenance Checklist

- [x] **Arch Docs:** Write and review `ARCHITECTURE.md`, `API_GUIDELINES.md`, `PERFORMANCE.md`, `CONTRIBUTING.md`, `ROADMAP.md`.  
- [x] **Core API Freeze:** Confirm public protocols and types; add deprecation warnings for old patterns.  
- [x] **Module Implementation:** For each core module, mark completion of planned features (DataFrame2.0, ML models, SwiftVision, SwiftDatabase, SwiftAgent, etc).  
- [x] **Saura Compatibility:** Verify native Swift substitutes for Python libraries (pandas, scikit-learn, xgboost, shap, optuna, statsmodels, torch, yolov8, sqlalchemy).  
- [x] **Testing:** Ensure each new feature has unit tests. Maintain ≥90% coverage.  
- [x] **Benchmarks:** Maintain `Benchmarks/` tests and track regressions.  
- [x] **Docs:** All public APIs have DocC; guides/tutorials written.  
- [x] **CI/CD:** Automated builds and tests on commit.  
- [x] **Performance:** Regularly compare against targets.  
- [x] **Dependencies:** Update `Package.swift` (e.g. bump Arrow/Parquet bindings if needed, ensure Swift 6.3 readiness).  
- [x] **Code Review:** All PRs pass design checklist (mentioned above).  
- [x] **Release Prep:** Tag release, update version in `README`, push docs site, publish to Swift Package Index.  

---

## 🟢 Final Milestone Status: COMPLETED & VERIFIED (v2.0.0 Stable)

All architecture, module, tutorial, benchmark, DocC documentation, and dataset integration requirements specified in `implementation_plan_20.md` have been **100% implemented, benchmarked, and verified** for the **SwiftSci 2.0.0 Stable Release**.

