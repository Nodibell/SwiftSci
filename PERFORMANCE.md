# SwiftSci 2.8.2 Complete Performance Benchmarks

Official comprehensive comparative benchmark suite results comparing **SwiftSci 2.8.2** (Release Build `-c release`) against Python data science libraries (**NumPy**, **Pandas**, **Scikit-Learn**, **Statsmodels**, **SHAP**, **PyTorch**, **Ultralytics**) on Apple Silicon (M-series / macOS 15 arm64).


> [!NOTE]
> All benchmarks are executed under identical conditions: deterministic seeds (`seed=42`), single-node execution, and release optimizations (`swift run -c release`).

---

## 📊 Complete Benchmark Matrix

| Benchmark Scenario | SwiftSci 2.8.2 (Swift) | Python Baseline | Speedup | Winner | Status / Notes |

| :--- | :---: | :---: | :---: | :---: | :--- |
| **YOLOPreprocessor Letterbox** (1920×1080 → 640×640) | **0.49 ms** | 1.85 ms (*OpenCV/Torch*) | ⚡ **3.78×** | 🟢 **Swift** | >2,000 FPS letterbox preprocessor |
| **UNetSegmentation Predict** (128×128 image) | **0.11 ms** | 0.42 ms (*PyTorch*) | ⚡ **3.82×** | 🟢 **Swift** | U-Net GPU segmentation forward pass |
| **YOLOv8Detector Detect** (640×640 real GPU) | **16.61 ms** | 28.50 ms (*PyTorch*) | ⚡ **1.72×** | 🟢 **Swift** | Real YOLOv8n GPU forward pass (~60 FPS) |
| **ARIMA(1,1,1) Fit** (50k pts) | **2.38 ms** | 210.94 ms (*Statsmodels*) | ⚡ **88.3×** | 🟢 **Swift** | Swift 6 native state-space solver |
| **ARIMA(1,1,1) Forecast** (horizon=24) | **2.42 ms** | 210.75 ms (*Statsmodels*) | ⚡ **86.8×** | 🟢 **Swift** | Zero-allocation forecast loop |
| **Holt-Winters Fit** (50k pts, period=12) | **6.83 ms** | 142.87 ms (*Statsmodels*) | ⚡ **20.9×** | 🟢 **Swift** | Vectorized level/trend updates |
| **LinearSVC Fit** (1k×4, Metal GPU) | **0.48 ms** | 3.85 ms (*Scikit-Learn*) | ⚡ **8.02×** | 🟢 **Swift** | MLX Metal GPU Hinge loss solver |
| **RandomForest Fit** (1k×4, 50 trees) | **5.43 ms** | 26.23 ms (*Scikit-Learn*) | ⚡ **4.83×** | 🟢 **Swift** | Pre-sorted DOD trees & SIMD MSE split |
| **GBDT Regressor Fit** (1k×4, 50 est) | **8.55 ms** | 33.15 ms (*Scikit-Learn*) | ⚡ **3.87×** | 🟢 **Swift** | Parallel tree gradient boosting |
| **KernelSHAP Explain** (100 coalitions) | **0.17 ms** | 0.47 ms (*SHAP*) | ⚡ **2.76×** | 🟢 **Swift** | Swift `TaskGroup` parallel coalitions |
| **Kalman Filter 1D** (10k obs) | **67.11 ms** | 83.78 ms (*NumPy*) | ⚡ **1.25×** | 🟢 **Swift** | Accelerate matrix updates |
| **CSV Stream + GroupBy** (100k rows) | **23.53 ms** | 28.34 ms (*Pandas*) | ⚡ **1.20×** | 🟢 **Swift** | Memory-mapped streaming reader |
| **LLM Forward Pass** (seqLen=64) | **0.43 ms** | 0.53 ms (*PyTorch*) | ⚡ **1.22×** | 🟢 **Swift** | MLX Metal GPU execution & compile cache |
| **CSV Read** (100k rows) | **17.29 ms** | 18.99 ms (*Pandas*) | ⚡ **1.10×** | 🟢 **Swift** | Memory-mapped zero-copy parser |
| **Pearson Correlation** (500k pairs) | **1.40 ms** | 1.24 ms (*NumPy*) | 0.88× | 🔴 **Python** | Vectorized dot product |
| **PCA SVD Fit** (1k×100 → 10 comps) | **0.98 ms** | 0.75 ms (*Scikit-Learn*) | 0.76× | 🔴 **Python** | Halko $O(MNk)$ `RandomizedSVD` |
| **KMeans Fit** (10k×4, 3 clusters) | **23.13 ms** | 11.45 ms (*Scikit-Learn*) | 0.50× | 🔴 **Python** | OpenMP parallel centroids in C |
| **LLM Token Generate** (10 tokens) | **7.16 ms** | 3.88 ms (*PyTorch*) | 0.54× | 🔴 **Python** | Includes streaming UI & tokenizer |



---

## 🔍 Detailed Analysis of Optimizations in v2.8.0

1. **`SwiftPreprocessing` Zero-Allocation Value Semantics**:
   - `MinMaxScaler`, `StandardScaler`, and `RobustScaler` operate as thread-safe `struct` value types.
   - `Pipeline` and `ColumnTransformer` mutate array elements directly by index in-place.
2. **SwiftNLP WordNet Engine & Accelerate Optimization**:
   - WordNet synset lookup and similarity metrics (`pathSimilarity`, `wupSimilarity`) leverage fast in-memory trie structures.
   - VADER Lexicon lookup uses zero-allocation binary search over pre-sorted static Key-Value arrays in `VADERLexicon.swift`.
   - Cosine similarity in `WordEmbeddings` is accelerated via `vDSP_dotprD` and `vDSP_svesqD`.
3. **SIMD Bitmask Filtering (`SwiftDataFrame`)**:
   - `filterIndicesDoubleSIMD` and `filterIndicesInt64SIMD` use `SIMD4` vector registers for fast row mask evaluation.
4. **Primitive Pointer Radix Sorting (`SwiftDataFrame`)**:
   - `sortIndicesPrimitiveFast` for zero-copy array sorting over `UnsafeBufferPointer<Double>`.
5. **vDSP FIR & Spectral Decomposition (`SwiftForecast`)**:
   - 1D moving average convolution via `vDSP_convD` and real FFT decomposition via `vDSP_fft_zipD`.

---

## 🖥️ Benchmark Platform Details

- **Hardware**: Apple Silicon M-series (Unified Memory Architecture - UMA)
- **Swift**: Swift 6 (Strict Concurrency Enabled, Accelerated via `vDSP` / `LAPACK` & `MLX`)
- **Python**: 3.11.9 (`NumPy 2.3.5`, `Pandas 3.0.2`, `Scikit-Learn 1.4`, `Statsmodels 0.14`, `PyTorch 2.11`, `SHAP 0.44`)

---

## 🛠️ How to Reproduce

Run native Swift release benchmarks:
```bash
swift run -c release SwiftSciBenchmarks --json Benchmarks/Results/swift_results.json
```

Run Python comparison suite:
```bash
python3 Benchmarks/Python/benchmarks.py --json Benchmarks/Results/python_results.json
python3 Benchmarks/Python/compare.py Benchmarks/Results/swift_results.json Benchmarks/Results/python_results.json
```
