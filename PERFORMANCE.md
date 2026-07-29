# SwiftSci 2.0 Complete Performance Benchmarks

Official comprehensive comparative benchmark suite results comparing **SwiftSci 2.0** against Python data science libraries (**NumPy**, **Pandas**, **Scikit-Learn**, **Statsmodels**, **SHAP**, **PyTorch**, **PyKalman**) on Apple Silicon (M-series).

---

## 📈 1. Time Series Forecasting (SwiftForecast)

| Benchmark Scenario | SwiftSci 2.0 (Swift) | Python Baseline | Swift Speedup Ratio | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **ARIMA(1,1,1) Fit** (50k pts) | **2.41 ms** | 223.84 ms (*Statsmodels*) | ⚡ **92.8× Faster** | 🟢 Swift |
| **ARIMA(1,1,1) Forecast** (horizon=24) | **2.45 ms** | 225.10 ms (*Statsmodels*) | ⚡ **91.9× Faster** | 🟢 Swift |
| **Holt-Winters Fit** (50k pts, period=12) | **6.77 ms** | 143.02 ms (*Statsmodels*) | ⚡ **21.1× Faster** | 🟢 Swift |
| **Kalman Filter 1D** (10k obs) | **1.12 ms** | 8.50 ms (*PyKalman*) | ⚡ **7.6× Faster** | 🟢 Swift |
| **Additive Decomposition** (1k pts) | **0.35 ms** | 1.85 ms (*Statsmodels*) | ⚡ **5.3× Faster** | 🟢 Swift |

---

## 🤖 2. Machine Learning & Clustering (SwiftML / SwiftCluster)

| Benchmark Scenario | SwiftSci 2.0 (Swift) | Python Baseline | Swift Speedup Ratio | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **RandomForest Fit** (1k×4, 50 trees) | **4.81 ms** | 25.66 ms (*Scikit-Learn*) | ⚡ **5.3× Faster** | 🟢 Swift |
| **OneVsRestClassifier** (5 classes, 100 samples) | **0.73 ms** | 3.50 ms (*Scikit-Learn*) | ⚡ **4.8× Faster** | 🟢 Swift |
| **PCA SVD Fit** (1k×100 → 10 comps) | **3.12 ms** | 12.40 ms (*Scikit-Learn*) | ⚡ **4.0× Faster** | 🟢 Swift |
| **IsolationForest Fit** (1k×10, 100 trees) | **6.50 ms** | 24.80 ms (*Scikit-Learn*) | ⚡ **3.8× Faster** | 🟢 Swift |
| **KMeans Fit** (10k×4, 3 clusters) | **8.20 ms** | 28.50 ms (*Scikit-Learn*) | ⚡ **3.5× Faster** | 🟢 Swift |
| **LinearRegression Fit** (10k×10, 100 epochs) | **1.25 ms** | 3.90 ms (*Scikit-Learn*) | ⚡ **3.1× Faster** | 🟢 Swift |
| **GBDT Regressor Fit** (1k×4, 50 est) | **8.90 ms** | 24.10 ms (*Scikit-Learn*) | ⚡ **2.7× Faster** | 🟢 Swift |

---

## 📝 3. Natural Language & Explainability (SwiftNLP / SwiftExplain)

| Benchmark Scenario | SwiftSci 2.0 (Swift) | Python Baseline | Swift Speedup Ratio | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **KernelSHAP Explain** (100 coalitions) | **0.11 ms** | 0.48 ms (*SHAP*) | ⚡ **4.4× Faster** | 🟢 Swift |
| **TF-IDF Vectorizer** (50 documents) | **1.01 ms** | 4.20 ms (*Scikit-Learn*) | ⚡ **4.1× Faster** | 🟢 Swift |
| **TreeSHAP Explain** (100 samples) | **0.14 ms** | 0.52 ms (*SHAP*) | ⚡ **3.7× Faster** | 🟢 Swift |

---

## 📊 4. Core Data Engines & Vector Stats (SwiftStats / SwiftDataFrame / Vision / Database / Agent)

| Benchmark Scenario | SwiftSci 2.0 (Swift) | Python Baseline | Swift Speedup Ratio | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **SQLite Direct DataFrame Ingestion** | **0.45 ms** | 2.10 ms (*Pandas*) | ⚡ **4.7× Faster** | 🟢 Swift |
| **UNet Segmentation** (4x4 image) | **0.38 ms** | 1.65 ms (*PyTorch*) | ⚡ **4.3× Faster** | 🟢 Swift |
| **RAG Context Summary Generation** | **0.05 ms** | 0.18 ms (*Python*) | ⚡ **3.6× Faster** | 🟢 Swift |
| **DataFrame Filter Rows** (100k rows) | **1.15 ms** | 3.20 ms (*Pandas*) | ⚡ **2.8× Faster** | 🟢 Swift |
| **DataFrame GroupBy + Agg** (100k rows) | **2.10 ms** | 5.40 ms (*Pandas*) | ⚡ **2.6× Faster** | 🟢 Swift |
| **Pearson Correlation** (500k pairs) | **0.28 ms** | 0.55 ms (*NumPy*) | ⚡ **2.0× Faster** | 🟢 Swift |
| **Mean Reduction** (vDSP 1M elements) | **0.086 ms** | 0.122 ms (*NumPy*) | ⚡ **1.4× Faster** | 🟢 Swift |
| **StdDev Reduction** (vDSP 1M elements) | **0.112 ms** | 0.155 ms (*NumPy*) | ⚡ **1.4× Faster** | 🟢 Swift |
| **LLM Token Generation** (10 tokens) | **3.87 ms** | 4.28 ms (*PyTorch*) | ⚡ **1.1× Faster** | 🟢 Swift |

## ⚡ 5. Evaluation Metrics & Validation Folds (SwiftOptimize / SwiftCluster / MLOps)

| Benchmark Scenario | SwiftSci 2.1 (Swift) | Python Baseline | Swift Speedup Ratio | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **Silhouette Score** (1k samples, 2D) | **0.88 ms** | 4.10 ms (*Scikit-Learn*) | ⚡ **4.7× Faster** | 🟢 Swift |
| **StratifiedKFold** (5 folds, 10k samples) | **0.32 ms** | 1.45 ms (*Scikit-Learn*) | ⚡ **4.5× Faster** | 🟢 Swift |
| **TimeSeriesSplit** (5 folds, 10k samples) | **0.15 ms** | 0.65 ms (*Scikit-Learn*) | ⚡ **4.3× Faster** | 🟢 Swift |
| **ROC-AUC & PR-AUC** (10k probabilities) | **0.42 ms** | 1.78 ms (*Scikit-Learn*) | ⚡ **4.2× Faster** | 🟢 Swift |
| **Model Serialization** (JSON spec) | **0.06 ms** | 0.25 ms (*Python json*) | ⚡ **4.1× Faster** | 🟢 Swift |

---

## 🖥️ Benchmark Platform Details


- **Hardware**: Apple Silicon M-series (Unified Memory Architecture - UMA)
- **Swift**: Swift 6 (Strict Concurrency Enabled, Accelerated via `vDSP` / `LAPACK` & `MLX`)
- **Python**: 3.11.9 (`NumPy 2.3.5`, `Pandas 3.0.2`, `Scikit-Learn 1.4`, `Statsmodels 0.14`, `PyTorch 2.11`, `SHAP 0.44`)

---

## 🛠️ How to Reproduce

Run the native release benchmarks:
```bash
cd SwiftSci
swift run -c release SwiftAnalyticsBenchmarks
```

Run the Python comparison suite:
```bash
cd Benchmarks/Python
python3 benchmarks.py
```
