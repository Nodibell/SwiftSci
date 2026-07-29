# SwiftSci 2.2.0 Complete Performance Benchmarks

Official comprehensive comparative benchmark suite results comparing **SwiftSci 2.2.0** against Python data science libraries (**NumPy**, **Pandas**, **Scikit-Learn**, **Statsmodels**, **SHAP**, **PyTorch**) on Apple Silicon (M-series).

---

## 📈 1. Time Series Forecasting (SwiftForecast)

| Benchmark Scenario | SwiftSci 2.2.0 (Swift) | Python Baseline | Swift Speedup Ratio | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **ARIMA(1,1,1) Fit** (50k pts) | **2.48 ms** | 227.34 ms (*Statsmodels*) | ⚡ **91.60× Faster** | 🟢 Swift |
| **ARIMA(1,1,1) Forecast** (horizon=24) | **2.49 ms** | 224.57 ms (*Statsmodels*) | ⚡ **90.33× Faster** | 🟢 Swift |
| **Holt-Winters Fit** (50k pts, period=12) | **7.42 ms** | 144.90 ms (*Statsmodels*) | ⚡ **19.52× Faster** | 🟢 Swift |

---

## 🤖 2. Machine Learning & Clustering (SwiftML / SwiftCluster)

| Benchmark Scenario | SwiftSci 2.2.0 (Swift) | Python Baseline | Swift Speedup Ratio | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **RandomForest Fit** (1k×4, 50 trees) | **4.63 ms** | 27.10 ms (*Scikit-Learn*) | ⚡ **5.86× Faster** | 🟢 Swift |
| **GBDT Regressor Fit** (1k×4, 50 est) | **8.51 ms** | 34.80 ms (*Scikit-Learn*) | ⚡ **4.09× Faster** | 🟢 Swift |

---

## 📝 3. Natural Language & Explainability (SwiftNLP / SwiftExplain)

| Benchmark Scenario | SwiftSci 2.2.0 (Swift) | Python Baseline | Swift Speedup Ratio | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **KernelSHAP Explain** (100 coalitions) | **0.18 ms** | 0.46 ms (*SHAP*) | ⚡ **2.57× Faster** | 🟢 Swift |
| **LLM Forward Pass** (seqLen=64) | **0.51 ms** | 0.67 ms (*PyTorch*) | ⚡ **1.31× Faster** | 🟢 Swift |

---

## 📊 4. Core Data Engines & Vector Stats (SwiftStats / SwiftDataFrame / Vision / Database / Agent)

| Benchmark Scenario | SwiftSci 2.2.0 (Swift) | Python Baseline | Swift Speedup Ratio | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **CSV Read** (100k rows) | **16.53 ms** | 20.11 ms (*Pandas*) | ⚡ **1.22× Faster** | 🟢 Swift |
| **CSV Stream + GroupBy** (100k rows) | **22.88 ms** | 30.05 ms (*Pandas*) | ⚡ **1.31× Faster** | 🟢 Swift |
| **Pearson Correlation** (500k pairs) | **0.866 ms** | 1.233 ms (*NumPy*) | ⚡ **1.42× Faster** | 🟢 Swift |
| **Mean Reduction** (vDSP 1M elements) | **0.082 ms** | 0.118 ms (*NumPy*) | ⚡ **1.44× Faster** | 🟢 Swift |
| **StdDev Reduction** (vDSP 1M elements) | **0.311 ms** | 0.516 ms (*NumPy*) | ⚡ **1.66× Faster** | 🟢 Swift |
| **Variance Reduction** (vDSP 1M elements) | **0.318 ms** | 0.511 ms (*NumPy*) | ⚡ **1.61× Faster** | 🟢 Swift |

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
swift run -c release SwiftAnalyticsBenchmarks --json Benchmarks/Results/swift_results.json
```

Run the Python comparison suite:
```bash
cd Benchmarks/Python
python3 benchmarks.py --json Benchmarks/Results/python_results.json
python3 compare.py Benchmarks/Results/swift_results.json Benchmarks/Results/python_results.json
```
