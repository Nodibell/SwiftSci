# SwiftSci Performance Benchmarks

Official comparative benchmark suite results comparing **SwiftSci 2.0** against Python data science libraries (**NumPy**, **Pandas**, **Scikit-Learn**, **Statsmodels**, **SHAP**, **PyTorch**) on Apple Silicon (M-series).

---

## ⚡ Summary Benchmark Highlights

| Benchmark Scenario | SwiftSci 2.0 (Swift) | Python Library | Swift Speedup Ratio |
| :--- | :---: | :---: | :---: |
| **ARIMA(1,1,1) Fit (50k pts)** | **2.41 ms** | 223.84 ms (Statsmodels) | ⚡ **92.8× Faster** |
| **Holt-Winters Fit (50k pts)** | **6.77 ms** | 143.02 ms (Statsmodels) | ⚡ **21.1× Faster** |
| **RandomForest Fit (1k×4, 50 trees)** | **4.81 ms** | 25.66 ms (Scikit-Learn) | ⚡ **5.3× Faster** |
| **OneVsRestClassifier (5 classes)** | **0.73 ms** | 3.50 ms (Scikit-Learn) | ⚡ **4.8× Faster** |
| **KernelSHAP Explain (100 coalitions)** | **0.11 ms** | 0.48 ms (SHAP Python) | ⚡ **4.4× Faster** |
| **TF-IDF Vectorizer (50 docs)** | **1.01 ms** | 4.20 ms (Scikit-Learn) | ⚡ **4.1× Faster** |
| **Mean Reduction (vDSP 1M)** | **0.086 ms** | 0.122 ms (NumPy) | ⚡ **1.4× Faster** |
| **LLM Generate (10 tokens)** | **3.87 ms** | 4.28 ms (PyTorch) | ⚡ **1.1× Faster** |

---

## 🖥️ Benchmark Platform Details

- **Hardware**: Apple Silicon M-series (Unified Memory Architecture - UMA)
- **Swift**: Swift 6 (Strict Concurrency Enabled, Accelerated via `vDSP` / `LAPACK` & `MLX`)
- **Python**: 3.11.9 (`NumPy 2.3.5`, `Pandas 3.0.2`, `Scikit-Learn 1.4`, `Statsmodels 0.14`, `PyTorch 2.11`)

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
