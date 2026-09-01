# 🎯 SwiftSci 3.5.0 — Accuracy & Quality Benchmarks

Порівняння точності моделей між **SwiftSci 3.5.0** (Swift 6, Apple Silicon Accelerate) та Python reference libraries (**Scikit-Learn**, **Statsmodels**, **SciPy**, **NLTK**).

**Датасети ідентичні** — Python-скрипт використовує точну реплікацію `BenchmarkLCG` з [`BenchmarkSuite.swift`](Benchmarks/Swift/BenchmarkSuite.swift) (`a = 6364136223846793005`, `c = 1442695040888963407`, `seed = 42`), що гарантує побітово однакові датасети для чесного порівняння.

> 📋 **Дата запуску:** 2026-09-02 · **Платформа:** Apple Silicon (arm64) · **Swift:** 6.x

---

## 📊 Accuracy Scorecard: SwiftSci 3.5.0 vs Python

### 🤖 Machine Learning (ідентичні датасети — LCG seed=42)

| Model | Config | SwiftSci 3.5.0 | Python Baseline | Δ Difference | Parity |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **GBDT Regressor** | 30 trees, depth=4, lr=0.1 | RMSE=**0.421**<br>MAE=0.344<br>R²=0.9879 | RMSE=**0.421**<br>MAE=0.344<br>R²=0.9879 | $\Delta = 0.000$ | ✅ **Точний збіг** |
| **Random Forest** | 30 trees, Gini, depth=5 | Acc=**99.00%**<br>F₁=0.991 | Acc=**99.00%**<br>F₁=0.991 | $\Delta = 0.00\%$ | ✅ **Точний збіг** |
| **Naive Bayes** | Multinomial, 3 класи | Acc=**35.00%**<br>Macro-F₁=0.342 | Acc=**35.00%**<br>Macro-F₁=0.342 | $\Delta = 0.00\%$ | ✅ **Точний збіг** |

---

### 🔮 Time-Series Forecasting (однакові датасети, різна оптимізація параметрів)

| Model | Config | SwiftSci 3.5.0 | Python (Statsmodels) | Причина різниці |
| :--- | :--- | :---: | :---: | :--- |
| **Holt-Winters** | Add. trend + seasonal, $h=24$ | RMSE=9.764<br>MAPE=6.11%<br>R²=−1.333 | RMSE=0.338<br>MAPE=0.21%<br>R²=0.997 | SwiftSci: фіксовані α=0.2, β=0.1, γ=0.1<br>Statsmodels: MLE-оптимізація |
| **ARIMA(1,1,1)** | $h=24$ | RMSE=10.218<br>MAPE=5.87%<br>R²=−1.555 | RMSE=22.158<br>MAPE=14.15%<br>R²=−11.014 | Різні солвери; Swift Kalman-фільтр дає кращу екстраполяцію |

> **Примітка:** Holt-Winters різниця — виключно через відсутність оптимізації гіперпараметрів α/β/γ в SwiftSci. Датасети ідентичні. Заплановано Nelder-Mead оптимізацію в Release 3.6.0.

---

### 📐 Statistical Tests

| Test | SwiftSci 3.5.0 | SciPy | Δ |
| :--- | :---: | :---: | :---: |
| **Welch's T-Test** | $t=4.7522$, $p=2.162\times10^{-6}$ | $t=4.7522$, $p=2.162\times10^{-6}$ | $< 10^{-7}$ ✅ |

> Statistical tests використовують окрему реалізацію з `scipy.stats.ttest_ind` для верифікації.

---

### 💬 NLP: VADER Sentiment

| Sentence | SwiftSci 3.5.0 | NLTK | Match |
| :--- | :---: | :---: | :---: |
| *"SwiftSci 3.5.0 is incredibly fast and robust!"* | Compound=0.8519 | Compound=0.8519 | ✅ 100% |

---

## ⚡ Runtime Performance (Debug Build, Apple Silicon arm64)

```
  Benchmark                                    SwiftSci 3.5.0        Python (sklearn)        Speedup
  ──────────────────────────────────────────────────────────────────────────────────────────────────
  Holt-Winters fit + forecast (N=500, h=24)    0.156 ± 0.006 ms      ~12 ms                  ⚡ ~77×
  ARIMA(1,1,1) fit + forecast (N=500, h=24)    1.331 ± 0.028 ms      ~213 ms                 ⚡ ~160×
  GBDT Regressor fit + predict (1k samples)    88.555 ± 0.757 ms     ~188 ms                 ⚡ ~2.1×
  RandomForest fit + predict (1k samples)      24.569 ± 0.192 ms     ~58 ms                  ⚡ ~2.4×
  NaiveBayes fit + predict (1k samples)        11.104 ± 0.090 ms     ~21 ms                  ⚡ ~1.9×
```

---

## 🔬 Відтворення результатів

### Swift (Native):
```bash
swift run SwiftSciBenchmarks --suite Accuracy
```

### Python (ідентичні датасети через LCG-реплікацію):
```bash
python3 Benchmarks/Python/accuracy_benchmarks.py
```

---

## 🛡️ Гарантії чисельної стабільності

1. **IEEE 754 Double Precision (64-bit):** Всі обчислення в `SwiftStats`, `SwiftForecast`, `SwiftML`.
2. **Accelerate BLAS/LAPACK:** Матричні розкладання з перевіркою умовного числа та структурованими помилками `SwiftMLError.singularMatrix`.
3. **Swift 6 Strict Concurrency:** Детермінізм у паралельних обчисленнях без data races.
4. **Ідентичні датасети:** Python-реплікація `BenchmarkLCG` (LCG `a=6364136223846793005`, `c=1442695040888963407`) гарантує побітово ідентичну генерацію даних.
