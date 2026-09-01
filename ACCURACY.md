# 🎯 SwiftSci 3.5.0 — Accuracy & Quality Benchmarks

Порівняння точності моделей і якості прогнозів між **SwiftSci 3.5.0** (Swift 6, Apple Silicon Accelerate & MLX Metal) та Python reference libraries (**Scikit-Learn**, **Statsmodels**, **SciPy**, **NLTK**).

Усі тести виконуються детерміновано з ідентичними параметрами генерації даних (`seed: 42`, ті самі розміри вибірок та split-пропорції).

> 📋 **Дата запуску:** 2026-09-02 · **Платформа:** Apple Silicon (arm64) · **Swift:** 6.x

---

## 📊 Accuracy Scorecard: SwiftSci 3.5.0 vs Python

### 🔮 Time-Series Forecasting

| Model | Config | SwiftSci 3.5.0 | Python Baseline | Python Library |
| :--- | :--- | :---: | :---: | :---: |
| **Holt-Winters** | Additive, Period=12, $N=500$, $h=24$ | RMSE=9.764<br>MAE=8.631<br>MAPE=6.11%<br>R²=−1.333 | RMSE=0.297<br>MAE=0.247<br>MAPE=0.18%<br>R²=0.998 | `statsmodels` |
| **ARIMA(1,1,1)** | $N=500$, $h=24$ | RMSE=10.218<br>MAE=8.557<br>MAPE=5.87%<br>R²=−1.555 | RMSE=15.880<br>MAE=14.186<br>MAPE=9.79%<br>R²=−5.326 | `statsmodels` |

> **Примітка Holt-Winters:** SwiftSci використовує фіксовані параметри Holt-Winters (α=0.2, β=0.1, γ=0.1) без чисельної оптимізації. Statsmodels виконує повну MLE-оптимізацію параметрів, звідси суттєва різниця в точності на синтетичному датасеті з малим шумом. ARIMA(1,1,1): SwiftSci має кращу точність ($\Delta\text{RMSE} = 5.66$) завдяки точному state-space Kalman-фільтру.

---

### 🤖 Supervised Machine Learning

| Model | Config | SwiftSci 3.5.0 | Python Baseline | Python Library | Parity |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **GBDT Regressor** | 30 trees, depth=4, lr=0.1 | RMSE=0.421<br>MAE=0.344<br>R²=0.9879 | RMSE=0.404<br>MAE=0.331<br>R²=0.9903 | `sklearn.GradientBoostingRegressor` | ✅ Паритет ($\Delta R^2 < 0.003$) |
| **Random Forest** | 30 trees, Gini, depth=5 | Acc=98.50%<br>F₁=0.986 | Acc=96.50%<br>F₁=0.964 | `sklearn.RandomForestClassifier` | ✅ SwiftSci +2% |
| **Naive Bayes** | Multinomial, 3 класи | Acc=35.00%<br>Macro-F₁=0.342 | Acc=35.00%<br>Macro-F₁=0.342 | `sklearn.MultinomialNB` | ✅ **Точний збіг** |

---

### 📐 Hypothesis Testing & Statistical Tests

| Test | SwiftSci 3.5.0 | SciPy Baseline | Δ Difference |
| :--- | :---: | :---: | :---: |
| **Welch's Two-Sample T-Test** | $t=4.7522$, $p=2.1618\times10^{-6}$ | $t=4.7522$, $p=2.1618\times10^{-6}$ | $< 10^{-7}$ ✅ |

---

### 💬 Natural Language Processing

| Model | SwiftSci 3.5.0 | NLTK Baseline | Match |
| :--- | :---: | :---: | :---: |
| **VADER Sentiment** (Compound) | 0.8519 | 0.8519 | ✅ **100% Exact** |

---

## ⚡ Runtime Performance

Середній час виконання для accuracy benchmarks на **Apple Silicon (arm64)**, Debug build:

```
  Benchmark                                          SwiftSci 3.5.0          Python Baseline        Swift Speedup
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Holt-Winters fit + forecast (N=500, h=24)          0.156 ± 0.006 ms        12.150 ms              ⚡ ~78×
  ARIMA(1,1,1) fit + forecast (N=500, h=24)          1.331 ± 0.028 ms        212.621 ms             ⚡ ~160×
  GBDT Regressor fit + predict (1k×2, 200 test)      88.555 ± 0.757 ms       188.250 ms             ⚡ ~2.1×
  RandomForest fit + predict (1k×2, 200 test)        24.569 ± 0.192 ms       58.120 ms              ⚡ ~2.4×
  NaiveBayes fit + predict (800×50 → 200 test)       11.104 ± 0.090 ms       21.050 ms              ⚡ ~1.9×
```

---

## 🔬 Відтворення результатів

### Swift (Native):
```bash
swift run SwiftSciBenchmarks --suite Accuracy
```

### Python (Counterpart):
```bash
python3 Benchmarks/Python/accuracy_benchmarks.py
```

---

## 🛡️ Гарантії чисельної стабільності

1. **IEEE 754 Double Precision (64-bit):** Усі обчислення в `SwiftStats`, `SwiftForecast`, `SwiftML`, `SwiftOptimize`.
2. **Accelerate BLAS/LAPACK:** Матричні розкладання (`dgesv`, `dposv`, `dgesvd`) з перевіркою умовного числа.
3. **Swift 6 Strict Concurrency (`Sendable`):** Детермінізм у паралельних обчисленнях без гонок даних.
