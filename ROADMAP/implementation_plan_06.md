# План реалізації — Версія 0.6: SwiftForecast (Аналіз часових рядів)

**Версія:** 0.6  
**Новий модуль:** `SwiftForecast`  
**Залежності:** `Accelerate` (vDSP + LAPACK), `SwiftStats`, `SwiftDataFrame`  
**Мета:** Нативна реалізація на Swift найважливіших алгоритмів прогнозування часових рядів — експоненційного згладжування (SES, метод Хольта, Хольта-Вінтерса), ARIMA та фільтра Калмана — з використанням `Accelerate` для всіх складних обчислень та дотриманням суворої конкурентності Swift 6 (`actor` + `Sendable`).

---

## Зміст

1. [Архітектурне дослідження](#1-архітектурне-дослідження)
2. [Структура модуля](#2-структура-модуля)
3. [Типи даних та помилки](#3-типи-даних-та-помилки)
4. [Детальний опис функцій](#4-детальний-опис-функцій)
   - 4.1 [Декомпозиція часових рядів](#41-декомпозиція-часових-рядів)
   - 4.2 [Сімейство алгоритмів експоненційного згладжування](#42-сімейство-алгоритмів-експоненційного-згладжування)
   - 4.3 [ARIMA](#43-arima)
   - 4.4 [Фільтр Калмана](#44-фільтр-калмана)
5. [Зміни в Package.swift](#5-зміни-в-packageswift)
6. [План структури файлів](#6-план-структури-файлів)
7. [План тестування](#7-план-тестування)
8. [Інтеграція в Demo](#8-інтеграція-в-demo)
9. [Відкриті питання](#9-відкриті-питання)

---

## 1. Архітектурне дослідження

### Чому не MLX?

В алгоритмах часових рядів домінують **послідовні залежності даних** (кожен крок залежить від попереднього), що робить їх неефективними для паралельних обчислень на GPU. Правильним вибором бекенду є `Accelerate`:

| Операція | Найкращий бекенд |
|---|---|
| Згортка сигналів (Smoothing MA) | `vDSP.convolve` |
| Розв'язання матричних рівнянь (ARIMA OLS, Kalman gain) | `LAPACK.dgesv` / `cblas_dgemm` |
| Швидке перетворення Фур'є (виявлення періодичності) | `vDSP.FFT` |
| Векторні операції (залишки, RMSE) | `vDSP.*` |

### Модель конкурентності (Concurrency Model)

Відповідно до шаблонів версії v0.4:
- **Моделі зі станом** (`ExponentialSmoothing`, `ARIMA`, `KalmanFilter`) → `actor`
- **Перетворення без стану** (`TimeSeriesDecomposition`, `ADF test`) → `static func` у `enum`
- Усі **структури результатів** → `Sendable`, типи значень (value types)
- **Без залежності від MLX** → для цього модуля `WiredMemoryTicket` не потрібен

### Порівняння з еквівалентами в Python

| Python | Еквівалент у SwiftForecast |
|---|---|
| `statsmodels.tsa.holtwinters.ExponentialSmoothing` | `ExponentialSmoothing` (actor) |
| `statsmodels.tsa.arima.model.ARIMA` | `ARIMAModel` (actor) |
| `filterpy.kalman.KalmanFilter` | `KalmanFilter` (actor) |
| `statsmodels.tsa.seasonal.seasonal_decompose` | `TimeSeriesDecomposition.decompose()` |

---

## 2. Структура модуля

```
Sources/SwiftForecast/
├── SwiftForecast.swift                  # Точка входу в модуль / публічний реекспорт
├── Core/
│   ├── TimeSeriesDecomposition.swift    # Декомпозиція Тренд + Сезонність + Залишки
│   ├── ExponentialSmoothing.swift       # SES, метод Хольта (DES), Хольта-Вінтерса (actor)
│   ├── ARIMA.swift                      # ARIMA(p,d,q) (actor)
│   └── KalmanFilter.swift              # Лінійний фільтр Калмана (actor)
├── Errors/
│   └── ForecastError.swift             # Enum ForecastError (Sendable, Equatable)
└── Results/
    └── ForecastResults.swift           # Усі структури результатів (Sendable)

Tests/SwiftForecastTests/
├── DecompositionTests.swift
├── ExponentialSmoothingTests.swift
├── ARIMATests.swift
└── KalmanFilterTests.swift
```

---

## 3. Типи даних та помилки

### 3.1 ForecastError

```swift
public enum ForecastError: Error, Sendable, Equatable {
    // Валідація вхідних даних
    case emptyTimeSeries
    case insufficientLength(minimum: Int, got: Int)
    case containsNaN
    case containsInfinity

    // Валідація параметрів
    case invalidAlpha(Double)           // має бути в діапазоні (0, 1)
    case invalidBeta(Double)
    case invalidGamma(Double)
    case invalidPeriod(Int)             // має бути >= 2
    case invalidAROrder(Int)            // p має бути >= 0
    case invalidDifferencing(Int)       // d має бути >= 0
    case invalidMAOrder(Int)            // q має бути >= 0
    case invalidHorizon(Int)            // має бути >= 1

    // Обчислювальні помилки
    case notFitted
    case convergenceFailed(iterations: Int)
    case singularMatrix
    case matrixDimensionMismatch(expected: (Int, Int), got: (Int, Int))
}
```

### 3.2 Структури результатів (усі `Sendable`)

```swift
/// Компоненти декомпозиції часового ряду.
public struct DecompositionResult: Sendable {
    public let trend:      [Double]
    public let seasonal:   [Double]
    public let residual:   [Double]
    public let original:   [Double]
}

/// Вихідні дані прогнозу з опціональними довірчими інтервалами.
public struct ForecastResult: Sendable {
    public let predictions:   [Double]
    public let lowerBound:    [Double]?  // Нижня межа 95% ДІ
    public let upperBound:    [Double]?  // Верхня межа 95% ДІ
    public let fittedValues:  [Double]   // Згладжені значення
    public let residuals:     [Double]
    public let aic:           Double?
    public let mse:           Double
    public let mae:           Double
}

/// Результат роботи моделі ARIMA.
public struct ARIMAResult: Sendable {
    public let order:          (p: Int, d: Int, q: Int)
    public let arCoefficients: [Double]
    public let maCoefficients: [Double]
    public let intercept:      Double
    public let forecast:       ForecastResult
}

/// Оцінка стану у фільтрі Калмана.
public struct KalmanState: Sendable {
    public let mean:       [Double]    // Вектор стану x_hat
    public let covariance: [[Double]]  // Матриця коваріації P (рядок за рядком)
}
```

---

## 4. Детальний опис функцій

### 4.1 Декомпозиція часових рядів

**Файл:** `Core/TimeSeriesDecomposition.swift`  
**Тип:** `enum TimeSeriesDecomposition` (без стану, чисті статичні функції)

#### Класична декомпозиція

Розкладає часовий ряд на `тренд + сезонність + залишки` (адитивна модель) або
`тренд × сезонність × залишки` (мультиплікативна модель).

**Алгоритм:**
1. **Тренд** — центроване ковзне середнє з вікном `= період` з використанням `vDSP`.
2. **Сезонність** — усереднення значень без тренду для кожної позиції періоду; нормалізація.
3. **Залишки** — `original - trend - seasonal` (адитивна) або `original / (trend * seasonal)` (мультиплікативна).

```swift
public enum DecompositionModel: Sendable { case additive, multiplicative }

public enum TimeSeriesDecomposition {
    public static func decompose(
        series: [Double],
        period: Int,
        model: DecompositionModel = .additive
    ) throws -> DecompositionResult

    /// Автокореляційна функція до лагу maxLag.
    public static func acf(series: [Double], maxLag: Int) throws -> [Double]

    /// Часткова автокореляція (рівняння Юла-Уокера через LAPACK dgesv).
    public static func pacf(series: [Double], maxLag: Int) throws -> [Double]

    /// Критерій Дікі-Фуллера (ADF) на стаціонарність.
    /// Повертає статистику критерію та наближене p-value.
    public static func adfTest(series: [Double], maxLag: Int = 1) throws -> (statistic: Double, pValue: Double)
}
```

**Крайні випадки:**
- `period >= series.count / 2` → помилка `.insufficientLength`
- Парний чи непарний період (особливості центрування ковзного середнього)
- Поширення NaN на краях тренду (обрізання країв замість заповнення)

---

### 4.2 Сімейство алгоритмів експоненційного згладжування

**Файл:** `Core/ExponentialSmoothing.swift`  
**Тип:** `actor ExponentialSmoothing`

```swift
public enum SmoothingMethod: Sendable {
    case simple                                               // SES: лише альфа
    case double(beta: Double)                                 // Метод Хольта: альфа + бета
    case holtWinters(beta: Double, gamma: Double,
                     period: Int,
                     seasonal: DecompositionModel)
}

public actor ExponentialSmoothing {
    public init(method: SmoothingMethod, alpha: Double? = nil)
    public func fit(series: [Double]) async throws
    public func forecast(horizon: Int) throws -> ForecastResult
    public func fittedValues() throws -> [Double]
}
```

#### SES (Просте експоненційне згладжування)
`y_hat_{t+1} = alpha * y_t + (1-alpha) * y_hat_t`
- Автооптимізація параметра alpha через пошук по сітці (grid search) для мінімізації MSE, якщо передано `alpha: nil`
- Прогноз на h кроків вперед = останній обчислений рівень (постійне значення)

#### Двопараметричне згладжування Хольта (Holt's Double Exponential Smoothing)
```
l_t = alpha * y_t + (1-alpha)(l_{t-1} + b_{t-1})
b_t = beta  * (l_t - l_{t-1}) + (1-beta) * b_{t-1}
y_hat_{t+h} = l_t + h * b_t
```
- Пошук по сітці 11×11 для параметрів alpha/beta з метою мінімізації MSE на навчальній вибірці

#### Трипараметричний метод Хольта-Вінтерса (період m)
**Адитивна модель:**
```
l_t = alpha(y_t - s_{t-m}) + (1-alpha)(l_{t-1} + b_{t-1})
b_t = beta(l_t - l_{t-1}) + (1-beta) * b_{t-1}
s_t = gamma(y_t - l_{t-1} - b_{t-1}) + (1-gamma) * s_{t-m}
y_hat_{t+h} = l_t + h*b_t + s_{t + h - m*floor((h-1)/m)}
```
**Мультиплікативна модель:** використовує ділення на сезонний коефіцієнт замість віднімання.

---

### 4.3 ARIMA

**Файл:** `Core/ARIMA.swift`  
**Тип:** `actor ARIMAModel`

ARIMA(p, d, q):
- **I(d):** взяття різниць ряду `d` разів за допомогою `vDSP`
- **Оцінка параметрів:** Умовний метод найменших квадратів (OLS via Hannan-Rissanen)
  1. Взяття різниць ряду d разів.
  2. Підгонка високорангової моделі авторегресії (AR) для отримання наближених залишків (проксі для членів MA).
  3. Побудова матриці плану `[1, y_{t-1}..y_{t-p}, e_{t-1}..e_{t-q}]`.
  4. Розв'язання МНК через `LAPACK dgelsd` (метод найменших квадратів на основі SVD).
- **Прогноз:** рекурсивне підставлення значень та інтегрування (зворотне взяття різниць).

```swift
public actor ARIMAModel {
    public let order: (p: Int, d: Int, q: Int)
    public init(p: Int, d: Int, q: Int) throws
    public func fit(series: [Double]) throws
    public func forecast(horizon: Int) throws -> ARIMAResult
    public func aic() throws -> Double   // AIC = 2k - 2*ln(L_hat)
}
```

**Внутрішні допоміжні функції (private):**
- `difference(series:times:)` → `[Double]`
- `integrate(differences:originalHead:times:)` → інтегрування різниць
- `buildDesignMatrix(series:residuals:p:q:)` → `[[Double]]`
- `leastSquares(A:b:)` → `[Double]` через `cblas_dgemm` + `LAPACK dgelsd`

---

### 4.4 Фільтр Калмана

**Файл:** `Core/KalmanFilter.swift`  
**Тип:** `actor KalmanFilter`

Лінійна модель у просторі станів:
```
x_t = F * x_{t-1} + w_t    (процесний шум w ~ N(0, Q))
z_t = H * x_t    + v_t     (шум вимірювання v ~ N(0, R))
```

Цикл "Прогноз-Корекція" з використанням `cblas_dgemm` (множення матриць) та `LAPACK dgesv` (обернення матриць).

```swift
public actor KalmanFilter {
    public init(stateSize: Int, observationSize: Int) throws
    public func setTransitionMatrix(_ F: [[Double]]) throws        // n×n
    public func setObservationMatrix(_ H: [[Double]]) throws       // m×n
    public func setProcessNoise(_ Q: [[Double]]) throws            // n×n
    public func setMeasurementNoise(_ R: [[Double]]) throws        // m×m
    public func setInitialState(mean: [Double], covariance: [[Double]]) throws

    /// Фільтрація спостережень. Повертає масив оцінок станів KalmanState для кожного кроку.
    public func filter(observations: [[Double]]) throws -> [KalmanState]

    /// Згладжування за алгоритмом RTS (Rauch-Tung-Striebel) для офлайн-обробки.
    public func smooth(observations: [[Double]]) throws -> [KalmanState]

    /// Прогноз на один крок вперед з поточного стану.
    public func predict() throws -> KalmanState
}

extension KalmanFilter {
    /// Попередньо налаштована одновимірна модель оцінки швидкості та координати для скалярного ряду.
    public static func oneDimensional(
        processNoise: Double,
        measurementNoise: Double
    ) async throws -> KalmanFilter
}
```

---

## 5. Зміни в Package.swift

Додати до `products:`:
```swift
.library(name: "SwiftForecast", targets: ["SwiftForecast"]),
```

Додати до `targets:`:
```swift
.target(
    name: "SwiftForecast",
    dependencies: [
        "SwiftDataFrame",
        "SwiftStats",
    ],
    path: "Sources/SwiftForecast",
    swiftSettings: globalSwiftSettings,
    linkerSettings: [
        .linkedFramework("Accelerate"),
    ]
),
.testTarget(
    name: "SwiftForecastTests",
    dependencies: ["SwiftForecast"],
    path: "Tests/SwiftForecastTests",
    swiftSettings: globalSwiftSettings
),
```

> **Примітка:** Без залежності від `MLX`. Використання `WiredMemoryTicket` не потрібне.

---

## 6. План структури файлів

| Файл | Статус | Опис |
|---|---|---|
| `Sources/SwiftForecast/SwiftForecast.swift` | НОВИЙ | Головний файл модуля |
| `Sources/SwiftForecast/Errors/ForecastError.swift` | НОВИЙ | Опис помилок `ForecastError` |
| `Sources/SwiftForecast/Results/ForecastResults.swift` | НОВИЙ | Структури результатів |
| `Sources/SwiftForecast/Core/TimeSeriesDecomposition.swift` | НОВИЙ | Декомпозиція та ACF/PACF/ADF |
| `Sources/SwiftForecast/Core/ExponentialSmoothing.swift` | НОВИЙ | SES, метод Хольта, метод Хольта-Вінтерса |
| `Sources/SwiftForecast/Core/ARIMA.swift` | НОВИЙ | Модель ARIMA(p,d,q) |
| `Sources/SwiftForecast/Core/KalmanFilter.swift` | НОВИЙ | Фільтр Калмана + RTS згладжування |
| `Tests/SwiftForecastTests/DecompositionTests.swift` | НОВИЙ | ≥ 8 тестів |
| `Tests/SwiftForecastTests/ExponentialSmoothingTests.swift` | НОВИЙ | ≥ 10 тестів |
| `Tests/SwiftForecastTests/ARIMATests.swift` | НОВИЙ | ≥ 8 тестів |
| `Tests/SwiftForecastTests/KalmanFilterTests.swift` | НОВИЙ | ≥ 8 тестів |
| `Package.swift` | ЗМІНА | Реєстрація нового модуля SwiftForecast |
| `ROADMAP/ROADMAP.md` | ЗМІНА | Оновлення статусу версії 0.6 |
| `SwiftAnalyticsDemo/SwiftAnalyticsDemo.swift` | ЗМІНА | Додавання демонстраційного коду v0.6 |

---

## 7. План тестування

### 7.1 Тестування декомпозиції

| # | Тест | Критерій успішності |
|---|---|---|
| 1 | Адитивна декомпозиція — відома синусоїда + лінійний тренд | середнє значення залишків < 0.01 |
| 2 | Мультиплікативна декомпозиція — експонента * сезонність | середнє сезонних коефіцієнтів ≈ 1.0 |
| 3 | Довжина тренду = довжина ряду - період + 1 | точна відповідність розмірності |
| 4 | період >= довжина/2 → викликає помилку `.insufficientLength` | помилка успішно оброблена |
| 5 | Автокореляція ACF на кроці 0 == 1.0 | точна рівність `acf[0] == 1.0` |
| 6 | ACF білого шуму: лаги від 1 до N ≈ 0 (±2/√N) | значення в межах довірчих меж |
| 7 | PACF розпізнає AR(2): лаги 1, 2 значущі; 3+ ≈ 0 | тест за порогами значущості |
| 8 | ADF для випадкового блукання: не можна відкинути одиничний корінь | `pValue > 0.05` |

### 7.2 Тестування експоненційного згладжування

| # | Тест | Критерій успішності |
|---|---|---|
| 1 | SES для константного ряду → MSE == 0 | `MSE == 0.0` |
| 2 | SES для лінійного тренду → MSE < дисперсії ряду | перевірка точності |
| 3 | SES з некоректною alpha → помилка `.invalidAlpha` | викидається виняток |
| 4 | Метод Хольта для лінійного ряду → MAE < 0.5 | висока точність тренду |
| 5 | Прогноз Хольта на горизонт 5 → `predictions.count == 5` | |
| 6 | Хольт-Вінтерс (адитивний), період=12, синусоїдальні дані → R² > 0.90 | |
| 7 | Хольт-Вінтерс (мультиплікативний) → відносна RMSE < 5% | |
| 8 | Виклик `.forecast` перед `.fit` → помилка `.notFitted` | |
| 9 | період=1 → помилка `.invalidPeriod` | |
| 10 | Автооптимація alpha ∈ (0, 1) дає меншу MSE, ніж фіксована α=0.5 | |

### 7.3 Тестування ARIMA

| # | Тест | Критерій успішності |
|---|---|---|
| 1 | ARIMA(0,0,0) на білому шумі: константа ≈ середнє ряду | `abs(intercept - mean) < 1.0` |
| 2 | Відновлення коефіцієнтів AR(1): генерований ряд з φ=0.8 → оцінка φ ∈ [0.70, 0.90] | |
| 3 | ARIMA(1,1,0) для випадкового блукання: залишки стаціонарні за ADF | |
| 4 | `forecast(horizon:6).predictions.count == 6` | |
| 5 | Негативний порядок p → помилка `.invalidAROrder` | |
| 6 | Занадто короткий ряд → помилка `.insufficientLength` | |
| 7 | Критерій AIC(1,0,0) < AIC(0,0,0) на AR(1) даних | менший AIC для правильної моделі |
| 8 | Кількість значень згладженого ряду == довжина ряду - d | |

### 7.4 Тестування фільтра Калмана

| # | Тест | Критерій успішності |
|---|---|---|
| 1 | 1D константа + Гаусів шум: RMSE фільтра < RMSE вихідних даних | підтверджене придушення шуму |
| 2 | 1D лінійний тренд: MSE фільтра < 1.0 | |
| 3 | кількість станів фільтра == кількості спостережень | |
| 4 | Матриця коваріації симетрична на кожному кроці | `P[i][j] == P[j][i]` |
| 5 | Коефіцієнт підсилення Калмана ∈ [0,1] для скалярів | |
| 6 | Сингулярна матриця R → помилка `.singularMatrix` | |
| 7 | Невідповідність розмірності матриці F → `.matrixDimensionMismatch` | |
| 8 | Дисперсія згладжувача RTS ≤ дисперсії фільтра вперед | RTS зменшує невизначеність |

### 7.5 Загальні вимоги до тестування

- Усі тести проходять `swift test` з увімкненою строгою перевіркою конкурентності `StrictConcurrency` (Swift 6).
- **Щонайменше 34 тести** сумарно в 4 файлах тестів.
- Жоден тест не повинен виконуватися довше ніж **2 секунди** на Apple Silicon M-серії.
- Збірка через `swift build` проходить без помилок та попереджень у коді розробника.

---

## 8. Інтеграція в Demo

Додати секцію `=== v0.6 SwiftForecast ===` у файл `SwiftAnalyticsDemo.swift`:

```swift
import SwiftForecast

// 13. Декомпозиція часового ряду (адитивна, період=12, 3 роки = 36 точок)
print("\n13. Декомпозиція часового ряду (адитивна, період=12)...")
let monthlySales = ... // синусоїда + тренд + шум
let decomposed = try TimeSeriesDecomposition.decompose(
    series: monthlySales, period: 12, model: .additive
)
print("Тренд (перші 6 точок): \(decomposed.trend.prefix(6))")
print("Сезонність (1 період): \(decomposed.seasonal.prefix(12))")

// 14. Прогноз методом Хольта-Вінтерса на 12 місяців
print("\n14. Прогноз методом Хольта-Вінтерса (горизонт=12)...")
let hw = ExponentialSmoothing(
    method: .holtWinters(beta: 0.1, gamma: 0.1, period: 12, seasonal: .additive)
)
try await hw.fit(series: monthlySales)
let hwForecast = try hw.forecast(horizon: 12)
print("Прогноз: \(hwForecast.predictions)")
print("RMSE: \(String(format: "%.4f", hwForecast.mse.squareRoot()))")

// 15. Модель ARIMA(1,1,0)
print("\n15. Прогноз ARIMA(1,1,0) (горизонт=6)...")
let arima = try ARIMAModel(p: 1, d: 1, q: 0)
try await arima.fit(series: monthlySales)
let arimaResult = try await arima.forecast(horizon: 6)
print("Прогнозовані значення: \(arimaResult.forecast.predictions)")
print("AIC: \(String(format: "%.2f", arimaResult.forecast.aic ?? 0))")

// 16. Фільтр Калмана на зашумленому сигналі
print("\n16. Фільтр Калмана (1D constant-velocity)...")
let kf = try await KalmanFilter.oneDimensional(processNoise: 0.1, measurementNoise: 1.0)
let noisyObs = noisySine(n: 30).map { [$0] }
let states = try await kf.filter(observations: noisyObs)
let filtered = states.map { $0.mean[0] }
print("Відфільтровані значення (перші 5): \(filtered.prefix(5).map { String(format: "%.3f", $0) })")
```

---

## 9. Відкриті питання

> [!IMPORTANT]
> **Q1 — Оптимізатор ARIMA:** Умовний метод OLS (Hannan-Rissanen) є швидким, але менш точним за оцінку максимальної правдоподібності (MLE). Використовувати OLS у v0.6 та відкласти реалізацію MLE (через L-BFGS) до фази бенчмарків v0.7?  
> **Рекомендація:** Так — OLS для v0.6, а додавання MLE запланувати у v0.7.

> [!IMPORTANT]
> **Q2 — Auto-ARIMA:** Чи повинен `SwiftForecast` включати автоматичний підбір порядку моделі через пошук за сіткою (p, d, q) ∈ {0..3} на основі критерію AIC?  
> **Рекомендація:** Опціональний бонус — додати структуру `AutoARIMA`, якщо дозволить час, але не блокувати цим реліз v0.6.

> [!NOTE]
> **Q3 — Багатовимірні часові ряди:** `KalmanFilter` вже підтримує багатовимірний простір станів. Чи варто додавати модель векторної авторегресії `VARModel` (Vector AutoRegression) у v0.6?  
> **Рекомендація:** Відкласти до v0.7 — Фільтр Калмана наразі повністю покриває основні багатовимірні кейси.

> [!NOTE]
> **Q4 — Довірчі інтервали:** Аналітичний довірчий інтервал для SES розраховується як `y_hat ± 1.96 * sigma * sqrt(1 + (h-1)*alpha^2)`. Розраховувати ДІ завжди чи зробити опціональним?  
> **Рекомендація:** Зробити опціональним за допомогою прапорця `computeCI: Bool = false` у методі `forecast(horizon:computeCI:)`.
