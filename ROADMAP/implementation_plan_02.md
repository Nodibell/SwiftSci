# Implementation Plan — Version 0.2: Передобробка та Лінійні Моделі

**Версія:** 0.2  
**Модулі:** `SwiftPreprocessing`, `SwiftML`  
**Мета:** Розширити екосистему інструментами нормалізації/кодування ознак та базовими лінійними моделями машинного навчання, оптимізованими для GPU Apple Silicon через бібліотеку `mlx-swift`.

---

## Зміст

1. [Огляд архітектури](#1-огляд-архітектури)
2. [Package.swift — структура залежностей та таргетів](#2-packageswift--структура-залежностей-та-таргетів)
3. [SwiftPreprocessing](#3-swiftpreprocessing)
4. [SwiftML (на базі MLX)](#4-swiftml-на-базі-mlx)
5. [Тести та верифікація](#5-тести-та-верифікація)
6. [Апаратна маршрутизація та обмеження](#6-апаратна-маршрутизація-та-обмеження)
7. [Критерії виходу (Definition of Done)](#7-критерії-виходу-definition-of-done)
8. [Відомі ризики та мітигація v0.2](#8-відомі-ризики-та-мітигація-v02)

---

## 1. Огляд архітектури

У цій фазі ми інтегруємо апаратне прискорення від Apple (через Metal) за допомогою фреймворку MLX.

```
┌──────────────────────────────────────────────────────────┐
│                    Клієнтський код                       │
└────────────────────┬──────────────────┬──────────────────┘
                     │                  │
          ┌──────────▼─────────┐   ┌────▼──────────────┐
          │  SwiftPreprocessing│   │     SwiftML       │
          │  (Scaler, Encoder) │   │(Linear/Logistic)  │
          └──────────┬─────────┘   └────┬──────────────┘
                     │                  │
          ┌──────────▼─────────┐   ┌────▼──────────────┐
          │   SwiftDataFrame   │   │     mlx-swift     │
          │    (Data Input)    │   │  (Metal / GPU)    │
          └────────────────────┘   └───────────────────┘
```

> **Важливо:** Модуль `SwiftPreprocessing` працює з даними `DataFrame` та вихідними числовими масивами Swift, готуючи їх до завантаження у тензори `MLXArray`. Модуль `SwiftML` повністю базується на `MLXArray` для обчислення ваг, градієнтів та виконання прогнозів.

---

## 2. Package.swift — структура залежностей та таргетів

Додаємо пакет `mlx-swift` як залежність та визначаємо два нові модулі:

```swift
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SwiftAnalytics",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
        .visionOS(.v2)
    ],
    products: [
        .library(name: "SwiftDataFrame",    targets: ["SwiftDataFrame"]),
        .library(name: "SwiftStats",        targets: ["SwiftStats"]),
        .library(name: "SwiftPreprocessing",targets: ["SwiftPreprocessing"]),
        .library(name: "SwiftML",           targets: ["SwiftML"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/apache/arrow-swift.git",
            from: "21.0.0"
        ),
        .package(
            url: "https://github.com/google/flatbuffers.git",
            exact: "25.2.10"
        ),
        .package(
            url: "https://github.com/ml-explore/mlx-swift.git",
            from: "0.31.5"
        )
    ],
    targets: [
        // ── SwiftDataFrame (v0.1) ───────────────────────────
        .target(
            name: "SwiftDataFrame",
            dependencies: [
                .product(name: "Arrow", package: "arrow-swift")
            ],
            path: "Sources/SwiftDataFrame"
        ),
        .testTarget(
            name: "SwiftDataFrameTests",
            dependencies: ["SwiftDataFrame"],
            path: "Tests/SwiftDataFrameTests"
        ),

        // ── SwiftStats (v0.1) ───────────────────────────────
        .target(
            name: "SwiftStats",
            dependencies: ["SwiftDataFrame"],
            path: "Sources/SwiftStats",
            linkerSettings: [.linkedFramework("Accelerate")]
        ),
        .testTarget(
            name: "SwiftStatsTests",
            dependencies: ["SwiftStats"],
            path: "Tests/SwiftStatsTests"
        ),

        // ── SwiftPreprocessing (v0.2) ───────────────────────
        .target(
            name: "SwiftPreprocessing",
            dependencies: [
                "SwiftDataFrame",
                .product(name: "MLX", package: "mlx-swift")
            ],
            path: "Sources/SwiftPreprocessing",
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "SwiftPreprocessingTests",
            dependencies: ["SwiftPreprocessing"],
            path: "Tests/SwiftPreprocessingTests"
        ),

        // ── SwiftML (v0.2) ──────────────────────────────────
        .target(
            name: "SwiftML",
            dependencies: [
                "SwiftPreprocessing",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift")
            ],
            path: "Sources/SwiftML",
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(
            name: "SwiftMLTests",
            dependencies: ["SwiftML"],
            path: "Tests/SwiftMLTests"
        )
    ]
)
```

---

## 3. SwiftPreprocessing

Модуль надає API для підготовки ознак до машинного навчання.

### 3.1 Публічний API

#### `StandardScaler`
Здійснює масштабування ознак шляхом вирахування середнього значення та ділення на стандартне відхилення:
$$z = \frac{x - \mu}{\sigma}$$

```swift
public final class StandardScaler: Sendable {
    public private(set) var mean: [Double]?
    public private(set) var std: [Double]?
    
    public init() {}
    
    /// Навчає масштабувальник на основі вхідних 2D даних
    public func fit(_ data: [[Double]]) throws
    
    /// Трансформує вхідні дані за допомогою обчислених mean та std
    public func transform(_ data: [[Double]]) throws -> [[Double]]
    
    /// Виконує fit та transform за один крок
    public func fitTransform(_ data: [[Double]]) throws -> [[Double]]
}
```

#### `MinMaxScaler`
Масштабує ознаки у заданий числовий діапазон (за замовчуванням $[0, 1]$):
$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}} \times (max - min) + min$$

```swift
public final class MinMaxScaler: Sendable {
    public private(set) var dataMin: [Double]?
    public private(set) var dataMax: [Double]?
    public let range: (min: Double, max: Double)
    
    public init(range: (Double, Double) = (0.0, 1.0))
    
    public func fit(_ data: [[Double]]) throws
    public func transform(_ data: [[Double]]) throws -> [[Double]]
    public func fitTransform(_ data: [[Double]]) throws -> [[Double]]
}
```

#### `LabelEncoder`
Кодує категорії (текстові або числові) у вигляді послідовних цілих чисел від $0$ до $C-1$.

```swift
public final class LabelEncoder: Sendable {
    public private(set) var classes: [String] = []
    
    public init() {}
    
    public func fit(_ categories: [String])
    public func transform(_ categories: [String]) throws -> [Int]
    public func inverseTransform(_ labels: [Int]) throws -> [String]
    public func fitTransform(_ categories: [String]) throws -> [Int] {
        fit(categories)
        return try transform(categories)
    }
}
```

#### `OneHotEncoder`
Кодує категоричні ознаки у вигляді бінарної матриці (one-hot vectors).

```swift
public final class OneHotEncoder: Sendable {
    public private(set) var categories: [[String]] = []
    
    public init() {}
    
    public func fit(_ data: [[String]])
    public func transform(_ data: [[String]]) throws -> [[Double]]
    public func fitTransform(_ data: [[String]]) throws -> [[Double]]
}
```

### 3.2 Файлова структура `SwiftPreprocessing`

```
Sources/SwiftPreprocessing/
├── Core/
│   ├── StandardScaler.swift
│   ├── MinMaxScaler.swift
│   ├── LabelEncoder.swift
│   └── OneHotEncoder.swift
├── Errors/
│   └── PreprocessingError.swift
```

---

## 4. SwiftML (на базі MLX)

Модуль використовує фреймворк `MLX` та його функціональні трансформації для навчання лінійних моделей.

### 4.1 Публічний API

#### `LinearRegression`
Реалізує класичну лінійну регресію з навчанням методом градієнтного спуску.

```swift
import MLX

public final class LinearRegression {
    public private(set) var weights: MLXArray?
    public private(set) var bias: MLXArray?
    
    public init() {}
    
    /// Навчає модель на основі вхідних ознак X та цільової змінної y
    /// - Parameters:
    ///   - X: Матриця ознак розмірності [N, M]
    ///   - y: Вектор відповідей розмірності [N, 1] або [N]
    ///   - lr: Learning rate (швидкість навчання)
    ///   - epochs: Кількість ітерацій навчання
    public func fit(
        X: MLXArray,
        y: MLXArray,
        learningRate lr: Float = 0.01,
        epochs: Int = 1000
    ) throws
    
    /// Обчислює прогноз для вхідних ознак
    public func predict(X: MLXArray) throws -> MLXArray
}
```

#### `LogisticRegression`
Реалізує бінарну логістичну регресію зі згладжуванням (sigmoid activation) та оптимізацією крос-ентропії.

```swift
import MLX

public final class LogisticRegression {
    public private(set) var weights: MLXArray?
    public private(set) var bias: MLXArray?
    
    public init() {}
    
    public func fit(
        X: MLXArray,
        y: MLXArray,
        learningRate lr: Float = 0.1,
        epochs: Int = 1000
    ) throws
    
    /// Повертає ймовірності приналежності до класу 1 (значення від 0 до 1)
    public func predictProbability(X: MLXArray) throws -> MLXArray
    
    /// Повертає дискретні класи (0 або 1) на основі порогу відсікання (threshold)
    public func predict(X: MLXArray, threshold: Float = 0.5) throws -> MLXArray
}
```

### 4.2 Алгоритм оптимізації (valueAndGrad)

Обидві моделі використовують диференціювання MLX. Приклад внутрішнього циклу навчання логістичної регресії:

```swift
// Внутрішня функція втрат (Binary Cross Entropy)
private func loss(weights: MLXArray, bias: MLXArray, X: MLXArray, y: MLXArray) -> MLXArray {
    let logits = matmul(X, weights) + bias
    return mean(sigmoidCrossEntropy(logits: logits, labels: y))
}

// У циклі навчання:
let gradFn = valueAndGrad(loss)
for epoch in 0..<epochs {
    // gradFn повертає кортеж (lossValue, (gradWeights, gradBias))
    let (lossVal, grads) = gradFn(weights, bias, X, y)
    
    weights = weights - lr * grads.0
    bias = bias - lr * grads.1
    
    // Примусове обчислення графа MLX для уникнення витоку пам'яті
    eval(weights, bias)
}
```

### 4.3 Файлова структура `SwiftML`

```
Sources/SwiftML/
├── Core/
│   ├── LinearRegression.swift
│   ├── LogisticRegression.swift
├── Losses/
│   └── Losses.swift
├── Errors/
│   └── MLError.swift
```

---

## 5. Тести та верифікація

### Структура тестів

```
Tests/
├── SwiftPreprocessingTests/
│   ├── StandardScalerTests.swift
│   ├── MinMaxScalerTests.swift
│   ├── LabelEncoderTests.swift
│   └── OneHotEncoderTests.swift
└── SwiftMLTests/
    ├── LinearRegressionTests.swift
    └── LogisticRegressionTests.swift
```

### Обов'язкові тест-кейси

#### `StandardScaler`
- `[x]` Масштабування константного стовпця не призводить до ділення на 0 (std стає 1 або повертає 0).
- `[x]` Масштабування випадкової матриці: результуючі колонки мають mean = 0 та std = 1 (в межах похибки $10^{-6}$).

#### `OneHotEncoder`
- `[x]` Кодування декількох колонок з різною кількістю категорій.
- `[x]` Поява невідомої категорії при `transform` генерує відповідну помилку `PreprocessingError.unknownCategory`.

#### `LinearRegression` (Верифікація проти scikit-learn)
- `[x]` Навчання на штучному лінійному датасеті ($y = 2X_1 + 3X_2 + 5 + \epsilon$). Модель повинна зійтися до коефіцієнтів $w \approx [2, 3]$ та $b \approx 5$.
- `[x]` Прогноз на нових даних має низьку середньоквадратичну помилку ($MSE < 0.01$).

#### `LogisticRegression`
- `[x]` Навчання на лінійно роздільному бінарному датасеті.
- `[x]` Точність класифікації (Accuracy) на тестовій вибірці становить $> 95\%$.

---

## 6. Апаратна маршрутизація та обмеження

1. **Metal GPU Acceleration**: MLX автоматично виконує обчислення тензорів на інтегрованому графічному процесорі Apple Silicon (Metal). Жодного додаткового коду для GPU-маршрутизації писати не потрібно.
2. **Swift 6 Concurrency**: Об'єкти `MLXArray` наразі не реалізують нативний протокол `Sendable`. Тому всі тренування моделей та трансформації мають відбуватися в одному потоці або бути інкапсульованими всередині окремого Isolated Actor/Task без передачі проміжних не-Sendable об'єктів між контекстами.

---

## 7. Критерії виходу (Definition of Done)

- `[x]` `swift build` успішно збирається на macOS 15+ без попереджень та помилок.
- `[x]` Всі тест-кейси Preprocessing та ML проходять успішно (`swift test`).
- `[x]` Покриття юніт-тестами нових модулів становить $\ge 80\%$.
- `[x]` Відповідність результатів нормалізації та навчання моделей аналогам зі scikit-learn (допустиме відхилення $\le 10^{-4}$).
- `[x]` Повна ізоляція не-Sendable об'єктів MLX відповідно до вимог Swift 6.

---

## 8. Відомі ризики та мітигація v0.2

| Ризик | Ступінь | Дія з мітигації |
|---|---|---|
| Не-Sendable статус `MLXArray` викликає збої компілятора у Swift 6 | Висока | Локалізувати використання `MLXArray` всередині методів та приватних структур, не виносити їх у публічні асинхронні інтерфейси без копіювання у стандартні типи. |
| Витік пам'яті при лінивих обчисленнях MLX | Середня | Забезпечити регулярний виклик `eval()` у циклі навчання градієнтного спуску для очищення графів виконання. |
| Несумісність версії `mlx-swift` з версією macOS у користувача | Низька | Обмежити мінімальну версію macOS до 15.0 на рівні пакета. |
