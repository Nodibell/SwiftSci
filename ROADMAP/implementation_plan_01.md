# Implementation Plan — Version 0.1: Фундамент Даних

**Версія:** 0.1
**Модулі:** `SwiftDataFrame`, `SwiftStats`
**Мета:** Закласти надійний фундамент для всієї екосистеми: нативне колоночне сховище даних та векторизована описова статистика.

---

## Зміст

1. [Огляд архітектури](#1-огляд-архітектури)
2. [Package.swift — структура](#2-packageswift--структура)
3. [SwiftDataFrame](#3-swiftdataframe)
4. [SwiftStats](#4-swiftstats)
5. [Тести та верифікація](#5-тести-та-верифікація)
6. [Апаратні обмеження та цільова платформа](#6-апаратні-обмеження-та-цільова-платформа)
7. [Критерії виходу (Definition of Done)](#7-критерії-виходу-definition-of-done)
8. [Відомі ризики v0.1](#8-відомі-ризики-v01)

---

## 1. Огляд архітектури

```
┌─────────────────────────────────────────────────────────┐
│                   Клієнтський код                       │
│         (використовує тільки public API)                │
└───────────────────┬─────────────────┬───────────────────┘
                    │                 │
         ┌──────────▼──────┐ ┌────────▼──────────┐
         │  SwiftDataFrame │ │    SwiftStats     │
         │  (public API)   │ │  (public API)     │
         └──────────┬──────┘ └────────┬──────────┘
                    │                 │
         ┌──────────▼──────┐ ┌────────▼──────────┐
         │  DataBuffer     │ │  vDSP + LAPACK    │
         │  (internal)     │ │  (Accelerate)     │
         └──────────┬──────┘ └───────────────────┘
                    │
         ┌──────────▼──────┐
         │  Apache Arrow   │
         │  arrow-swift    │
         │  >= 21.0.0      │
         └─────────────────┘
```

> **ВАЖЛИВО:** `DataBuffer` — ключовий internal-протокол. Arrow не повинен
> просочуватись у `public` API жодного модуля. Це захищає від
> breaking changes у самій бібліотеці Arrow Swift.

---

## 2. Package.swift — структура

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
        .library(name: "SwiftDataFrame", targets: ["SwiftDataFrame"]),
        .library(name: "SwiftStats",     targets: ["SwiftStats"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/apache/arrow-swift.git",
            from: "21.0.0"
        ),
    ],
    targets: [
        // ── SwiftDataFrame ──────────────────────────────────
        .target(
            name: "SwiftDataFrame",
            dependencies: [
                .product(name: "Arrow", package: "arrow-swift")
            ],
            path: "Sources/SwiftDataFrame",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "SwiftDataFrameTests",
            dependencies: ["SwiftDataFrame"],
            path: "Tests/SwiftDataFrameTests"
        ),

        // ── SwiftStats ──────────────────────────────────────
        .target(
            name: "SwiftStats",
            dependencies: ["SwiftDataFrame"],
            path: "Sources/SwiftStats",
            linkerSettings: [
                .linkedFramework("Accelerate")
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "SwiftStatsTests",
            dependencies: ["SwiftStats"],
            path: "Tests/SwiftStatsTests"
        ),
    ]
)
```

---

## 3. SwiftDataFrame

### 3.1 Публічний API

#### Типи даних колонок

```swift
/// Підтримувані типи колонок у v0.1
public enum ColumnDType: Sendable {
    case int32
    case int64
    case float32
    case float64
    case boolean
    case utf8       // рядки
    case date32     // дати (дні від епохи Unix)
}
```

#### Протокол `AnyColumn`

```swift
public protocol AnyColumn: Sendable {
    var name: String       { get }
    var dtype: ColumnDType { get }
    var count: Int         { get }
    var nullCount: Int     { get }

    func filtered(by mask: [Bool]) throws -> any AnyColumn
}
```

#### `TypedColumn<T>`

```swift
public struct TypedColumn<T: SupportedType>: AnyColumn {
    public let name: String
    public let dtype: ColumnDType
    public var count: Int      { values.count }
    public var nullCount: Int  { values.filter { $0 == nil }.count }

    /// Елементи у вигляді Optional — nil означає null у Arrow
    public let values: [T?]

    public subscript(index: Int) -> T? { values[index] }

    public func filtered(by mask: [Bool]) throws -> any AnyColumn
    public func map<U: SupportedType>(_ transform: (T?) -> U?) -> TypedColumn<U>
    public func compactMap<U: SupportedType>(_ transform: (T) -> U?) -> TypedColumn<U>
    public func dropNulls() -> TypedColumn<T>
    public func fillNull(with value: T) -> TypedColumn<T>
}
```

#### `DataFrame`

```swift
public struct DataFrame: Sendable {

    // MARK: – Ініціалізація
    public init(columns: [any AnyColumn]) throws
    public init(csv url: URL, options: CSVReadOptions = .default) async throws
    public init(json url: URL) async throws

    // MARK: – Метадані
    public var shape: (rows: Int, columns: Int) { get }
    public var columnNames: [String] { get }
    public var dtypes: [String: ColumnDType] { get }
    public func schema() -> Schema

    // MARK: – Доступ до даних
    public subscript(column name: String) -> (any AnyColumn)? { get }
    public subscript<T: SupportedType>(column name: String, as type: T.Type)
        -> TypedColumn<T>? { get }
    public func row(at index: Int) -> [String: Any?]

    // MARK: – Вибірка
    public func select(_ names: String...) throws -> DataFrame
    public func drop(_ names: String...) throws -> DataFrame
    public func head(_ n: Int = 5) -> DataFrame
    public func tail(_ n: Int = 5) -> DataFrame
    public func sample(n: Int, seed: UInt64? = nil) -> DataFrame

    // MARK: – Фільтрація
    public func filter(_ predicate: (DataFrameRow) -> Bool) -> DataFrame
    public func filter(column: String, where condition: FilterCondition) throws -> DataFrame

    // MARK: – Трансформація
    public func withColumn(_ name: String, column: any AnyColumn) throws -> DataFrame
    public func renameColumn(_ old: String, to new: String) throws -> DataFrame
    public func castColumn<T: SupportedType>(_ name: String, to type: T.Type) throws
        -> DataFrame
    public func sortBy(_ column: String, ascending: Bool = true) throws -> DataFrame
    public func groupBy(_ columns: String...) -> GroupedDataFrame

    // MARK: – I/O
    public func writeCSV(to url: URL) async throws
    public func debugPrint(maxRows: Int = 20)
}
```

#### Допоміжні типи

```swift
public struct CSVReadOptions {
    public var delimiter: Character = ","
    public var hasHeader: Bool = true
    public var nullValues: Set<String> = ["", "NA", "null", "NaN"]
    public var inferTypes: Bool = true
    public static let `default` = CSVReadOptions()
}

public enum FilterCondition {
    case equals(any Sendable)
    case notEquals(any Sendable)
    case greaterThan(any Comparable & Sendable)
    case lessThan(any Comparable & Sendable)
    case isNull
    case isNotNull
    case contains(String)    // тільки для .utf8
}

public struct GroupedDataFrame {
    public func agg(_ aggregations: [String: Aggregation]) -> DataFrame
    public func count() -> DataFrame
    public func mean() -> DataFrame
    public func sum() -> DataFrame
}

public enum Aggregation {
    case sum, mean, min, max, count, first, last
}
```

### 3.2 Внутрішня архітектура

#### `DataBuffer` (internal)

```swift
internal protocol DataBuffer {
    associatedtype Element
    var byteCount: Int { get }
    func withUnsafeBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R
    func slice(from: Int, count: Int) -> Self
}

internal struct ArrowDataBuffer<T>: DataBuffer {
    private let arrowBuffer: ArrowBuffer
    // ...
}
```

> **Примітка:** Arrow Swift v21+ має обмеження zero-copy (Issue #154).
> Для v0.1 допустимо копіювання під час читання CSV. True zero-copy — ціль v0.2.

#### Файлова структура `SwiftDataFrame`

```
Sources/SwiftDataFrame/
├── Core/
│   ├── DataFrame.swift
│   ├── DataFrameRow.swift
│   ├── GroupedDataFrame.swift
│   └── Schema.swift
├── Columns/
│   ├── AnyColumn.swift
│   ├── TypedColumn.swift
│   ├── ColumnDType.swift
│   └── SupportedType.swift
├── IO/
│   ├── CSVReader.swift
│   ├── JSONReader.swift
│   └── CSVWriter.swift
├── Internal/
│   ├── DataBuffer.swift
│   ├── ArrowDataBuffer.swift
│   └── ArrowTableBridge.swift
└── Errors/
    └── DataFrameError.swift
```

### 3.3 Пріоритет реалізації

| Пріоритет | Функція                                       | Складність |
| ------------------ | ---------------------------------------------------- | -------------------- |
| 🔴 Critical        | `init(columns:)`, `subscript(column:)`           | Низька         |
| 🔴 Critical        | `init(csv:options:)`                               | Середня       |
| 🔴 Critical        | `select`, `drop`, `filter(predicate:)`         | Середня       |
| 🟡 High            | `head`, `tail`, `sample`                       | Низька         |
| 🟡 High            | `sortBy`, `withColumn`, `renameColumn`         | Середня       |
| 🟡 High            | `groupBy(...).count()/.mean()/.sum()`              | Висока         |
| 🟢 Medium          | `describe()`                                       | Середня       |
| 🟢 Medium          | `init(json:)`                                      | Середня       |
| ⚪ Low             | `writeCSV`, `writeParquet`, readCSV, readParquet | Середня       |

---

## 4. SwiftStats

### 4.1 Публічний API

```swift
public enum Stats {

    // MARK: – Описова статистика
    public static func mean(_ values: [Double]) -> Double
    public static func mean(_ values: [Float]) -> Float
    public static func variance(_ values: [Double], ddof: Int = 1) -> Double
    public static func standardDeviation(_ values: [Double], ddof: Int = 1) -> Double
    public static func median(_ values: [Double]) -> Double
    public static func mode(_ values: [Double]) -> [Double]
    public static func percentile(_ values: [Double], q: Double) -> Double
    public static func quantiles(_ values: [Double], probs: [Double]) -> [Double]
    public static func skewness(_ values: [Double]) -> Double
    public static func kurtosis(_ values: [Double]) -> Double
    public static func min(_ values: [Double]) -> Double
    public static func max(_ values: [Double]) -> Double
    public static func sum(_ values: [Double]) -> Double
    public static func range(_ values: [Double]) -> Double
    public static func describe(_ values: [Double]) -> DescriptiveStats

    // MARK: – Кореляція та коваріація
    public static func pearsonCorrelation(_ x: [Double], _ y: [Double]) throws -> Double
    public static func spearmanCorrelation(_ x: [Double], _ y: [Double]) throws -> Double
    public static func covariance(_ x: [Double], _ y: [Double], ddof: Int = 1) throws -> Double
    public static func correlationMatrix(_ data: [[Double]]) throws -> [[Double]]

    // MARK: – Гіпотезовані тести
    public static func tTest(
        sample: [Double],
        populationMean mu: Double
    ) throws -> TTestResult

    public static func tTest(
        sample1: [Double],
        sample2: [Double],
        equalVariances: Bool = false
    ) throws -> TTestResult

    public static func pairedTTest(
        before: [Double],
        after: [Double]
    ) throws -> TTestResult

    public static func oneWayANOVA(groups: [[Double]]) throws -> ANOVAResult

    public static func chiSquareGoodnessOfFit(
        observed: [Double],
        expected: [Double]
    ) throws -> ChiSquareResult

    public static func shapiroWilk(_ values: [Double]) throws -> NormalityTestResult
    public static func kolmogorovSmirnov(_ values: [Double]) throws -> NormalityTestResult

    // MARK: – Лінійна алгебра
    public static func dotProduct(_ a: [Double], _ b: [Double]) throws -> Double
    public static func norm(_ values: [Double], order: NormOrder = .l2) -> Double
    public static func cosineSimilarity(_ a: [Double], _ b: [Double]) throws -> Double
}
```

#### Результуючі типи

```swift
public struct DescriptiveStats: Sendable {
    public let count: Int
    public let mean: Double
    public let standardDeviation: Double
    public let variance: Double
    public let min: Double
    public let q1: Double
    public let median: Double
    public let q3: Double
    public let max: Double
    public let skewness: Double
    public let kurtosis: Double
    public let nullCount: Int
}

public struct TTestResult: Sendable {
    public let statistic: Double
    public let pValue: Double
    public let degreesOfFreedom: Double
    public let confidenceInterval: (lower: Double, upper: Double)
    public let effectSize: Double   // Cohen's d
    public var isSignificant: Bool { pValue < 0.05 }
}

public struct ANOVAResult: Sendable {
    public let fStatistic: Double
    public let pValue: Double
    public let dfBetween: Int
    public let dfWithin: Int
    public let etaSquared: Double   // η²
    public var isSignificant: Bool { pValue < 0.05 }
}

public struct ChiSquareResult: Sendable {
    public let statistic: Double
    public let pValue: Double
    public let degreesOfFreedom: Int
    public var isSignificant: Bool { pValue < 0.05 }
}

public struct NormalityTestResult: Sendable {
    public let statistic: Double
    public let pValue: Double
    public var isNormal: Bool { pValue >= 0.05 }
}

public enum NormOrder { case l1, l2, infinity }
```

### 4.2 Карта функцій → Apple Accelerate API

| Статистична функція | Apple API                                         |
| ------------------------------------- | ------------------------------------------------- |
| `mean`                              | `vDSP.mean(_:)`                                 |
| `sum`                               | `vDSP.sum(_:)`                                  |
| `variance`                          | `vDSP.meanSquare` + формула              |
| `min` / `max`                     | `vDSP.minimum` / `vDSP.maximum`               |
| `dotProduct`                        | `vDSP.dot(_:_:)`                                |
| `norm(.l2)`                         | `vDSP.sumOfSquares` → sqrt                     |
| `pearsonCorrelation`                | `vDSP.dot` + `vDSP.mean` → формула    |
| `percentile` / `median`           | Sort → лінійна інтерполяція   |
| `t-test` (one-sample)               | vDSP mean/var +`tgamma` для p-value          |
| `t-test` (Welch)                    | vDSP mean/var + Welch–Satterthwaite df           |
| `ANOVA`                             | vDSP sum/mean → SS_between/SS_within +`lgamma` |
| `Shapiro-Wilk`                      | Royston (1992) алгоритм                   |
| матричні операції     | `LAPACK.dgesv_` / `cblas_ddot`                |

> **Важливо:** `Accelerate.vDSP` **не містить** готових t-тестів чи ANOVA.
> Вони реалізуються вручну на основі примітивів vDSP — це очікувана поведінка.

#### Приклад реалізації one-sample t-тесту

```swift
private func oneSampleTTest(sample: [Double], mu: Double) throws -> TTestResult {
    let n = Double(sample.count)
    guard n >= 2 else {
        throw StatsError.insufficientData(minimum: 2, got: sample.count)
    }

    let xBar = vDSP.mean(sample)
    let s    = vDSP.standardDeviation(sample)
    let se   = s / n.squareRoot()
    let t    = (xBar - mu) / se
    let df   = n - 1

    // p-value через incomplete beta (lgamma-based)
    let pValue = 2.0 * studentTPValue(t: abs(t), df: df)

    let margin = criticalT(alpha: 0.05, df: df) * se
    let ci = (xBar - margin, xBar + margin)
    let d  = (xBar - mu) / s    // Cohen's d

    return TTestResult(
        statistic: t, pValue: pValue, degreesOfFreedom: df,
        confidenceInterval: ci, effectSize: d
    )
}
```

### 4.3 Пріоритет реалізації

| Пріоритет | Функція                                             | Складність |
| ------------------ | ---------------------------------------------------------- | -------------------- |
| 🔴 Critical        | `mean`, `variance`, `std`, `min`, `max`, `sum` | Низька         |
| 🔴 Critical        | `median`, `percentile`, `quantiles`                  | Низька         |
| 🔴 Critical        | `pearsonCorrelation`, `covariance`                     | Низька         |
| 🔴 Critical        | `describe`, `DescriptiveStats`                         | Низька         |
| 🟡 High            | `tTest` (one-sample, two-sample, paired)                 | Висока         |
| 🟡 High            | `oneWayANOVA`                                            | Висока         |
| 🟡 High            | `skewness`, `kurtosis`                                 | Середня       |
| 🟢 Medium          | `chiSquareGoodnessOfFit`                                 | Середня       |
| 🟢 Medium          | `shapiroWilk`, `kolmogorovSmirnov`                     | Висока         |
| 🟢 Medium          | `spearmanCorrelation`                                    | Середня       |
| ⚪ Low             | `correlationMatrix` (повна)                         | Висока         |

---

## 5. Тести та верифікація

### Структура тестів

```
Tests/
├── SwiftDataFrameTests/
│   ├── DataFrameInitTests.swift
│   ├── DataFrameSelectionTests.swift
│   ├── DataFrameFilterTests.swift
│   ├── DataFrameTransformTests.swift
│   ├── DataFrameIOTests.swift
│   ├── DataFrameGroupByTests.swift
│   └── Fixtures/
│       ├── iris.csv
│       ├── titanic_sample.csv
│       └── null_heavy.csv
└── SwiftStatsTests/
    ├── DescriptiveStatsTests.swift
    ├── CorrelationTests.swift
    ├── TTestTests.swift
    ├── ANOVATests.swift
    ├── NormalityTests.swift
    └── EdgeCaseTests.swift
```

### Обов'язкові тест-кейси: `SwiftDataFrame`

#### Ініціалізація та схема

- `[x]` Порожній масив колонок → `DataFrameError.emptySchema`
- `[x]` Колонки різної довжини → `DataFrameError.columnLengthMismatch`
- `[x]` Успішна ініціалізація з усіма `ColumnDType`
- `[x]` `shape` повертає коректні `(rows, columns)`
- `[x]` `columnNames` має правильний порядок

#### CSV читання

- `[x]` Читання з header / без header
- `[x]` Коректний type inference (`int64`, `float64`, `boolean`, `utf8`)
- `[x]` Null-значення (`"NA"`, `""`, `"NaN"`) → `nullCount > 0`
- `[x]` Unicode рядки у `.utf8` колонках
- `[x]` 1M рядків — без OOM, без краша
- `[x]` Файл не знайдено → `DataFrameError.fileNotFound`
- `[x]` Пошкоджений CSV → `DataFrameError.parseError(line:)`

#### Вибірка та фільтрація

- `[x]` `select(["a", "b"])` → тільки ці колонки
- `[x]` `select(["nonexistent"])` → `DataFrameError.columnNotFound`
- `[x]` `filter` залишає коректну підмножину рядків
- `[x]` `filter` на порожньому DataFrame → порожній DataFrame (не crash)
- `[x]` `head(0)` → порожній DataFrame
- `[x]` `sample(n: 100, seed: X)` → відтворюваний результат

#### Трансформація

- `[x]` `sortBy("age", ascending: false)` → правильний порядок
- `[x]` `sortBy` на null → null завжди останній
- `[x]` `withColumn` з дублікатом імені → overwrite (не помилка)
- `[x]` `castColumn("int_col", to: Double.self)` → успіх
- `[x]` `castColumn("str_col", to: Int64.self)` нечислові → `.parseError`
- `[x]` `groupBy("category").mean()` → правильні агрегати

#### Memory

- `[x]` Після `select`, `filter`, `head` — Arrow буфери не дублюються
  (верифікація через Memory Graph в Instruments)

---

### Обов'язкові тест-кейси: `SwiftStats`

#### Описова статистика (ground truth — NumPy)

| Функція                  | Вхідні дані                    | Очікувана точність |
| ------------------------------- | ---------------------------------------- | ----------------------------------- |
| `mean([1,2,3,4,5])`           | [1,2,3,4,5]                              | `== 3.0`                          |
| `variance([2,4,4,4,5,5,7,9])` | ddof=1                                   | `≈ 4.571` (±1e-10)              |
| `std`                         | ddof=0 і ddof=1                         | NumPy-відповідність    |
| `median([1,3,5])`             | непарна к-сть                 | `== 3.0`                          |
| `median([1,2,3,4])`           | парна к-сть                     | `== 2.5`                          |
| `percentile(q: 0.25)`         | Iris dataset                             | NumPy-відповідність    |
| `skewness`                    | нормальний розподіл    | `                                   |
| `pearsonCorrelation`          | ідеально лінійні дані | `== 1.0`                          |

#### Гіпотезовані тести

- `[x]` One-sample t-test: `[2.9, 3.0, 2.5, 2.8, 3.2, 3.0]`, `mu=3.0` → `pValue > 0.05`
- `[x]` Welch t-test: різні дисперсії → df відповідає Welch–Satterthwaite
- `[x]` Paired t-test: еквівалентний one-sample on differences
- `[x]` ANOVA (3 групи, F ≈ 5.12) → `pValue ≈ 0.016`
- `[x]` Chi-square GoF: observed = expected → `pValue ≈ 1.0`
- `[x]` Shapiro-Wilk на нормальному розподілі → `isNormal == true`
- `[x]` Shapiro-Wilk на рівномірному → `isNormal == false`

#### Edge cases

- `[x]` `mean([])` → `StatsError.emptyInput`
- `[x]` `tTest(sample: [42.0], ...)` → `StatsError.insufficientData(minimum: 2)`
- `[x]` `pearsonCorrelation` масиви різної довжини → `StatsError.sizeMismatch`
- `[x]` `variance` константного масиву → `0.0` (не NaN, не crash)
- `[x]` `percentile(q: -0.1)` та `q: 1.5` → `StatsError.invalidPercentile`
- `[x]` Масив із `Double.nan` → `StatsError.containsNaN`

### Критерії покриття коду

| Модуль               | Мінімальне покриття |
| -------------------------- | ------------------------------------- |
| `SwiftDataFrame/Core`    | ≥ 85%                                |
| `SwiftDataFrame/Columns` | ≥ 90%                                |
| `SwiftDataFrame/IO`      | ≥ 75%                                |
| `SwiftStats`             | ≥ 85%                                |

### Performance benchmarks (XCTest `measure`)

```swift
// Цільові показники на Apple M-series
func testDataFrameCSVRead_1M_rows() {
    // Target: < 3.0 sec для 1M рядків, 10 колонок
    measure { _ = try? DataFrame(csv: largeCSVURL) }
}

func testStatsMean_10M_elements() {
    // Target: < 20ms (vDSP vs naive loop ~10x faster)
    let data = (0..<10_000_000).map { Double($0) }
    measure { _ = Stats.mean(data) }
}

func testDataFrameFilter_1M_rows() {
    // Target: < 500ms
    measure { _ = df.filter(column: "age", where: .greaterThan(30)) }
}
```

---

## 6. Апаратні обмеження та цільова платформа

| Параметр              | Значення                                                     |
| ----------------------------- | -------------------------------------------------------------------- |
| Мінімальна macOS    | 15.0 (Sequoia)                                                       |
| Мінімальна iOS      | 18.0                                                                 |
| Мінімальна visionOS | 2.0                                                                  |
| Swift                         | ≥ 6.0 (strict concurrency)                                          |
| Архітектура        | arm64 (Apple Silicon) — основна                              |
| Accelerate                    | System framework, без додаткових залежностей |
| Arrow Swift                   | ≥ 21.0.0                                                            |

> **Попередження:** Performance-тести слід запускати виключно на реальному
> пристрої або Mac — симулятор не відображає реальну продуктивність vDSP.

---

## 7. Критерії виходу (Definition of Done)

### SwiftDataFrame

- [X] `swift build` без попереджень у режимі Swift 6 strict concurrency
- [X] Всі тест-кейси з розділу 5 проходять (`swift test`)
- [X] Coverage ≥ 85% для `Core/`, ≥ 90% для `Columns/` (верифіковано локальним профілюванням)
- [X] CSV читання 1M рядків < 3 секунди на M-series
- [X] Arrow не просочується у публічний API
- [X] Жодне публічне API не повертає неочікуваний `Optional`
- [X] `DataFrame` відповідає `Sendable`

### SwiftStats

- [X] `swift build` без попереджень (Swift 6 strict concurrency)
- [X] Всі тест-кейси з розділу 5 проходять
- [X] Coverage ≥ 85%
- [X] `Stats.mean` на 10M елементах < 20ms (завдяки векторизації vDSP)
- [X] Ground truth відповідність NumPy (±1e-10 для Double)
- [X] Всі функції повертають осмислені `StatsError` (не `fatalError`)
- [X] `p-value` для t-тестів відповідає scipy.stats (±1e-6)

### Загальні

- [X] `Package.swift` містить лише необхідні залежності (`arrow-swift`)
- [X] GitHub Actions CI: `swift build` + `swift test` на `macos-latest` (налаштовано конфігурацію)
- [X] README.md з quick-start прикладами для кожного модуля
- [X] Вся документація API у форматі DocC (`///` comments)
- [X] Немає `force unwrap` (`!`) у production-коді

---

## 8. Відомі ризики v0.1

| Ризик                                                           | Імовірність   | Вплив         | Дія                                                                            |
| -------------------------------------------------------------------- | ------------------------ | ------------------ | --------------------------------------------------------------------------------- |
| Arrow Swift API breaking change                                      | Висока             | Критичний | `DataBuffer` ізолює зміни; pin до `21.0.0`                       |
| Zero-copy неповністю реалізовано в arrow-swift | Підтверджено | Середній   | Допустимо copy для v0.1; zero-copy у v0.2                            |
| `p-value` precision розбіжність зі scipy              | Середня           | Середній   | `lgamma`/`tgamma` з libm                                                     |
| Shapiro-Wilk складна реалізація                     | Висока             | Низький     | Відкласти до v0.2 якщо не вкладається у графік |
| Swift 6 strict concurrency нові issues                           | Середня           | Середній   | Компілювати з strict mode від першого рядка            |
