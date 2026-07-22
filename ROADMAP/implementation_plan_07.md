# План реалізації — Версія 0.7: Тестування та Бенчмарки (Testing & Benchmarks)

**Версія:** 0.7  
**Цільові модулі:** Усі модулі екосистеми (`SwiftDataFrame`, `SwiftStats`, `SwiftPreprocessing`, `SwiftML`, `SwiftCluster`, `SwiftNLP`, `SwiftOptimize`, `SwiftForecast`)  
**Мета:** Створення інфраструктури для автоматичного тестування продуктивності (benchmarking) порівняно з Python-стеком (NumPy, pandas, scikit-learn, statsmodels), верифікація та оптимізація споживання пам'яті (зокрема zero-copy гарантій Apache Arrow), а також налаштування надійної CI/CD системи на базі GitHub Actions.

---

## Зміст

1. [Архітектурне дослідження та цілі](#1-архітектурне-дослідження-та-цілі)
2. [Бенчмарки продуктивності (Swift vs Python)](#2-бенчмарки-продуктивності-swift-vs-python)
   - 2.1 [Структура Benchmarks директорії](#21-структура-benchmarks-директорії)
   - 2.2 [Додавання таргету в Package.swift](#22-додавання-таргету-в-packageswift)
   - 2.3 [Реалізація Swift-бенчмарків](#23-реалізація-swift-бенчмарків)
   - 2.4 [Реалізація Python-бенчмарків](#24-реалізація-python-бенчмарків)
3. [Профілювання пам'яті та верифікація Zero-Copy](#3-профілювання-памяті-та-верифікація-zero-copy)
   - 3.1 [Поточний стан інтеграції Arrow (Копіювання vs Zero-Copy)](#31-поточний-стан-інтеграції-arrow-копіювання-vs-zero-copy)
   - 3.2 [Пропозиція рефакторингу під справжній Zero-Copy](#32-пропозиція-рефакторингу-під-справжній-zero-copy)
   - 3.3 [Профілювання через Xcode Instruments](#33-профілювання-через-xcode-instruments)
4. [CI/CD Конфігурація (GitHub Actions)](#4-cicd-конфігурація-github-actions)
5. [Відкриті питання для обговорення](#5-відкриті-питання-для-обговорення)

---

## 1. Архітектурне дослідження та цілі

Версія 0.7 фокусується на забезпеченні стабільності, швидкості та ефективності використання пам'яті. Ми хочемо довести, що нативна Swift-реалізація на Apple Silicon перевершує традиційний Python-стек завдяки компіляції в машинний код, паралелізму Swift 6 та апаратній оптимізації (Accelerate / GPU).

### Основні виклики:
1. **Чесне порівняння (Apples-to-Apples):** Ми повинні тестувати алгоритми на однакових даних, однакових розмірах матриць та ідентичних конфігураціях (наприклад, однакова глибина дерев, однакова кількість ітерацій K-Means).
2. **Апаратна специфіка в CI:** Оскільки модулі `SwiftStats`, `SwiftCluster` та `SwiftForecast` критично залежать від фреймворку `Accelerate` (вбудованого в macOS), наша CI-система повинна працювати виключно на macOS-раннерах. Безкоштовні GitHub Actions раннери на Linux не підтримують macOS SDK, тому CI має запускатися на `macos-14` або `macos-latest` (Apple Silicon M1/M2).
3. **Флуктуація часу виконання в CI:** Віртуальні машини в хмарі мають високу варіативність швидкості CPU ("noisy neighbors"). Ми маємо розробити критерії оцінки деградації продуктивності, які не будуть давати хибних збоїв CI.

---

## 2. Бенчмарки продуктивності (Swift vs Python)

Ми створимо єдину систему запуску бенчмарків, яка порівняє час виконання ключових операцій у Swift та Python.

### 2.1 Структура Benchmarks директорії

Бенчмарки будуть розташовані на одному рівні з `Sources` та `Tests` у репозиторії `SwiftAnalytics`:

```
SwiftAnalytics/
├── Package.swift
├── Sources/
├── Tests/
├── Benchmarks/
│   ├── Swift/
│   │   ├── main.swift                     # Точка входу, парсинг аргументів
│   │   ├── DataFrameBenchmarks.swift      # Завантаження CSV, фільтрація, групування
│   │   ├── MLBenchmarks.swift             # LinearReg, RandomForest, GBDT
│   │   ├── ForecastBenchmarks.swift       # ARIMA, Holt-Winters, Kalman
│   │   └── BenchmarkSuite.swift           # Допоміжні протоколи, таймінги (ContinuousClock)
│   ├── Python/
│   │   ├── requirements.txt               # pandas, numpy, scikit-learn, statsmodels, mlx
│   │   └── benchmarks.py                  # Python-еквіваленти тестів продуктивності
│   └── Results/                           # Збереження результатів у форматі JSON/CSV
│       ├── swift_results.json
│       └── python_results.json
```

### 2.2 Додавання таргету в Package.swift

Для запуску бенчмарків ми додамо новий виконуваний таргет (executable target) в `Package.swift`:

```swift
// Package.swift
// Додати в targets:
.executableTarget(
    name: "SwiftAnalyticsBenchmarks",
    dependencies: [
        "SwiftDataFrame",
        "SwiftStats",
        "SwiftPreprocessing",
        "SwiftML",
        "SwiftCluster",
        "SwiftNLP",
        "SwiftOptimize",
        "SwiftForecast"
    ],
    path: "Benchmarks/Swift",
    swiftSettings: globalSwiftSettings
)
```

Це дозволить запускати Swift-бенчмарки однією командою:
```bash
swift run -c release SwiftAnalyticsBenchmarks
```
> [!IMPORTANT]
> Запуск обов'язково має відбуватися з прапорцем `-c release` (або `--configuration release`), інакше компілятор Swift не оптимізує код (наприклад, операції `vDSP` та `MLX`), що призведе до некоректних результатів порівняння з оптимізованими C/C++ бібліотеками Python.

### 2.3 Реалізація Swift-бенчмарків

Ми реалізуємо легковагу систему вимірювання за допомогою `ContinuousClock` без використання важких зовнішніх залежностей.

```swift
// Benchmarks/Swift/BenchmarkSuite.swift
import Foundation

protocol Benchmark {
    var name: String { get }
    func run() async throws -> Double // повертає середній час виконання в мілісекундах
}

struct BenchmarkRunner {
    static func measure(warmup: Int = 2, iterations: Int = 10, block: () async throws -> Void) async rethrows -> Double {
        // Теплий старт (warmup) для прогріву CPU/GPU та ініціалізації кешів
        for _ in 0..<warmup {
            try await block()
        }
        
        let clock = ContinuousClock()
        var durations: [Duration] = []
        
        for _ in 0..<iterations {
            let duration = try await clock.measure {
                try await block()
            }
            durations.append(duration)
        }
        
        // Середнє значення
        let totalSeconds = durations.reduce(0.0) { $0 + Double($1.components.attoseconds) / 1e18 + Double($1.components.seconds) }
        return (totalSeconds / Double(iterations)) * 1000.0 // в мілісекунди
    }
}
```

#### Сценарії для бенчмарків:
1. **`SwiftDataFrame` vs Pandas:**
   - Читання CSV-файлу (100,000 рядків, 5 колонок різних типів).
   - Фільтрація рядків за логічною умовою.
   - Агрегація `groupBy` (групування за текстовою колонкою, підрахунок середнього та суми для числових колонок).
2. **`SwiftStats` vs NumPy:**
   - Векторні операції: середнє, стандартне відхилення, дисперсія на масиві з 1,000,000 елементів типу `Double` (порівняння `vDSP` та NumPy).
3. **`SwiftML` vs Scikit-Learn / MLX:**
   - Навчання лінійної регресії (10,000 рядків, 10 ознак, 100 епох).
   - Навчання Random Forest Classifier (1,000 рядків, 4 ознаки, 50 дерев, максимальна глибина 4).
   - Навчання Gradient Boosted Trees (1,000 рядків, 4 ознаки, 50 ітерацій).
4. **`SwiftCluster` vs Scikit-Learn:**
   - K-Means (10,000 точок, 3 кластери, 20 ітерацій, порівняння GPU-прискорення через MLX у Swift та CPU Scikit-Learn).
   - PCA (SVD розклад матриці 1000x100 через Accelerate LAPACK).
5. **`SwiftForecast` vs Statsmodels:**
   - Holt-Winters експоненційне згладжування (1,000 точок, сезонність = 12).
   - ARIMA(1,1,1) оцінка параметрів OLS (500 точок).
   - Kalman Filter (10,000 ітерацій фільтрації 2D стану).

### 2.4 Python-бенчмарки

Python-скрипт `benchmarks.py` буде генерувати ідентичні датасети за допомогою фіксованого seed (`np.random.seed(42)`) та виконувати аналогічні операції. Результати вимірювань зберігатимуться у `Benchmarks/Results/python_results.json` для подальшого аналізу.

Ми порівняємо:
- NumPy (векторизовані операції)
- Pandas (маніпуляції з даними)
- Scikit-Learn (ML, кластеризація, препроцесинг)
- Statsmodels (аналіз часових рядів)

---

## 3. Профілювання пам'яті та верифікація Zero-Copy

### 3.1 Поточний стан інтеграції Arrow (Копіювання vs Zero-Copy)

Під час дослідження `ArrowTableBridge.swift` було виявлено, що поточні методи конвертації **не є zero-copy**:
- Метод `toDataFrame(_:)` ітерується по елементах `ChunkedArray` і копіює їх в новий масив Swift `[T?]` через цикл `for i in 0..<UInt(count)`.
- Метод `toArrowTable(_:)` використовує `ArrowArrayBuilders` та копіює дані з `TypedColumn` поелементно через `appendAny`.

Це створює дублювання пам'яті в купі (heap) та знижує швидкість обробки великих обсягів даних. 

### 3.2 Пропозиція рефакторингу під справжній Zero-Copy

Для реалізації справжнього zero-copy доступу до пам'яті Apache Arrow, нам необхідно змінити модель збереження даних в `TypedColumn`:
1. Наразі `TypedColumn` жорстко тримає масив `values: [T?]`.
2. Потрібно інтегрувати протокол `DataBuffer` безпосередньо в структуру колонок:
   - Впровадити збереження через абстракцію `DataBuffer` (яка може бути як `ArrayDataBuffer`, так і `ArrowDataBuffer` з прямим вказівником `UnsafeRawPointer` на Arrow-пам'ять).
   - Додати підтримку масок null-значень (Arrow validity bitmaps) без розгортання всього масиву опціоналів `[T?]`.

#### План верифікації zero-copy:
Ми додамо спеціальний тест `testArrowZeroCopyInvariant()` у `DataFrameArrowTests.swift`:
```swift
@Test("Arrow to DataFrame conversion maintains raw memory pointer")
func testArrowZeroCopyInvariant() throws {
    // 1. Створити ArrowTable з великим об'ємом даних
    // 2. Конвертувати в DataFrame
    // 3. Перевірити, що вказівник UnsafeRawPointer в ArrowDataBuffer збігається з оригінальним буфером пам'яті Arrow
}
```

### 3.3 Профілювання через Xcode Instruments

Для ручної перевірки пікового споживання пам'яті та виявлення витоків пам'яті (memory leaks) ми опишемо детальну інструкцію з профілювання за допомогою Xcode Instruments:
1. Запуск схеми з профілюванням: `Product -> Profile (⌘I)` в Xcode.
2. Вибір шаблону **Allocations** для моніторингу загального об'єму виділеної пам'яті та верифікації відсутності сплесків (memory spikes) під час навчання моделей.
3. Вибір шаблону **Leaks** для підтвердження відсутності retain-циклів у асинхронних акторах (наприклад, у `ExponentialSmoothing` чи `ARIMAModel`).

---

## 4. CI/CD Конфігурація (GitHub Actions)

Ми налаштуємо робочий процес GitHub Actions для автоматичного збирання, тестування та перевірки якості коду на кожен Pull Request.

### Файл конфігурації: `.github/workflows/ci.yml`

```yaml
name: SwiftAnalytics CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  build-and-test:
    name: Build and Test (macOS)
    runs-on: macos-14 # Apple Silicon M1 раннер для Accelerate та Metal GPU
    
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Set up Swift
        uses: swift-actions/setup-swift@v2
        with:
          swift-version: '6.0'

      - name: Build Package
        run: swift build -v
        working-directory: SwiftAnalytics

      - name: Run Tests with Code Coverage
        run: swift test --enable-code-coverage
        working-directory: SwiftAnalytics

      - name: Process Code Coverage
        run: |
          # Отримання шляху до бінарника тестів
          XCTEST_PATH=$(swift build --show-bin-path -c debug)/SwiftAnalyticsPackageTests.xctest/Contents/MacOS/SwiftAnalyticsPackageTests
          if [ ! -f "$XCTEST_PATH" ]; then
            # Альтернативний шлях для macOS CLI
            XCTEST_PATH=$(swift build --show-bin-path -c debug)/SwiftDataFrameTests.xctest/Contents/MacOS/SwiftDataFrameTests
          fi
          
          # Експорт покриття в lcov формат за допомогою llvm-cov
          xcrun llvm-cov export \
            -format="lcov" \
            -instr-profile=.build/debug/codecov/default.profdata \
            "$XCTEST_PATH" > coverage.lcov
        working-directory: SwiftAnalytics
        continue-on-error: true

      - name: Upload Coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          file: SwiftAnalytics/coverage.lcov
          token: ${{ secrets.CODECOV_TOKEN }}
        continue-on-error: true

  performance-check:
    name: Performance Regression Check
    runs-on: macos-14
    needs: build-and-test
    
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Set up Swift
        uses: swift-actions/setup-swift@v2
        with:
          swift-version: '6.0'

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Python Dependencies
        run: |
          pip install -r SwiftAnalytics/Benchmarks/Python/requirements.txt

      - name: Run Swift Benchmarks (Release)
        run: |
          swift run -c release SwiftAnalyticsBenchmarks --export-json Benchmarks/Results/swift_results.json
        working-directory: SwiftAnalytics

      - name: Run Python Benchmarks
        run: |
          python3 Benchmarks/Python/benchmarks.py --export-json Benchmarks/Results/python_results.json
        working-directory: SwiftAnalytics

      - name: Compare Results & Check Regressions
        run: |
          # Скрипт порівняння, який аналізує деградацію відносно попереднього коміту або порівнює з Python
          python3 Benchmarks/Python/compare.py Benchmarks/Results/swift_results.json Benchmarks/Results/python_results.json
        working-directory: SwiftAnalytics
```

> [!WARNING]
> Оскільки хмарні macOS-раннери на GitHub Actions можуть показувати нестабільні результати за часом CPU (флуктуації до 20-30%), ми пропонуємо:
> 1. Не провалювати білд автоматично при незначних коливаннях швидкості.
> 2. Скрипт `compare.py` буде генерувати коментар до PR з таблицею порівняння швидкості, маркуючи червоним лише явні регресії (> 50%).

---

## 5. Відкриті питання для обговорення

Для повної деталізації плану, нам потрібно узгодити кілька архітектурних та операційних моментів:

1. **Глибина інтеграції Zero-Copy:**
   Чи маємо ми у межах цієї версії (0.7) повністю переписати внутрішній механізм `TypedColumn` на conform до `DataBuffer` з підтримкою null-bitmap від Arrow, чи достатньо лише додати детальні тести, що перевіряють відсутність копіювання при передачі сирих байтів через `UnsafeRawPointer`?
   *(Рекомендовано: Почати з рефакторингу `TypedColumn`, оскільки без цього справжній zero-copy неможливий для користувача бібліотеки).*

2. **Залежність Python-бенчмарків:**
   Які саме версії Python-бібліотек ми фіксуємо як базові? Чи достатньо використовувати стандартні версії з conda-forge/pip для архітектури `arm64` (Apple Silicon)?
   *(Рекомендовано: `pandas>=2.1`, `numpy>=1.26`, `scikit-learn>=1.3`, `statsmodels>=0.14`).*

3. **Локальні baseline-файли:**
   Чи повинні ми зберігати еталонні результати бенчмарків (baseline) прямо в Git (наприклад, у `Benchmarks/Results/baseline.json`) для локального порівняння під час розробки?
   *(Рекомендовано: Так, створити baseline для стандартного Apple M1/M3 процесора, щоб розробник міг локально виконати `swift run -c release SwiftAnalyticsBenchmarks --compare` і відразу побачити прогрес/регрес після своїх змін).*
