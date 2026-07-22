# ROADMAP: SwiftSci Ecosystem

**Візія:** Створення нативної, модульної екосистеми Data Science для Apple Silicon, що замінює традиційний Python-стек, оптимізованої для суворої багатопотоковості Swift 6 та Unified Memory Architecture (UMA).

---

## 📊 Поточний Статус

| Модуль                   | Версія | Статус      |
| ------------------------------ | ------------ | ----------------- |
| Package Bootstrap              | 0.0          | 🟢 Completed      |
| `SwiftDataFrame`             | 0.1          | 🟢 Completed      |
| `SwiftStats`                 | 0.1          | 🟢 Completed      |
| `SwiftPreprocessing`         | 0.2          | 🟢 Completed      |
| `SwiftML`                    | 0.2          | 🟢 Completed      |
| `SwiftCluster`               | 0.3          | 🟢 Completed      |
| `SwiftML` Concurrency        | 0.4          | 🟢 Completed      |
| `SwiftOptimize`              | 0.5          | 🟢 Completed      |
| `SwiftForecast`              | 0.6          | 🟢 Completed      |
| Testing & Benchmarks           | 0.7          | 🟢 Completed      |
| Hardware Routing               | 0.8          | 🟢 Completed      |
| API Stabilization & Docs       | 0.9          | 🟢 Completed      |
| `SwiftExplain`               | 1.0          | 🟢 Completed      |
| `SwiftLLM` / `SwiftVision` | 1.0          | 🟢 Completed      |
| `SwiftPrivacy`               | 1.0          | 🟠 Removed in 1.1 |
| Streaming I/O & SafeTensors    | 1.1          | 🟢 Completed      |
| SwiftSci Rename & Refactor     | 1.2          | 🟢 Completed (plan 12)|
| Sklearn Parity Roadmap         | 1.3          | 🟢 Completed (plan 13)|

---

## 🗂 Структура Бібліотеки (Swift Package)

Екосистема розробляється як єдиний Swift Package, розділений на незалежні таргети (modules) для забезпечення чіткої інкапсуляції та управління залежностями:

* `SwiftDataFrame` — Колоночні маніпуляції даними (залежність: `Apache Arrow Swift`).
* `SwiftStats` — Статистичне тестування та розподіли (залежність: `Accelerate vDSP`).
* `SwiftPreprocessing` — Трансформація даних (Scaler, Encoder), що генерує `MLXArray`.
* `SwiftML` — Моделі керованого навчання (регресія, дерева ухвалення рішень).
* `SwiftCluster` — Неконтрольоване навчання, зниження розмірності (PCA, K-Means).
* `SwiftOptimize` — Мета-модуль для крос-валідації та налаштування гіперпараметрів.
* `SwiftExplain` — Інтерпретація моделей (SHAP).
* `SwiftForecast` — Аналіз часових рядів (ARIMA, Kalman Filter).
* `SwiftLLM` / `SwiftVision` — Високорівневі API для генеративного ШІ (залежність: `Core AI`, `Foundation Models`).
* `SwiftPrivacy` — Криптографічне машинне навчання на зашифрованих даних (залежність: `swift-homomorphic-encryption`). - видалено

---

## ⚠️ Ризики та Залежності

| Залежність                | Ризик                                                                                                   | Стратегія мітигації                                                                                                                     |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Apache Arrow Swift`              | Нестабільні bindings, можливі breaking changes                                             | Ізолювати за`DataBuffer` протоколом; бібліотека не повинна просочуватись у публічний API |
| `Foundation Models`               | Обмежений/приватний API Apple; зміна між OS-версіями                       | Feature flag + fallback напряму на`MLX`; прив'язка до конкретної OS-версії                                           |
| `swift-homomorphic-encryption`    | Низька зрілість бібліотеки, висока обчислювальна вартість | Ізолювати у`SwiftPrivacy`; не блокує релізи до v1.0                                                                           |
| `MLXArray` (не `Sendable`)    | Порушення Swift 6 strict concurrency перевірок                                             | Розв'язано у фазі v0.4 (actor isolation +`WiredMemoryTicket`)                                                                             |
| `SwiftXGBoost` (C-інтероп) | Тимчасове рішення може стати постійним                                     | Запланувати нативну реалізацію або заміну у v0.6+                                                                   |

---

## 🚀 Поетапний План Реалізації (Phased Implementation)

### Версія 0.0: Ініціалізація Проєкту (🟢 Completed)

* **Swift Package створено:** Визначено структуру багатомодульного Swift Package (`Package.swift`, `swift-tools-version: 6.0`) з підтримкою платформ macOS 15, iOS 18, visionOS 2.
* **Зовнішні залежності підключено:** Інтегровано `apache/arrow-swift` та `ml-explore/mlx-swift` як package-залежності; налаштовано глобальні `SwiftSetting` (`StrictConcurrency`, `ExistentialAny`).
* **Модульна топологія описана:** Визначено перелік цільових модулів (`SwiftDataFrame`, `SwiftStats`, `SwiftPreprocessing`, `SwiftML`, `SwiftCluster`, `SwiftNLP`, `SwiftOptimize`) та їх залежності одне від одного.
* **Demo-проєкт (`SwiftAnalyticsDemo`):** Налаштовано окремий виконуваний Swift Package, що посилається на основний пакет через локальний `path:`-шлях. Використовується як живий інтеграційний тест усіх модулів.
* **Playground:** Створено `SwiftAnalytics_Tutorial.playground` для інтерактивного дослідження API.

### Версія 0.1: Фундамент Даних (🟢 Completed)

* **Інтеграція Arrow:** [Completed] Розробка `SwiftDataFrame` з використанням пам'яті `Apache Arrow` для доступу до даних без копіювання (zero-copy). Arrow-залежність ізолюється за `internal DataBuffer` протоколом.
* **Базова Статистика:** [Completed] Створення `SwiftStats` з використанням подпрограм `Accelerate (vDSP)` для виконання швидкої векторизованої арифметики (t-тести, ANOVA).

### Версія 0.2: Передобробка та Лінійні Моделі (🟢 Completed)

* **Конвеєр Передобробки:** [Completed] Створення інструментів масштабування та кодування у `SwiftPreprocessing`, які на виході формують тензори `MLXArray`.
* **Оптимізація MLX:** [Completed] Реалізація лінійної та логістичної регресії у `SwiftML` шляхом визначення функцій втрат на `MLXArray` та застосування вбудованої функції `valueAndGradient` для автоматичного диференціювання.

### Версія 0.3: Неконтрольоване Навчання та Текст (🟢 Completed)

* **Алгоритм PCA:** [Completed] Реалізація Principal Component Analysis (PCA) у `SwiftCluster`, який напряму мапиться на подпрограми `Accelerate LAPACK` для сингулярного розкладу матриць (SVD).
* **Кластеризація:** [Completed] Розробка K-Means, де матричні обчислення відстаней паралелізуються через `MLXArray` на GPU.
* **NLP База:** [Completed] Реалізація TF-IDF, де токенізація трансформується у частотні словники та розріджені матриці через `Swift Collections`.

### Версія 0.4: Архітектурна Стабілізація (🟢 Completed)

* **Безпека Потоків:** [Completed] Вирішення проблеми об'єкта `MLXArray`, який не підтримує протокол `Sendable` і порушує перевірки безпеки під час компіляції.
* **Ізоляція:** [Completed] Розробка стандартизованих шаблонів суворої ізоляції акторів (actor isolation) та `WiredMemoryTicket` для безпечної координації асинхронних завдань.

### Версія 0.5: Ансамблеві Моделі та Оптимізація (🟢 Completed)

* **Дерева Рішень:** [Completed] Реалізовано `DecisionTreeClassifier` (Gini/Entropy) та `DecisionTreeRegressor` (MSE) на чистому Swift з actor-ізоляцією та підтримкою нелінійних датасетів (XOR-сумісне розщеплення).
* **Випадковий Ліс:** [Completed] Реалізовано `RandomForestClassifier` та `RandomForestRegressor` з паралельним bootstrap-тренуванням через `TaskGroup`. Доступна `predictProbability` для м'якого голосування.
* **Градієнтний Бустинг:** [Completed] Нативна реалізація `GradientBoostedTreesRegressor` на чистому Swift — послідовне виправлення залишків без C-залежностей.
* **SwiftOptimize:** [Completed] Новий модуль з `Metrics` (Accuracy, Precision, Recall, F1, MSE, RMSE, MAE, R², ClassificationReport), `KFold` крос-валідацією та паралельним `GridSearchCV`.

### Версія 0.6: Аналіз Часових Рядів — `SwiftForecast` & Рефакторинг (🟢 Completed)

**Новий модуль:** `SwiftForecast` · **Compute:** `Accelerate` (vDSP + LAPACK) · **Без MLX**
**Додаткові покращення:** Інтеграція з `DataFrame` (безпосередня робота з колонками), узагальнена валідація, безпека потоків.

* **Декомпозиція часових рядів:** `TimeSeriesDecomposition` (stateless enum) — класична адитивна/мультиплікативна декомпозиція через центроване ковзне середнє (`vDSP`). ACF, PACF (Yule-Walker via `LAPACK dgesv`), ADF-тест стаціонарності.
* **Експоненційне згладжування:** `ExponentialSmoothing` (actor) — три режими: SES (Simple, один параметр α), Holt's DES (α + β, лінійний тренд), Holt-Winters (α + β + γ + period, адитивна та мультиплікативна сезонність). Автооптимізація параметрів grid search за MSE.
* **ARIMA(p, d, q) та ARIMAX:** `ARIMAModel` (actor) — диференціювання через `vDSP`, оцінка параметрів методом Conditional OLS (Hannan-Rissanen) через `LAPACK dgelsd`, рекурсивний прогноз з undifferencing, AIC. Додано підтримку екзогенних факторів (ARIMAX).
* **Фільтр Калмана:** `KalmanFilter` (actor) — лінійна модель простору станів, predict/update через `cblas_dgemm` + `LAPACK dgesv`, RTS-ретросглажування, фабричний метод `oneDimensional(processNoise:measurementNoise:)`.
* **Інтеграція з DataFrame (Glue Methods):** Додано extension-методи безпосереднього скейлінгу/кодування (`standardScale`, `minMaxScale`, `labelEncode`), кластеризації (`fitKMeans`, `fitPCA`), векторизації текстів (`fitTFIDF`) та лагування часових рядів (`withLaggedColumn`) напряму на `DataFrame`.
* **Уніфіковані інтерфейси:** Впроваджено протоколи `ClassifierEstimator` та `RegressorEstimator` та додано підтримку узагальненої крос-валідації в `CrossValidator`.
* **Діагностика збіжності:** Інтегровано перевірки на `NaN`/`Inf` під час навчання лінійних/логістичних регресій для раннього виявлення дивергенції.
* **Тести:** Розширено до 110 тест-кейсів; додано тести для нових DataFrame-екстеншенів, кастомного екранування в CSV-файлах та суворої асинхронності (Swift 6 strict concurrency).

### Версія 0.7: Тестування та Бенчмарки (🟢 Completed)

* **Swift Benchmark Suite:** Executable таргет `SwiftAnalyticsBenchmarks` з live progress (`…` / `✓`), сьюти для `SwiftStats`, `SwiftDataFrame`, `SwiftML` / `SwiftCluster`, `SwiftForecast`.
* **Python Benchmark Suite:** Дзеркальні сценарії (NumPy / Pandas / Scikit-Learn / Statsmodels); LinReg = SGD 100 epochs; Kalman = NumPy constant-velocity; Decomposition = `seasonal_decompose`.
* **Порівняння:** `compare.py` з CI gate (Forecast + Pearson + Decomposition + Kalman); DataFrame / trees — informational до 0.8. `--gate-all` для повного аудиту.
* **CI/CD:** `build-and-test` + `performance-check` на `macos-14`.
* **Arrow Integration Tests:** `testArrowRoundTripLargeBuffer`, `testArrowNullBitmapPreservation`.

### Версія 0.8: Апаратна Маршрутизація (Hardware Routing) та Розширення Передобробки *(🟢 Completed)*

**Перший фокус:** прискорення `SwiftDataFrame` (filter → sort → groupBy → CSV) vs Pandas — найбільший чесний gap у бенчмарках 0.7; впровадження повного набору методів передобробки.

* **Розширення SwiftPreprocessing (Completed):** Реалізовано 10 нових компонентів: `Imputer`, `Normalizer`, `TrainTestSplit`, `Pipeline`, `PolynomialFeatures`, `OrdinalEncoder`, `RobustScaler`, `PowerTransformer`, `KBinsDiscretizer`, `MissingValueIndicator` разом із DataFrame-екстеншенами та повним набором юніт-тестів.
* **Filter / Sort / GroupBy (Completed):** Lazy row access + typed Double filter; typed `gathered(at:)` for mask/sort; specialized sort indices; String-key `groupBy` + typed numeric agg. Release medians (100k rows): Filter ~31 ms, GroupBy ~2.3 ms (≈ pandas), Sort ~82 ms. CSV still ~174 ms.
* **Апаратна маршрутизація (Hardware Routing) (Completed):** Впроваджено шар маршрутизації в `PCA`, `LinearRegression` та `LogisticRegression` (`requestedDevice` / `resolvedDevice`). Робота перемикається між CPU (Accelerate LAPACK / чистий Swift GD) та GPU (MLX/Metal з CPU-fallbacks для непідтримуваних операцій типу SVD).
* **Алгоритм DBSCAN (Completed):** Реалізовано новий CPU-оптимізований алгоритм просторової кластеризації з шумом `DBSCAN` з BFS розширенням черги та DataFrame-екстеншеном `fitDBSCAN`.
* **Виправлення тестів (Completed):** Динамічна реєстрація ресурсного bundle `mlx-swift_Cmlx` та розміщення `default.metallib` у робочій директорії для безперешкодного тестування Metal через `swift test`.

Для забезпечення максимальної продуктивності, математичні завдання жорстко розподіляються між підсистемами:

* **CPU:** Виконує алгоритми з високим розходженням гілок (branch divergence), такі як рекурсивне розбиття в деревах ухвалення рішень та просторове індексування (DBSCAN).
* **GPU:** Обробляє матричні навантаження (K-Means, глибоке навчання) через ліниві обчислення `MLX`, які зливаються в єдине ядро `Metal` для зменшення циклів пам'яті.
* **ANE (Apple Neural Engine):** Зарезервований виключно для виконання попередньо натренованих ембеддінг-моделей та невеликих LLM.
* **M-series:** Використання переваг чипів від Apple для того, щоб показати переваги над Python бібліотеками, які працють окремо з CPU та GPU.

### Версія 0.9: Стабілізація API, Документація та Підготовка до Релізу *(🟢 Completed)*

* **DocC документація (Completed):** Повне покриття DocC-коментарями кожного публічного API (типи, методи, протоколи) з `DocC Article`-вступами (Початок роботи, Цінності, Міграція).
* **Семантичне версіонування (SemVer) (Completed):** Встановлення політики відповідно до SemVer 2.0: `major.minor.patch`. Чергування та історичні записи змін задокументовані через `CHANGELOG.md`.
* **Swift Package Index (Completed):** Додано `.spi.yml` конфігурацію для автоматичної генерації документації та збірки підтримуваних платформ на Swift Package Index.
* **API аудит (Completed):** Перевірено всі публічні інтерфейси на відповідність Swift API Design Guidelines та strict concurrency (`Sendable`).
* **`SwiftNLP` розширення (Completed):** Додано інтерфейс токенізації `Tokenizer`, subword BPE токенізатор `BPETokenizer`, та завантажувач статичних ембеддінгів `WordEmbeddings` для підготовки `SwiftLLM` (v1.0).

### Версія 1.0: LLM, Інтерпретація та Конфіденційність *(🟢 Completed)*

* **Генеративний ШІ (Completed):** Реалізовано модуль `SwiftLLM` — нативний казуальний Transformer Decoder на базі тензорів `MLX` та `MLXNN`. Додано логіт-семплер `Sampler` з підтримкою greedy search, Top-K фільтрації та температурного масштабування, а також підтримку асинхронної потокової генерації тексту (`generate(prompt:options:)` → `AsyncStream<String>`).
* **Пояснення Моделей (Completed):** Розроблено `KernelSHAP` у модулі `SwiftExplain` для конкурентного обчислення Shapley-значень чорних скриньок із використанням швидкого лінійного рішальника Gauss-Jordan.
* **Конфіденційність (PNNS) (Completed):** Реалізовано схему гомоморфного шифрування Ring-LWE та алгоритм приватного пошуку найближчого сусіда `PNNS` (Manhattan/Euclidean distance & dot product) без потреби розшифрування даних на сервері.

### Версія 1.1: Потоковий Ввід/Вивід, SafeTensors, CKKS & SARIMA *(🟢 Completed)*

* **Потоковий CSV-парсер (`SwiftDataFrame`):** [Completed] Асинхронний буферизований парсер CSV-файлів (`readCSVStream`) для потокової обробки чанками без завантаження всього файлу в пам'ять.
* **Пряме завантаження SafeTensors & GGUF (`SwiftLLM`):** [Completed] Нативні Swift-парсер вагових форматів `SafeTensors` та `GGUF` з підтримкою zero-copy відображення у `MLXArray`.
* **Схема CKKS (`SwiftPrivacy`):** [Completed] Схема гомоморфного шифрування CKKS (Cheon-Kim-Kim-Song) для обчислень над дійсними числами.
* **SARIMA & GARCH (`SwiftForecast`):** [Completed] Сезонна ARIMA (SARIMA) та розширена модель умовної волатильності GARCH(1,1).

### Версія 1.2: Перейменування SwiftSci, `addColumn` та Рефакторинг *(🟢 Completed)*

*Детальний план впровадження:* [implementation_plan_12.md](file:///Users/oleksiichumak/Developer/Xcode.projects/SwiftAnalytics/ROADMAP/implementation_plan_12.md)
* **Перейменування пакета:** Зміна назви з `SwiftAnalytics` на `SwiftSci`.
* **Форма Джозефа для Kalman Filter:** Оновлення коваріації `P` для запобігання накопиченню похибок округлення.
* **BPE-токенізатор на UTF-8 байтах:** Перехід від Character/grapheme clusters на byte-level BPE для коректної токенізації non-ASCII тексту.
* **Удосконалення DataFrame:** Захищений `castColumn`, діагностичний `gathered(at:)`, розширений `sample(n:seed:ordered:)`, новий `addColumn(_:as:using:)`.
* **Визначення долі SwiftPrivacy:** Або видалення, або перехід на `apple/swift-homomorphic-encryption`.

### Версія 1.3: Sklearn Parity Roadmap *(🟢 Completed)*

*Детальний план впровадження:* [implementation_plan_13.md](file:///Users/oleksiichumak/Developer/Xcode.projects/SwiftAnalytics/ROADMAP/implementation_plan_13.md)
* **Уніфікація Pipeline ↔ Estimator:** `ClassificationPipeline` та `RegressionPipeline` для стикування препроцесингу з фінальним класифікатором/регресором без витоку даних.
* **`ColumnTransformer`:** Маршрутизація підмножин колонок DataFrame до окремих трансформерів передобробки.
* **`RandomizedSearchCV`:** Паралельний рандомізований пошук гіперпараметрів із `KFold` крос-валідацією.
* **Розширення метрик та `predictProbability`:** Додавання 2D матриці ймовірностей класів до `ClassifierEstimator`, ROC/PR кривих, MCC, LogLoss, BrierScore.
* **Outlier Detection (`SwiftCluster`):** Алгоритми виявлення аномалій `IsolationForest` та `LocalOutlierFactor`.
* **Imbalanced Learning (`SwiftPreprocessing`):** Семплювання датасетів `SMOTE` та `RandomUndersampler`.
* **Probability Calibration (`SwiftML`):** `CalibratedClassifier` із регуляризованим Platt scaling.
* **Feature Selection & Engineering (`SwiftPreprocessing`):** `VarianceThreshold`, `SelectKBest`, `InteractionFeatures`, `DateFeatures`.
* **Time Series Transformers (`SwiftForecast`):** `LagTransformer` та `RollingWindow`.
* **Dataset Utilities (`SwiftML`):** Синтетичні генератори `makeClassification`, `makeRegression`, `makeMoons`.

---

## 🏛 Рекомендації щодо Інтеграції (Клієнтський Додаток)

Завдяки модульній структурі, бібліотека ідеально адаптується для використання в застосунках із чітким розділенням відповідальності:

* **Моделі Представлення (View Models):** Вся ініціалізація математичних моделей, завантаження даних (через `SwiftDataFrame`) та конфігурація пайплайнів передобробки виконуються у шарі View Model.
* **Фонове Виконання:** Виклики методу `.fit()` для важких алгоритмів (наприклад, Random Forest або виконання евалюації графа `MLX`) мають бути загорнуті в ізольовані фонові задачі (`Task.detached { }`), щоб не блокувати головний потік та забезпечити плавний рендеринг інтерфейсу.
* **Управління Складністю:** Внутрішні обгортки пам'яті Arrow та не-Sendable структура `MLXArray` приховані за модифікатором доступу `internal` у бібліотеці. Клієнтський застосунок працює виключно з безпечним публічним API.
