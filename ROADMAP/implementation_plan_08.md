# План реалізації — Версія 0.8: Апаратна Маршрутизація (Hardware Routing)

**Версія:** 0.8  
**Цільові модулі:** `SwiftCluster`, `SwiftML`, `SwiftForecast`, `SwiftStats`, `SwiftPreprocessing`  
**Мета:** Впровадження інтелектуальної системи апаратної маршрутизації (Hardware Routing) обчислень між CPU (Accelerate/vDSP/LAPACK), GPU (MLX/Metal) та ANE (Apple Neural Engine via CoreML). Використання переваг архітектури об'єднаної пам'яті (Unified Memory Architecture - UMA) в Apple Silicon для мінімізації накладних витрат на перенесення даних та забезпечення максимальної швидкості обчислень.

---

## Зміст

1. [Архітектурне дослідження та концепція](#1-архітектурне-дослідження-та-концепція)
2. [Компоненти та API апаратної маршрутизації](#2-компоненти-та-api-апаратної-маршрутизації)
   - 2.1 [Enum ExecutionDevice та конфігурація](#21-enum-executiondevice-та-конфігурація)
   - 2.2 [HardwareRouter & Device Manager](#22-hardwarerouter--device-manager)
3. [Рефакторинг існуючих алгоритмів](#3-рефакторинг-існуючих-алгоритмів)
   - 3.1 [K-Means: GPU (MLX) vs CPU (vDSP)](#31-k-means-gpu-mlx-vs-cpu-vdsp)
   - 3.2 [PCA: CPU (LAPACK SVD) vs GPU (MLX)](#32-pca-cpu-lapack-svd-vs-gpu-mlx)
   - 3.3 [Linear & Logistic Regression: CPU (vDSP GD) vs GPU (MLX)](#33-linear--logistic-regression-cpu-vdsp-gd-vs-gpu-mlx)
4. [Нові алгоритми для демонстрації CPU Routing](#4-нові-алгоритми-для-демонстрації-cpu-routing)
   - 4.1 [DBSCAN (Clustering з високим branch divergence)](#41-dbscan-clustering-з-високим-branch-divergence)
5. [План верифікації та бенчмаркінгу](#5-план-верифікації-та-бенчмаркінгу)
   - 5.1 [Автоматичні тести](#51-автоматичні-тести)
   - 5.2 [Бенчмарки продуктивності (CPU vs GPU vs Auto)](#52-бенчмарки-продуктивності-cpu-vs-gpu-vs-auto)
6. [Відкриті питання](#6-відкриті-питання)

---

## 1. Архітектурне дослідження та концепція

Apple Silicon чипи (M1/M2/M3/M4 та їх варіації Pro/Max/Ultra) мають унікальну архітектуру об'єднаної пам'яті (UMA). Це означає, що CPU, GPU та ANE мають доступ до однієї фізичної RAM. Проте в коді операції з різними фреймворками мають різні накладні витрати:
- **Accelerate (vDSP/LAPACK):** Працює на CPU. Дуже низька латентність запуску, ідеально для малих та середніх масивів, але обмежена паралельність ядрами CPU.
- **MLX (Metal):** Ліниві обчислення на GPU. Має фіксовані накладні витрати на диспетчеризацію та компіляцію графа обчислень (Metal JIT). Ефективно для великих матриць, глибокого навчання та паралельних операцій без розгалужень (branching).
- **CoreML/ANE:** Оптимізовано для фіксованих шарів нейромереж (інференсу). Важкий запуск, але практично нульове енергоспоживання та висока пропускна здатність для підтримуваних операцій.

### Стратегія маршрутизації (Routing Strategy):

| Алгоритм / Сценарій | CPU (vDSP / Swift) | GPU (MLX / Metal) | ANE (CoreML) | Обґрунтування |
| :--- | :---: | :---: | :---: | :--- |
| **Decision Tree / RF / GBDT** | **Основний** | Ні | Ні | Високий branch divergence (умовні переходи). Погана паралелізація на SIMD/GPU. |
| **K-Means (малі дані < 1k)** | **Основний** | Fallback | Ні | GPU overhead на старт і копіювання перевищує час обчислень на CPU. |
| **K-Means (великі дані >= 1k)**| Fallback | **Основний** | Ні | GPU-прискорення за рахунок паралельного обчислення відстаней між тисячами точок. |
| **PCA (SVD)** | **Основний (LAPACK)**| Альтернатива | Ні | LAPACK SVD на CPU є надзвичайно оптимізованим для не-велетенських матриць. |
| **DBSCAN** | **Основний** | Ні | Ні | Алгоритм обходу графа / просторового пошуку (Kd-Tree). Не підходить для GPU. |
| **Linear / Logistic Reg (GD)** | Fallback | **Основний** | Ні | Прості матричні множення та оновлення градієнтів чудово лягають на MLX/GPU. |
| **Embeddings / Transformer** | Ні | Fallback | **Основний** | ANE ідеально підходить для запуску тензорних операцій трансформерів. |

---

## 2. Компоненти та API апаратної маршрутизації

Створимо новий внутрішній підмодуль або структуру конфігурації в `SwiftPreprocessing` чи в кожному модулі окремо, яка дозволить динамічно вибирати бbackend.

### 2.1 Enum ExecutionDevice та конфігурація

Введемо глобальний тип пристрою виконання:

```swift
public enum ExecutionDevice: String, Sendable, Codable {
    case cpu
    case gpu
    case ane
    case auto
}
```

Кожен оцінювач (Estimator), такий як `KMeans`, `PCA`, `LinearRegression`, отримає опціональний параметр `device: ExecutionDevice` у своєму ініціалізаторі (за замовчуванням `.auto`).

### 2.2 HardwareRouter & Device Manager

Створимо допоміжний актор `HardwareRouter` для керування пристроями:

```swift
public actor HardwareRouter {
    public static let shared = HardwareRouter()
    
    private init() {}
    
    /// Визначає оптимальний пристрій для завдання на основі розміру вхідних даних
    public func resolveDevice(
        for algorithm: String,
        sampleCount: Int,
        featureCount: Int,
        requestedDevice: ExecutionDevice
    ) -> ExecutionDevice {
        guard requestedDevice == .auto else {
            return requestedDevice
        }
        
        switch algorithm {
        case "KMeans":
            // Якщо точок менше 1000, GPU Metal JIT overhead не вартий запуску
            return (sampleCount * featureCount < 4000) ? .cpu : .gpu
        case "PCA":
            // LAPACK dgesvd дуже швидкий на CPU до розмірів 2000x2000
            return (sampleCount < 2000 && featureCount < 500) ? .cpu : .gpu
        case "LinearRegression", "LogisticRegression":
            return (sampleCount < 1000) ? .cpu : .gpu
        default:
            return .cpu
        }
    }
    
    /// Налаштовує MLX Device context перед виконанням обчислень
    public func setMLXDevice(to device: ExecutionDevice) {
        #if canImport(MLX)
        import MLX
        switch device {
        case .gpu:
            MLX.Device.setDefault(device: .gpu)
        case .cpu:
            MLX.Device.setDefault(device: .cpu)
        default:
            MLX.Device.setDefault(device: .gpu) // За замовчуванням для MLX
        }
        #endif
    }
}
```

---

## 3. Рефакторинг існуючих алгоритмів

### 3.1 K-Means: GPU (MLX) vs CPU (vDSP)

Поточна реалізація `KMeans` у `SwiftCluster` жорстко використовує `MLX` (який за замовчуванням виконується на GPU).
Ми проведемо рефакторинг:

1. **Додамо CPU-backend** з використанням `vDSP` та чистого Swift паралелізму.
2. **Маршрутизація**:
   - Якщо `device == .cpu`, обчислюємо відстані через векторизовані операції `vDSP.distanceSquared` або `vDSP.linearInterpolation` та паралелізуємо розподіл точок по кластерах за допомогою `TaskGroup`.
   - Якщо `device == .gpu`, використовуємо наявний MLX-код, але попередньо викликаємо `HardwareRouter.shared.setMLXDevice(to: .gpu)`.

**Приклад CPU-обчислення відстаней (vDSP):**
```swift
// Обчислення відстаней на CPU для кожної точки до центроїдів
func computeDistancesCPU(features: [[Double]], centroids: [[Double]]) -> [Int] {
    let nPoints = features.count
    let nClusters = centroids.count
    var labels = [Int](repeating: 0, count: nPoints)
    
    // Паралелізуємо по точках через TaskGroup
    DispatchQueue.concurrentPerform(iterations: nPoints) { i in
        let point = features[i]
        var minDistance = Double.greatestFiniteMagnitude
        var bestLabel = 0
        
        for k in 0..<nClusters {
            let centroid = centroids[k]
            // vDSP-оптимізоване обчислення евклідової відстані
            var dist = 0.0
            vDSP_distancesqD(point, 1, centroid, 1, &dist, vDSP_Length(point.count))
            if dist < minDistance {
                minDistance = dist
                bestLabel = k
            }
        }
        labels[i] = bestLabel
    }
    return labels
}
```

### 3.2 PCA: CPU (LAPACK SVD) vs GPU (MLX)

Поточна реалізація `PCA` жорстко використовує `dgesvd` з Accelerate LAPACK (CPU).
Додамо альтернативний GPU-backend через `MLX`:

1. Зсув середнього (centering) виконуємо засобами MLX.
2. Обчислюємо коваріаційну матрицю або виконуємо лінійний розклад.
3. Оскільки MLX має нативні тензорні операції розкладу (наприклад, `mlx.core.linalg.svd`), ми можемо перенаправити обчислення туди при `device == .gpu`.

### 3.3 Linear & Logistic Regression: CPU (vDSP GD) vs GPU (MLX)

Для невеликих моделей навчання на GPU займає більше часу через overhead.
Додамо CPU-версію градієнтного спуску для `LinearRegression` та `LogisticRegression`:
1. Використовуємо `vDSP` для оновлення ваг: `w = w - lr * grad`.
2. Матричне множення `X * w` виконуємо через `cblas_dgemm` (BLAS).

---

## 4. Нові алгоритми для демонстрації CPU Routing

Щоб повністю розкрити потенціал CPU Routing у версії 0.8, додамо алгоритм кластеризації, який має високу розгалуженість та погано працює на GPU.

### 4.1 DBSCAN (Clustering з високим branch divergence)

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** ідеально підходить для демонстрації CPU Routing, оскільки він базується на чергах, просторових запитах сусідові (Eps-neighborhood) та не містить фіксованих матричних множень.

**Специфікація DBSCAN:**
- **Модуль**: `SwiftCluster`
- **Клас**: `public actor DBSCAN`
- **Параметри**: `eps: Double`, `minSamples: Int`, `device: ExecutionDevice = .cpu` (якщо користувач спробує вибрати `.gpu`, буде автоматичний fallback на `.cpu` з попередженням).
- **Алгоритм**:
  1. Побудова Kd-Tree (або простого просторового індексу) на CPU для швидкого пошуку найближчих сусідів (Eps-neighborhood).
  2. Обхід точок, маркування як Core, Border أو Noise.
  3. Використання черги (BFS/DFS) для розширення кластерів.

---

## 5. План верифікації та бенчмаркінгу

### 5.1 Автоматичні тести

Додамо нову групу тестів `HardwareRoutingTests.swift` у `SwiftClusterTests` та `SwiftMLTests`:

1. **`testKMeansDeviceRouting`**:
   - Ініціалізація `KMeans` з примусовим `.cpu` та `.gpu`.
   - Перевірка, що результати кластеризації на однаковому датасеті збігаються (з точністю до `1e-4`).
   - Перевірка, що при виборі `.cpu` властивість `centroids` обчислюється без виклику графів MLX (перевірка внутрішнього стану пристрою).
2. **`testPCADeviceRouting`**:
   - Порівняння результатів проектування PCA на CPU (LAPACK) та GPU (MLX).
3. **`testDBSCANFallback`**:
   - Спроба ініціалізувати `DBSCAN` з `.gpu` та перевірка, що система коректно виконує fallback на `.cpu` та успішно завершує кластеризацію.

### 5.2 Бенчмарки продуктивності (CPU vs GPU vs Auto)

У наш Benchmark Suite (створений у версії 0.7) додамо порівняльний аналіз пристроїв:
- Запуск `KMeans` на матрицях розміром $100 \times 4$, $1000 \times 4$, $50000 \times 4$ з примусовими `.cpu` та `.gpu`.
- Очікуваний результат:
  - На $100 \times 4$: CPU має бути швидшим за GPU (через нульовий Metal overhead).
  - На $50000 \times 4$: GPU має показати значне прискорення (наприклад, 5x-10x) порівняно з CPU.
- Результати мають автоматично заноситися до `baseline.json` у `Benchmarks/Results/`.

---

## 6. Відкриті питання

> [!IMPORTANT]
> **Питання 1: Підтримка ANE (Apple Neural Engine)**
> Чи варто реалізовувати нативний CoreML-конвертер для простих лінійних моделей у цій версії, чи обмежитись підготовкою шару маршрутизації (ANE Placeholder) для майбутньої інтеграції зі `SwiftLLM` (v1.0)?
> *Рекомендація:* Обмежитись створенням структури маршрутизації та тестуванням CPU/GPU, оскільки повноцінне використання ANE потребує компіляції `.mlmodel` файлів під час runtime, що додасть велику кількість важкого коду Xcode-SDK.

> [!WARNING]
> **Питання 2: Вплив Strict Concurrency у Swift 6 на MLX Device context**
> Оскільки `MLX.Device.setDefault(device:)` є глобальним побічним ефектом (глобальний стан), паралельний запуск кількох моделей з різними налаштуваннями `device` у різних задачах (`Task.detached`) може призвести до race conditions на контексті пристрою.
> *Рішення:* Будь-яка зміна пристрою повинна синхронізуватись через `WiredMemoryManager` або виконуватись послідовно. Наш актор `HardwareRouter` має повністю контролювати ці виклики.
