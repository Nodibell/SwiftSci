# Implementation Plan — Version 0.3: Неконтрольоване Навчання та Текст

**Версія:** 0.3  
**Модулі:** `SwiftCluster`, `SwiftNLP`  
**Мета:** Реалізувати інструменти неконтрольованого навчання (PCA на базі Accelerate LAPACK, K-Means на базі GPU MLX) та базові алгоритми обробки природної мови (TF-IDF для векторизації текстів).

---

## Зміст

1. [Огляд архітектури](#1-огляд-архітектури)
2. [Package.swift — нові таргет-модулі](#2-packageswift--нові-таргет-модулі)
3. [SwiftCluster](#3-swiftcluster)
   - [Principal Component Analysis (PCA)](#31-principal-component-analysis-pca)
   - [K-Means Clustering](#32-k-means-clustering)
4. [SwiftNLP](#4-swiftnlp)
   - [Векторизатор TF-IDF](#41-векторизатор-tf-idf)
5. [Тести та верифікація](#5-тести-та-верифікація)
6. [Критерії виходу (Definition of Done)](#6-критерії-виходу-definition-of-done)

---

## 1. Огляд архітектури

У версії 0.3 ми додаємо два нових модулі в пакет `SwiftAnalytics`:

```
┌────────────────────────────────────────────────────────────────────────┐
│                            Клієнтський код                             │
└──────────────────┬───────────────────┬───────────────────┬─────────────┘
                   │                   │                   │
         ┌─────────▼─────────┐   ┌─────▼─────────────┐   ┌─▼───────────┐
         │  SwiftDataFrame   │   │   SwiftCluster    │   │  SwiftNLP   │
         │ (Data Ingestion)  │   │  (PCA, K-Means)   │   │  (TF-IDF)   │
         └─────────┬─────────┘   └─────┬───────────┬─┘   └─────────────┘
                   │                   │           │
                   │         ┌─────────▼─┐   ┌─────▼─────────────┐
                   │         │Accelerate │   │     mlx-swift     │
                   │         │ (LAPACK)  │   │  (Metal / GPU)    │
                   └─────────┴───────────┴───┴───────────────────┘
```

* **`SwiftCluster`** розділить свої обчислення:
  * **PCA** використовуватиме **Accelerate LAPACK** (CPU) для точного та швидкого сингулярного розкладу матриць (SVD).
  * **K-Means** працюватиме на **MLX** (GPU) для швидкого паралельного обчислення відстаней між точками та центроїдами.
* **`SwiftNLP`** працюватиме на чистому Swift з використанням `Swift Collections` для токенізації, побудови словників та обчислення частот термінів.

---

## 2. Package.swift — нові таргет-модулі

Ми розширюємо конфігурацію нашого Swift-пакета, додаючи два нові бібліотечні продукти та відповідні таргети:

```swift
// Products
.library(name: "SwiftCluster",     targets: ["SwiftCluster"]),
.library(name: "SwiftNLP",         targets: ["SwiftNLP"]),

// Targets
.target(
    name: "SwiftCluster",
    dependencies: [
        "SwiftDataFrame",
        .product(name: "MLX", package: "mlx-swift")
    ],
    path: "Sources/SwiftCluster"
),
.testTarget(
    name: "SwiftClusterTests",
    dependencies: ["SwiftCluster"],
    path: "Tests/SwiftClusterTests"
),

.target(
    name: "SwiftNLP",
    dependencies: [
        "SwiftDataFrame"
    ],
    path: "Sources/SwiftNLP"
),
.testTarget(
    name: "SwiftNLPTests",
    dependencies: ["SwiftNLP"],
    path: "Tests/SwiftNLPTests"
)
```

---

## 3. SwiftCluster

Модуль надаватиме інструменти кластеризації та зниження розмірності ознак.

### 3.1. Principal Component Analysis (PCA)

PCA використовується для зниження розмірності числових ознак шляхом проєктування даних на ортогональні напрямки максимальної дисперсії (головні компоненти).

#### Математичний алгоритм:
1. Центрування даних: від кожного стовпця матриці віднімається його середнє значення.
2. Обчислення сингулярного розкладу матриці (SVD) центрованої матриці $X$ розміром $N \times M$:
   $$X = U \Sigma V^T$$
   Де $V$ — матриця правих сингулярних векторів (напрямки головних компонент), а $\Sigma$ містить сингулярні числа, пропорційні дисперсії.
3. Проєкція даних на перші $k$ головних компонент:
   $$X_{projected} = X \cdot V_{1..k}$$

#### Інтеграція з Accelerate LAPACK:
Для SVD обчислень на CPU використовується функція `dgesdd_` або `dgesvd_` з фреймворку `Accelerate`:
```swift
import Accelerate

// Виклик DGESVD для розкладу матриці подвійної точності
// jobu: "N" (не обчислювати U), jobvt: "S" (обчислити перші мін(N,M) векторів V^T)
```

#### Публічний API (`PCA`):
```swift
public final class PCA {
    public let nComponents: Int
    public private(set) var mean: [Double]?
    public private(set) var components: [[Double]]? // [nComponents, nFeatures]
    public private(set) var explainedVariance: [Double]?
    
    public init(nComponents: Int) {
        self.nComponents = nComponents
    }
    
    public func fit(_ X: [[Double]]) throws
    public func transform(_ X: [[Double]]) throws -> [[Double]]
    public func fitTransform(_ X: [[Double]]) throws -> [[Double]]
}
```

---

### 3.2. K-Means Clustering

Алгоритм розділяє точки даних на $K$ кластерів, де кожна точка належить до кластера з найближчим середнім значенням (центроїдом).

#### Алгоритм на базі MLX (GPU):
1. **Ініціалізація**: центроїди вибираються випадково або за допомогою K-Means++ з вхідного тензора $X$ (розмір `[N, M]`).
2. **Крок призначення**: для кожного вектора обчислюється евклідова відстань до всіх центроїдів. Точка призначається найближчому центроїду.
   * Розрахунок матриці відстаней на MLX:
     $$\text{Distances} = \sqrt{\sum (X_i - C_j)^2}$$
     За допомогою трансляції розмірностей (broadcasting):
     `let diff = X.expandedDimensions(axes: [1]) - centroids.expandedDimensions(axes: [0])`
     `let dists = sqrt((diff * diff).sum(axis: -1))`
3. **Крок оновлення**: кожен центроїд оновлюється як середнє арифметичне всіх точок, призначених цьому центроїду.
4. **Критерій зупинки**: ітерації завершуються, коли зміна центроїдів стає меншою за заданий поріг `tol` або досягнуто `maxIterations`.

#### Публічний API (`KMeans`):
```swift
public final class KMeans {
    public let nClusters: Int
    public let maxIterations: Int
    public let tolerance: Double
    
    public private(set) var centroids: MLXArray? // [nClusters, nFeatures]
    
    public init(nClusters: Int, maxIterations: Int = 300, tolerance: Double = 1e-4) {
        self.nClusters = nClusters
        self.maxIterations = maxIterations
        self.tolerance = tolerance
    }
    
    public func fit(X: MLXArray) throws
    public func predict(X: MLXArray) throws -> MLXArray // Повертає індекси кластерів [N]
}
```

---

## 4. SwiftNLP

Модуль для роботи з текстовими даними та вилучення текстових ознак.

### 4.1. Векторизатор TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) використовується для оцінки важливості слова в контексті документа, який є частиною колекції документів (корпусу).

#### Математична модель:
1. **Term Frequency (TF)**: частота слова в поточному документі.
   $$TF(t, d) = \frac{\text{Кількість входжень } t \text{ в } d}{\text{Загальна кількість слів в } d}$$
2. **Inverse Document Frequency (IDF)**: міра унікальності слова в усьому корпусі.
   $$IDF(t, D) = \log \left( \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} \right) + 1$$
3. **TF-IDF**:
   $$TF\text{-}IDF(t, d, D) = TF(t, d) \times IDF(t, D)$$

#### Публічний API (`TFIDFVectorizer`):
```swift
public final class TFIDFVectorizer {
    public private(set) var vocabulary: [String: Int] = [:] // Слово -> Індекс
    public private(set) var idfs: [Double] = [] // IDF для кожного слова у словнику
    
    public init() {}
    
    /// Навчає словник та обчислює IDF на корпусі текстів
    public func fit(_ documents: [String]) throws
    
    /// Трансформує тексти в матрицю TF-IDF [кількість документів, розмір словника]
    public func transform(_ documents: [String]) throws -> [[Double]]
    
    public func fitTransform(_ documents: [String]) throws -> [[Double]]
}
```

#### Деталі токенізації:
* Приведення до нижнього регістру.
* Видалення пунктуації та спецсимволів за допомогою `CharacterSet.alphanumerics.inverted`.
* Розбиття по пробілах.
* Опціонально: фільтрація базових англійських стоп-слів (a, an, the, and, of, to, in, is, that, etc.).

---

## 5. Тести та верифікація

Нові модулі покриваються тестами за допомогою бібліотеки `Testing`:

### `SwiftCluster` Tests:
* **PCA**:
  * Порівняння результатів зниження розмірності з еталоном (наприклад, scikit-learn).
  * Ортогональність отриманих головних компонент (скалярний добуток різних векторів компонент має бути $\approx 0$).
  * Обробка одновимірних або пустих масивів (має викликати помилки `ClusterError.emptyInput`).
* **K-Means**:
  * Кластеризація штучного датасету з чітко розділеними 3 хмарами точок. Алгоритм повинен ідеально визначити центри та розподілити мітки.
  * Перевірка збіжності за меншу кількість кроків, ніж `maxIterations`.

### `SwiftNLP` Tests:
* **TFIDFVectorizer**:
  * Тест на правильність токенізації (ігнорування регістру, крапок, ком).
  * Розрахунок TF-IDF вручну для короткого тестового корпусу з 3 речень та звірка з кодом.
  * Перевірка роботи з невідомими словами при `transform()` (вони мають ігноруватися без помилок).

---

## 6. Критерії виходу (Definition of Done)

- `[x]` Всі нові таргети (`SwiftCluster`, `SwiftNLP`) та відповідні тестові таргети додано у `Package.swift`.
- `[x]` Реалізовано повний функціонал:
  - PCA на базі Accelerate LAPACK SVD.
  - KMeans на базі GPU MLX.
  - TFIDFVectorizer на чистому Swift з базовою токенізацією.
- `[x]` `swift build` та `swift test` успішно збираються та проходять без попереджень чи помилок.
- `[x]` Юніт-тести покривають не менше 80% логіки нових модулів.
- `[x]` Створено демонстраційний приклад використання нових фіч у тестовому проєкті `SwiftAnalyticsDemo`.
