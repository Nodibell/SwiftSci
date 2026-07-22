# Implementation Plan — Version 0.5: Ансамблеві Моделі та Оптимізація

**Версія:** 0.5  
**Модулі:** `SwiftML` (Нові моделі), `SwiftOptimize` (Новий таргет), `Package.swift` (Оновлення)  
**Мета:** Додати класичні алгоритми машинного навчання на основі дерев рішень (Decision Trees, Random Forest, GBDT) та створити інструменти для валідації й підбору гіперпараметрів моделей у новому модулі `SwiftOptimize`.

---

## Зміст

1. [Огляд та архітектурне дослідження](#1-огляд-та-архітектурне-дослідження)
2. [Дерева рішень та випадковий ліс (SwiftML)](#2-дерева-рішень-та-випадковий-ліс-swiftml)
3. [Новий таргет SwiftOptimize](#3-новий-таргет-swiftoptimize)
4. [План змін у файловій структурі](#4-план-змін-у-файловій-структурі)
5. [План верифікації](#5-план-верифікації)

---

## 1. Огляд та архітектурне дослідження

На відміну від лінійних моделей, які чудово прискорюються на GPU, алгоритми на основі дерев рішень містять багато логічних розгалужень (branch divergence). Вони є більш ефективними при виконанні на багатоклавіатурних CPU.

Ми розробимо:
* **Decision Tree** (Classifier/Regressor) на чистому Swift з паралельним пошуком найкращого розщеплення ознак (split) через `TaskGroup`.
* **Random Forest** (Classifier/Regressor), де навчання кожного окремого дерева відбувається повністю паралельно на фонових потоках.
* **Gradient Boosted Decision Trees (GBDT)** — нативна реалізація бустингу регресійних дерев (послідовне виправлення залишків). Це дозволяє уникнути складних С-залежностей на кшталт `libxgboost`.
* **SwiftOptimize** — новий таргет, що реалізує розрахунок метрик (F1, MSE тощо), `K-Fold` крос-валідацію та `GridSearchCV` для паралельного пошуку найкращих параметрів.

---

## 2. Дерева рішень та випадковий ліс (SwiftML)

### 2.1. Структура дерева рішень (Decision Tree)
Ми створимо загальну структуру вузла дерева:

```swift
public final class DecisionTreeNode: Sendable {
    public let featureIndex: Int?
    public let threshold: Double?
    public let left: DecisionTreeNode?
    public let right: DecisionTreeNode?
    public let value: Double? // Значення для регресії або ймовірність/клас для класифікації
    
    public var isLeaf: Bool { left == nil && right == nil }
}
```

* **Класифікація**: Критерій розщеплення — **Gini Impurity** (Нечистота Джіні) або **Entropy** (Ентропія).
* **Регресія**: Критерій розщеплення — **Mean Squared Error (MSE)** (Середньоквадратична помилка).

Для оптимізації пошуку найкращого спліту при великій кількості ознак, сканування порогових значень (thresholds) буде паралелізовано за допомогою `TaskGroup`:

```swift
// Паралельний пошук сплітів для кожної ознаки
let bestSplit = await withTaskGroup(of: SplitResult?.self) { group in
    for featureIdx in 0..<numFeatures {
        group.addTask {
            return findBestSplitForFeature(featureIdx, X: X, y: y)
        }
    }
    // Агрегація та вибір найкращого результату
}
```

### 2.2. Випадковий ліс (Random Forest)
Random Forest навчає $N$ дерев на випадкових підвибірках даних (Bootstrapping) та випадкових підмножинах ознак (Feature Bagging).

Оскільки дерева не залежать одне від одного під час навчання, весь ліс буде навчатися паралельно:

```swift
public actor RandomForestClassifier {
    private var trees: [DecisionTreeClassifier] = []
    
    public func fit(features: [[Double]], targets: [Double], nEstimators: Int = 100) async throws {
        self.trees = try await withThrowingTaskGroup(of: DecisionTreeClassifier.self) { group in
            for _ in 0..<nEstimators {
                group.addTask {
                    let (bootstrappedX, bootstrappedY) = bootstrapSample(features, targets)
                    let tree = DecisionTreeClassifier(maxDepth: 10)
                    try tree.fit(features: bootstrappedX, targets: bootstrappedY)
                    return tree
                }
            }
            
            var trainedTrees = [DecisionTreeClassifier]()
            for try await tree in group {
                trainedTrees.append(tree)
            }
            return trainedTrees
        }
    }
}
```

---

## 3. Новий таргет SwiftOptimize

Модуль `SwiftOptimize` фокусується на оцінці та налаштуванні моделей.

### 3.1. Розрахунок метрик (Metrics)
* `classificationReport(yTrue: [Int], yPred: [Int])` -> Точність (Precision), Повнота (Recall), F1-Score, Акуратність (Accuracy).
* `meanSquaredError(yTrue: [Double], yPred: [Double])` -> MSE.
* `r2Score(yTrue: [Double], yPred: [Double])` -> Коефіцієнт детермінації $R^2$.

### 3.2. K-Fold Крос-валідація
Клас `KFold` ділить набір даних на $K$ частин (folds). На кожній ітерації одна частина служить валідаційною, а інші — тренувальними. Навчання та тестування на кожному fold відбуваються паралельно.

### 3.3. Grid Search (GridSearchCV)
Паралельний пошук оптимальних гіперпараметрів (наприклад, `maxDepth` для дерева, `learningRate` для регресії):

```swift
public struct GridSearchCV {
    // Приймає конфігураційну сітку та фабрику створення моделей
    // Повертає найкращі гіперпараметри та оцінку моделі
}
```

---

## 4. План змін у файловій структурі

### [Package.swift](file:///Users/oleksiichumak/Developer/Xcode.projects/SwiftAnalytics/SwiftAnalytics/Package.swift)
* Додати бібліотеку `SwiftOptimize`.
* Додати таргет `SwiftOptimize` та його тести `SwiftOptimizeTests`.

### [MODIFY] SwiftML
* Додати `Sources/SwiftML/Core/DecisionTree.swift` [NEW]
* Додати `Sources/SwiftML/Core/RandomForest.swift` [NEW]
* Додати `Sources/SwiftML/Core/GradientBoosting.swift` [NEW]

### [NEW] SwiftOptimize
* `Sources/SwiftOptimize/Metrics/Metrics.swift`
* `Sources/SwiftOptimize/Validation/KFold.swift`
* `Sources/SwiftOptimize/Search/GridSearchCV.swift`

---

## 5. План верифікації

### Автоматичні тести:
1. **Decision Tree & Random Forest Tests**:
   * Перевірка здатності класифікувати нелінійні датасети (наприклад, XOR-проблема) та збіжності регресійних дерев.
2. **K-Fold & Grid Search Tests**:
   * Перевірка валідаційного спліту та знаходження заздалегідь відомих найкращих параметрів синтетичної моделі.

### Інтеграційний тест у Demo:
* Навчання `RandomForestClassifier` на згенерованому у `DataFrame` нелінійному наборі даних.
* Пошук глибини дерев через `GridSearchCV`.
* Виведення таблиці класифікації (Classification Report).
