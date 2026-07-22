# Implementation Plan — Version 0.4: Архітектурна Стабілізація (Swift 6)

**Версія:** 0.4  
**Модулі:** `SwiftML`, `SwiftCluster`, `SwiftPreprocessing` (Оновлення)  
**Мета:** Забезпечити повну сумісність екосистеми з вимогами **Swift 6 Strict Concurrency**, ізолювати не-`Sendable` типи (`MLXArray`) за допомогою акторів та реалізувати паттерн `WiredMemoryTicket` для координації доступу до Unified Memory на Apple Silicon.

---

## Зміст

1. [Огляд архітектурної проблеми](#1-огляд-архітектурної-проблеми)
2. [Ізоляція моделей (Actor Isolation)](#2-ізоляція-моделей-actor-isolation)
3. [Управління пам'яттю (WiredMemoryTicket)](#3-управління-памяттю-wiredmemoryticket)
4. [Зміни у конфігурації Swift 6 та Package.swift](#4-зміни-у-конфігурації-swift-6-та-packageswift)
5. [План верифікації](#5-план-верифікації)

---

## 1. Огляд архітектурної проблеми

У Swift 6 увімкнено сувору перевірку безпеки потоків (Strict Concurrency). Основна проблема нашої екосистеми:
* Об'єкти **`MLXArray`** представляють вказівники на буфери Metal GPU і не відповідають протоколу `Sendable`.
* Передача `MLXArray` між потоками (наприклад, з фонового Task в головний потік) викликає помилки компіляції.
* Моделі `LinearRegression`, `LogisticRegression` та `KMeans` є класами з мутабельним станом (ваги, центроїди), що робить їх небезпечними для паралельного використання.
* Ліниві обчислення MLX можуть накопичувати великі графи обчислень у Unified Memory, викликаючи витік пам'яті (Memory Bloat) при асинхронних навантаженнях.

---

## 2. Ізоляція моделей (Actor Isolation)

Для вирішення проблем безпеки потоків ми перетворимо основні класи моделей на **`actor`** або ізолюємо їхні інтерфейси за допомогою `Sendable` обгорток.

### 2.1. Перетворення моделей на `actor`
Моделі, які зберігають мутабельний стан GPU (`LinearRegression`, `LogisticRegression`, `KMeans`), будуть оголошені як `actor`:

```swift
public actor LinearRegression {
    public private(set) var weights: MLXArray?
    public private(set) var bias: MLXArray?
    
    public init() {}
    
    // Асинхронний метод навчання, що приймає Sendable-сумісні типи даних
    public func fit(X: [[Double]], y: [Double], learningRate: Double = 0.01, epochs: Int = 100) async throws {
        // Конвертація у локальні MLXArray на території актора
        let xArray = MLXArray(X.flatMap { $0.map { Float($0) } }).reshaped([X.count, X[0].count])
        let yArray = MLXArray(y.map { Float($0) })
        
        try await trainInternal(X: xArray, y: yArray, lr: learningRate, epochs: epochs)
    }
    
    // Асинхронний прогноз
    public func predict(X: [[Double]]) async throws -> [Double] {
        guard let weights = self.weights, let bias = self.bias else {
            throw MLError.fittingRequired
        }
        let xArray = MLXArray(X.flatMap { $0.map { Float($0) } }).reshaped([X.count, X[0].count])
        let predictions = (xArray.matmul(weights) + bias).asArray(Float.self)
        return predictions.map { Double($0) }
    }
}
```

* **Чому це працює**: Всі операції всередині `actor` виконуються послідовно на його власному серіалізованому виконавці (executor).
* **Приховування не-Sendable типів**: `MLXArray` більше не виходить за межі актора. Клієнтський потік передає звичайні `Sendable` матриці `[[Double]]` та отримує назад `[Double]`.

---

## 3. Управління пам'яттю (WiredMemoryTicket)

Unified Memory Architecture (UMA) на Apple Silicon дозволяє CPU та GPU ділитися однією пам'яттю. При великих паралельних обчисленнях нам потрібен обмежувач (rate limiter) та механізм примусового вивільнення ресурсів Metal.

### 3.1. WiredMemoryManager
Актор-координатор, який розподіляє дозволи («квитки») на виконання важких обчислень:

```swift
public actor WiredMemoryManager {
    public static let shared = WiredMemoryManager(maxConcurrentTasks: 2)
    
    private let maxConcurrentTasks: Int
    private var activeTasksCount = 0
    private var suspensionQueue = [CheckedContinuation<Void, Never>]()
    
    public init(maxConcurrentTasks: Int) {
        self.maxConcurrentTasks = maxConcurrentTasks
    }
    
    /// Запит на виконання важкої операції
    public func acquireTicket() async -> WiredMemoryTicket {
        if activeTasksCount < maxConcurrentTasks {
            activeTasksCount += 1
            return WiredMemoryTicket(manager: self)
        }
        
        return await withCheckedContinuation { continuation in
            suspensionQueue.append(continuation)
        }
    }
    
    /// Звільнення квитка
    public func releaseTicket() {
        activeTasksCount -= 1
        if !suspensionQueue.isEmpty {
            activeTasksCount += 1
            let next = suspensionQueue.removeFirst()
            next.resume()
        }
    }
}
```

### 3.2. WiredMemoryTicket
Клас-контейнер, що автоматично звільняє ресурси MLX при виході з області видимості:

```swift
public final class WiredMemoryTicket: Sendable {
    private let manager: WiredMemoryManager
    
    internal init(manager: WiredMemoryManager) {
        self.manager = manager
    }
    
    /// Примусова евалюація та очищення кешу Metal
    public func finish() async {
        // Очищаємо кеш виділеної пам'яті MLX (Metal Allocator)
        MLX.GPU.clearCache() 
        await manager.releaseTicket()
    }
    
    deinit {
        // Запобіжник: якщо користувач забув викликати finish()
        Task {
            await manager.releaseTicket()
        }
    }
}
```

---

## 4. Зміни у конфігурації Swift 6 та Package.swift

Щоб увімкнути суворі перевірки багатопоточності на рівні всього пакета `SwiftAnalytics`, ми додамо наступні налаштування у `Package.swift`:

```swift
let globalSwiftSettings: [SwiftSetting] = [
    .enableUpcomingFeature("StrictConcurrency"),
    .enableUpcomingFeature("ExistentialAny")
]

// Додати ці налаштування до кожного таргету:
.target(
    name: "SwiftML",
    dependencies: ["SwiftPreprocessing", ...],
    path: "Sources/SwiftML",
    swiftSettings: globalSwiftSettings
)
```

---

## 5. План верифікації

### Автоматичні тести (Swift Concurrency Tests):
1. **Тест на стан гонитви (Race Condition)**:
   * Запуск 10 паралельних завдань `Task.detached`, які одночасно викликають навчання або прогнозування на одному акторі моделі. Перевірка відсутності витоків пам'яті та коректності результатів.
2. **Перевірка ліміту пам'яті (WiredMemoryManager)**:
   * Тест, який запускає 5 одночасних тренувань на GPU і перевіряє, що не більше `maxConcurrentTasks` виконується в один момент часу, а інші чекають на квиток.
3. **Строга перевірка Concurrency**:
   * Збірка пакета командою `swift build --configuration release` з активованим прапором `StrictConcurrency` без жодного попередження компілятора (warnings).
