# Version 1.6: DataFrame ↔ ML Bridge, NLP/Time-Series Completion & Dependency Hygiene

## Background

Три незалежні джерела вказують на одні й ті самі відкладені прогалини:
1. Titanic-демо показала, що `Imputer`/`LabelEncoder`/`OneHotEncoder` не мають DataFrame-входу — кожен реальний workflow руками пише `addColumn` + type-branching цикл для побудови feature-матриці.
2. Оригінальний sklearn-parity список (план 13) досі має незакриті пункти, які жоден наступний changelog не згадував: `Stemmer`/`Lemmatizer`/`StopWords`/`TextNormalizer`, `HolidayEncoder`, `makeClusters`/`makeCircles`.
3. Аудит залежностей показав, що `SwiftOptimize`/`SwiftForecast`/`SwiftExplain` досі декларують невикористану залежність від `SwiftDataFrame`.

Це не performance-реліз (той — v1.5) — це реліз завершення й гігієни.

---

## Part A — DataFrame ↔ ML/Preprocessing Bridge

### A1. `SwiftDataFrame` — Matrix Extraction

Корінь проблеми: немає жодного способу перетворити підмножину колонок DataFrame у `[[Double]]`/`[Double]` без ручного циклу з type-branching, який до того ж мовчки нулить непідтримувані типи.

**[NEW]** `Sources/SwiftDataFrame/Core/DataFrame+Matrix.swift`

```swift
extension DataFrame {
    /// Extracts named columns as a row-major [[Double]] matrix.
    /// Throws `DataFrameError.castFailed` for any column that isn't Double, Int64, or Bool.
    public func toFeatureMatrix(_ columns: [String]) throws -> [[Double]] {
        var matrix = [[Double]](repeating: [Double](repeating: 0, count: columns.count), count: shape.rows)
        for (colIdx, name) in columns.enumerated() {
            if let col = self[column: name, as: Double.self] {
                for row in 0..<shape.rows { matrix[row][colIdx] = col[row] ?? .nan }
            } else if let col = self[column: name, as: Int64.self] {
                for row in 0..<shape.rows { matrix[row][colIdx] = col[row].map(Double.init) ?? .nan }
            } else if let col = self[column: name, as: Bool.self] {
                for row in 0..<shape.rows { matrix[row][colIdx] = (col[row] == true) ? 1.0 : 0.0 }
            } else {
                throw DataFrameError.castFailed(column: name, targetType: "Double")
            }
        }
        return matrix
    }

    public func toTargetVector(_ column: String) throws -> [Double] {
        try toFeatureMatrix([column]).map { $0[0] }
    }
}
```

Реалізує саме те, що зараз руками пише кожен виклик сайт (порівняй з ручним циклом побудови `XRaw` і `yRaw` у Titanic-демо) — і кидає помилку, а не мовчки нулить, на невідомому типі колонки.

### A2. `SwiftPreprocessing` — DataFrame-Native Fit/Transform

Розширити вже існуючий `Sources/SwiftPreprocessing/Core/DataFrame+Preprocessing.swift` (не новий файл — там уже є DataFrame-міст):

```swift
extension PreprocessingTransformer {
    public func fit(_ df: DataFrame, columns: [String]) throws {
        try fit(try df.toFeatureMatrix(columns))
    }
    public func transform(_ df: DataFrame, columns: [String]) throws -> DataFrame {
        let result = try transform(try df.toFeatureMatrix(columns))
        var out = df
        for (i, name) in columns.enumerated() {
            out = try out.withColumn(name, column: TypedColumn<Double>(
                name: name, values: result.map { Optional($0[i]) }
            ))
        }
        return out
    }
}
```

Заміняє ручне `addColumn`-імпутування/енкодування на `imputer.transform(df, columns: ["Age", "Fare"])`, безпосередньо використовуючи вже існуючий `Imputer`/`OneHotEncoder`.

### A3. `SwiftML` — Естіматори напряму з DataFrame

**Важлива відмінність від A1/A2**: `SwiftML` наразі **не залежить** від `SwiftDataFrame` взагалі (нуль файлів це підтверджують). Це треба додати явно:

**[MODIFY]** `Package.swift` — додати `"SwiftDataFrame"` до `dependencies:` таргету `SwiftML`.

**[NEW]** `Sources/SwiftML/Core/DataFrame+ML.swift`

```swift
extension ClassifierEstimator {
    public func fit(_ df: DataFrame, features: [String], target: String) async throws {
        try await fit(features: try df.toFeatureMatrix(features), targets: try df.toTargetVector(target))
    }
    public func predict(_ df: DataFrame, features: [String]) async throws -> [Int] {
        try await predict(features: try df.toFeatureMatrix(features))
    }
}
// дзеркально для RegressorEstimator
```

### A4. `SwiftDataFrame` — CSV Type Override

Поки `A1`/`A5` (план 1.5) уже переписують `CSVReader.swift` — додати вихід з Bool-перед-Int неоднозначності:

**[MODIFY]** `Sources/SwiftDataFrame/IO/CSVReader.swift` — `CSVReadOptions`

```swift
public struct CSVReadOptions: Sendable {
    // існуючі поля...
    public var columnTypeOverrides: [String: ColumnDType] = [:]
}
```

`inferColumn(name:cells:)` перевіряє `columnTypeOverrides[name]` перед авто-визначенням — 0/1-колонка, яка семантично не Bool, більше не "вгадується" неправильно.

---

## Part B — NLP Completion (`SwiftNLP`)

Із оригінального sklearn-parity списку (план 13) — єдине, що жоден наступний реліз не торкнувся.

- **`StopWords`**: статичний набір + `filter(tokens:language:) -> [String]`. Найдешевше й найменш ризиковане з чотирьох. ✅ Можна імплементувати одразу.
- **`TextNormalizer`**: lowercase, Unicode-нормалізація (NFC/NFKC через `String`), пунктуація — механічне, без мовної залежності. ✅ Можна імплементувати одразу.
- **`Stemmer`/`Lemmatizer`**: ⛔ **BLOCKED — не імплементувати, доки не визначена мова.** Див. Open Questions. Claude Code: пропустити цей пункт повністю, не вгадувати scope.

---

## Part C — Dataset & Time-Series Loose Ends

### C1. `SwiftML` — `DatasetUtilities` доповнення

`makeClassification`/`makeRegression`/`makeMoons` вже є (`DatasetUtilities.swift`); `makeClusters`/`makeCircles` з оригінального списку — ні. Ідентичний патерн генерації, що вже є для `makeMoons`. ✅ Можна імплементувати одразу.

### C2. `SwiftForecast` — `HolidayEncoder`

⛔ **BLOCKED — не імплементувати, доки не визначені календарі/локалі.** Див. Open Questions. Claude Code: пропустити цей пункт повністю, не вгадувати scope.

---

## Part D — Dependency Hygiene

**[MODIFY]** `Package.swift` — прибрати `"SwiftDataFrame"` з `dependencies:` таргетів `SwiftOptimize`, `SwiftForecast`, `SwiftExplain` (задекларовано, нуль файлів модуля її фактично використовують).

---

## Verification Plan

Не performance-реліз — акцент на тестах, не бенчмарках:

```
swift test --configuration release -Xswiftc -O
```

- `toFeatureMatrix`: тест на змішаний Double/Int64/Bool набір колонок + тест, що невідомий тип кидає `castFailed`, а не нулить.
- `ClassifierEstimator.fit(_:features:target:)`: end-to-end на Titanic-подібному DataFrame, звірити результат з ручним шляхом (як у демо) — має збігатись побітово.
- Dependency hygiene: `swift build` для `SwiftOptimize`/`SwiftForecast`/`SwiftExplain` окремо (`--target`) після видалення залежності — має збиратись без змін у їхньому коді.
- `columnTypeOverrides`: regression-тест на колонку з тільки 0/1 значеннями, форсовану на `.int64`.

---

## Open Questions

- **Stemmer/Lemmatizer — яка(і) мова(и)?** Англійська (Porter/Snowball — стандартні, добре описані алгоритми) суттєво простіша за українську (морфологія, потрібен зовнішній словник на кшталт pymorphy-аналогів). Визначити scope перед стартом, інакше ризик того самого "правильний API, невірний результат", що вже було з BPE.
- **HolidayEncoder — які локалі/календарі?** Тільки US/ISO, чи включно з UA-святами? Джерело даних для свят (статична таблиця на N років, чи обчислювані правила)?
- **`toFeatureMatrix` на Bool-колонках** — зараз мапить `true→1.0/false→0.0`, ідентично до того, що Titanic-демо робила вручну для `Survived`. Чи достатньо цього, чи потрібен явний параметр для контролю мапінгу (наприклад, якщо колонка не бінарна семантично)?
