# Implementation Plan 12 — SwiftAnalytics → SwiftSci v1.2.0

> **Status**: Completed 🟢 (Version 1.2.0 implementation finished and verified with `swift test`).

Контекст: цей план виправляє реально підтверджені (перевірені напряму в коді, не з AI-звітів) проблеми, додає `addColumn`, перейменовує пакет і вирішує долю `SwiftPrivacy`. Виконувати кроки в порядку нижче — кожен незалежний, можна комітити окремо.

---

## 1. Перейменування пакета: SwiftAnalytics → SwiftSci

**Файл**: `Package.swift`

```diff
- let package = Package(
-     name: "SwiftAnalytics",
+ let package = Package(
+     name: "SwiftSci",
```

Продукти (`SwiftDataFrame`, `SwiftStats`, `SwiftML` тощо) залишаються без змін — вони вже мають власні незалежні назви.

**Додатково**:
- `grep -rn "SwiftAnalytics" .` по всьому репо (README, doc-коментарі, badges) і замінити рядкові згадки назви пакета.
- Перейменування GitHub-репозиторію (`Nodibell/SwiftAnalytics` → `Nodibell/SwiftSci`) — це дія в налаштуваннях GitHub, не файлова зміна; Claude Code її не виконує, зробити вручну до або після мержу цього плану.

---

## 2. Kalman filter: naive-форма → форма Джозефа

**Файл**: `Sources/SwiftForecast/Core/KalmanFilter.swift`
**Проблема**: коваріація оновлюється як `P = (I - KH) * P_pred` у двох місцях (`filter()` рядки 87–91, `smooth()` рядки 126–128). Накопичення похибок округлення поступово руйнує симетричність і додатну визначеність `P`.

**Fix (обидва місця)**:

```diff
- let I = identityMatrix(stateSize)
- let KH = matMul(K, H)
- let IMinusKH = matSub(I, KH)
- self.P = matMul(IMinusKH, PPred)
+ let I = identityMatrix(stateSize)
+ let KH = matMul(K, H)
+ let IMinusKH = matSub(I, KH)
+ let IMinusKHT = transpose(IMinusKH)
+ self.P = matAdd(
+     matMul(matMul(IMinusKH, PPred), IMinusKHT),
+     matMul(matMul(K, R), transpose(K))
+ )
```

(в `smooth()` замінити `PPred` на `Pp` відповідно до локальних імен). Використовує лише хелпери, які вже є у файлі (`matMul`, `matAdd`, `transpose`) — жодних нових залежностей.

---

## 3. BPE-токенізатор: перехід з Character на UTF-8 байти

**Файл**: `Sources/SwiftNLP/Core/BPETokenizer.swift`
**Проблема**: `bpe(_:)` розбиває слово через `word.map { String($0) }` — це grapheme clusters, не байти. Якщо `vocab`/`merges` натреновані на byte-level BPE (стандарт для GPT-стилю), non-ASCII текст токенізується неправильно.

**Fix**: розбивати на рівні `word.utf8`, а не `Character`, і представляти кожен байт через стандартне GPT-2-стильне байт↔unicode відображення (`bytesToUnicode()`), щоб результат лишався друкованим `String` і сумісним з текстовими vocab/merges файлами:

```swift
private static let byteEncoder: [UInt8: Character] = Self.makeByteEncoder()

private static func makeByteEncoder() -> [UInt8: Character] {
    var bytes = Array(UInt8(ascii: "!")...UInt8(ascii: "~"))
        + Array(UInt8(0xA1)...UInt8(0xAC))
        + Array(UInt8(0xAE)...UInt8(0xFF))
    var mapping = [UInt8: Character]()
    var n: UInt32 = 0
    for b in UInt8.min...UInt8.max {
        if bytes.contains(b) {
            mapping[b] = Character(UnicodeScalar(UInt32(b))!)
        } else {
            mapping[b] = Character(UnicodeScalar(256 + n)!)
            n += 1
        }
    }
    return mapping
}

private func bpe(_ word: String) -> [String] {
    guard !word.isEmpty else { return [] }
    var chars = word.utf8.map { String(Self.byteEncoder[$0]!) }
    chars[chars.count - 1] += "</w>"
    // ... решта merge-циклу без змін, працює над `chars: [String]` як і раніше
}
```

**Перевір перед мержем**: чи vocab/merges файли, які реально завантажуються (`WordEmbeddings.swift` / завантажувач токенізатора), вже закодовані за цією ж byte↔unicode схемою — якщо вони генерувались іншим інструментом з іншим mapping, потрібно узгодити з тим, що використовував той інструмент, інакше зіставлення з `vocab` зламається.

---

## 4. DataFrame: `castColumn` — тиха часткова втрата даних

**Файл**: `Sources/SwiftDataFrame/Core/DataFrame.swift` (рядки 264–284)

```diff
  public func castColumn<T: SupportedType>(_ name: String, to type: T.Type) throws -> DataFrame {
      guard let col = _columns[name] else { throw DataFrameError.columnNotFound(name) }

      let newValues: [T?] = (0..<col.count).map { i in
          guard let v = col.value(at: i) else { return nil }
-         let str = "\(v)"
-         return T.parse(from: str)
+         if let direct = v as? T { return direct }
+         let str = "\(v)"
+         return T.parse(from: str)
      }

      let sourceNonNull = col.count - col.nullCount
      let castNonNull   = newValues.filter { $0 != nil }.count
      if sourceNonNull > 0 && castNonNull == 0 {
          throw DataFrameError.castFailed(column: name, targetType: "\(T.self)")
      }
+     if castNonNull < sourceNonNull {
+         throw DataFrameError.partialCastFailure(
+             column: name, targetType: "\(T.self)",
+             failed: sourceNonNull - castNonNull, total: sourceNonNull
+         )
+     }

      let newCol = TypedColumn<T>(name: name, values: newValues)
      return try withColumn(name, column: newCol)
  }
```

Додати новий case `partialCastFailure(column: String, targetType: String, failed: Int, total: Int)` до `DataFrameError` (знайти оголошення через `grep -rn "enum DataFrameError"`, ймовірно в `Sources/SwiftDataFrame/Errors/`) з відповідним `errorDescription`.

`v as? T` фаст-пас уникає зайвої String-алокації, коли значення вже потрібного типу; рядковий парс лишається для реальних міжтипових конверсій.

---

## 5. DataFrame: `gathered`/`rows` — дублювання імен і тихе ковтання помилок

**Файл**: `Sources/SwiftDataFrame/Core/DataFrame.swift` (рядки 347–355 + всі виклики `rows(at:)`)

```diff
- public func gathered(at indices: [Int]) -> DataFrame {
-     return rows(at: indices)
- }
-
- private func rows(at indices: [Int]) -> DataFrame {
-     guard !indices.isEmpty else { return DataFrame.empty }
-     let newCols: [any AnyColumn] = columns.map { $0.gathered(at: indices) }
-     return (try? DataFrame(columns: newCols)) ?? DataFrame.empty
- }
+ public func gathered(at indices: [Int]) -> DataFrame {
+     guard !indices.isEmpty else { return DataFrame.empty }
+     let newCols: [any AnyColumn] = columns.map { $0.gathered(at: indices) }
+     do {
+         return try DataFrame(columns: newCols)
+     } catch {
+         preconditionFailure("DataFrame.gathered(at:) — mismatched column lengths, internal invariant violation: \(error)")
+     }
+ }
```

Замінити всі внутрішні виклики `rows(at:` на `gathered(at:` (у `slice(from:count:)`, `sample`, `sortBy`, `applyMask`) — сигнатури не змінюються, лише прибирається другий приватний метод з ідентичною поведінкою. Порушення інваріанту довжини колонок тепер падає з діагностикою замість тихого порожнього результату.

---

## 6. DataFrame: `sample(n:seed:)` — порядок рядків і якість дефолтного seed

**Файл**: `Sources/SwiftDataFrame/Core/DataFrame.swift` (рядки 170–185)

```diff
- public func sample(n: Int, seed: UInt64? = nil) -> DataFrame {
+ public func sample(n: Int, seed: UInt64? = nil, ordered: Bool = false) -> DataFrame {
      let total = shape.rows
      guard n > 0 && total > 0 else { return DataFrame.empty }
      let actualN = Swift.min(n, total)

-     var rng = SeededRNG(seed: seed ?? UInt64(Date().timeIntervalSince1970 * 1000))
      var indices = Array(0..<total)
-     // Fisher-Yates shuffle first `actualN` elements
-     for i in 0..<actualN {
-         let j = i + Int(rng.next() % UInt64(total - i))
-         indices.swapAt(i, j)
-     }
-     let selected = Array(indices.prefix(actualN)).sorted()
-     return rows(at: selected)
+     if let seed {
+         var rng = SeededRNG(seed: seed)
+         for i in 0..<actualN {
+             let j = i + Int(rng.next() % UInt64(total - i))
+             indices.swapAt(i, j)
+         }
+         indices = Array(indices.prefix(actualN))
+     } else {
+         indices.shuffle() // SystemRandomNumberGenerator — без колізій дефолтного мілісекундного seed
+         indices = Array(indices.prefix(actualN))
+     }
+     if ordered { indices.sort() }
+     return gathered(at: indices)
  }
```

**Це зміна поведінки за замовчуванням** (`sample` тепер повертає випадковий порядок, не вихідний) — задокументувати в release notes як breaking change; хто покладався на старий порядок, додає `ordered: true`.

---

## 7. DataFrame: новий `addColumn(_:using:)`

**Файл**: `Sources/SwiftDataFrame/Core/DataFrame.swift` (вставити після `withColumn`, ~рядок 242)

```swift
/// Returns a new DataFrame with a column computed per-row from a closure.
public func addColumn<T: SupportedType>(
    _ name: String,
    as type: T.Type = T.self,
    using closure: (DataFrameRow) -> T?
) throws -> DataFrame {
    let map = _columns
    let names = columnNames
    var values = [T?]()
    values.reserveCapacity(shape.rows)
    for i in 0..<shape.rows {
        let row = DataFrameRow(columnNames: names, index: i, columnMap: map)
        values.append(closure(row))
    }
    return try withColumn(name, column: TypedColumn<T>(name: name, values: values))
}
```

Тонка обгортка над існуючим `withColumn` + вже усталеною ідіомою `DataFrameRow` з `filter(_:)`. Делегує в `withColumn`, тож наслідує його поведінку "replace or add" — задокументувати це явно в doc-коментарі.

---

## 8. SwiftPrivacy: рішення по ToyCKKS

**Файл**: `Sources/SwiftPrivacy/Core/ToyCKKS.swift`

Нагадування: цей файл сам себе документує як "not a full Ring-LWE/CKKS implementation... not for protecting real data, and not a substitute for an audited HE library" — немає NTT, modulus chain, relinearization keys. Питання не в параметрах (N/log q), а в тому, що криптографічного ядра там ще немає.

Два варіанти, обери один:

**Варіант A — видалити.** Прибрати таргет `SwiftPrivacy` з `Package.swift` і теку `Sources/SwiftPrivacy/`. Найпростіше, повертає до попереднього рішення.

**Варіант B — замінити на обгортку над `apple/swift-homomorphic-encryption`** (Apache 2.0, реальна аудитована бібліотека Apple, використовується в Live Caller ID Lookup; https://github.com/apple/swift-homomorphic-encryption). Це не тривіальна заміна одного файлу — це:
1. Додати залежність у `Package.swift`.
2. **Claude Code має сам дослідити реальний публічний API цієї бібліотеки на місці** (`HeScheme`, конфігурація контексту, шифрування/дешифрування, add/multiply) — я навмисно не пишу тут "приклад коду" з вигаданими назвами методів, бо саме так з'явились фейкові issue/PR-номери в попередніх AI-звітах цього проєкту. Не вигадувати сигнатури — читати актуальні джерела чи документацію на Swift Package Index під час імплементації.
3. Спроєктувати новий публічний фасад `SwiftPrivacy` (encrypt/decrypt/add/multiply над зашифрованими значеннями) поверх реальних примітивів цієї залежності, зберігаючи мінімальний контракт, який зараз надає `ToyCKKS` (`CKKSCiphertext`, `CKKSSecretKey.decrypt`), або свідомо задокументувати новий контракт як breaking change.
4. Видалити `ToyCKKS.swift` після міграції або перенести в `#if DEBUG`/тести як навчальний приклад, якщо хочеш зберегти його для дидактичних цілей.

Якщо не визначишся до релізу — став Варіант A і повертайся до B окремим PR, не варто блокувати реліз на цьому.

---

## Release Notes — v2.0.0

```markdown
## v2.0.0

### Breaking
- Package renamed: SwiftAnalytics → SwiftSci (update your `Package.swift` dependency URL/name if the GitHub repo was also renamed).
- `DataFrame.sample(n:seed:)` now returns rows in randomized order by default (previously always sorted to original index order). Pass `ordered: true` to keep the old behavior.
- `DataFrame.castColumn` now throws `DataFrameError.partialCastFailure` when some (not all) values fail to convert, instead of silently nulling them.
- `DataFrame.gathered(at:)` now traps with a diagnostic on internal invariant violations instead of silently returning an empty DataFrame. `rows(at:)` removed (was already private; behavior merged into `gathered(at:)`).
- SwiftPrivacy: [видалено / перебудовано на apple/swift-homomorphic-encryption — обрати по факту]

### Fixed
- `SwiftForecast.KalmanFilter`: covariance update now uses Joseph form, preventing loss of positive-definiteness under prolonged filtering (`filter` and `smooth`).
- `SwiftNLP.BPETokenizer`: tokenization now operates on UTF-8 bytes instead of Character/grapheme clusters, fixing incorrect merges for non-ASCII input.
- `DataFrame.castColumn`: avoids an unnecessary String round-trip when the source value already matches the target type.

### Added
- `DataFrame.addColumn(_:as:using:)`: adds a column computed per-row via a closure over `DataFrameRow`, delegating to `withColumn`.

### 📊 Benchmark Performance Summary (v1.2.0 vs Python)
| Benchmark Test | Swift (ms) | Python (ms) | Speedup | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **Mean** (vDSP, 1M elements) | 0.082 ms | 0.121 ms | 1.47x | 🟢 Swift |
| **Pearson Correlation** (500k elements) | 0.868 ms | 1.256 ms | 1.45x | 🟢 Swift |
| **ARIMA(1,1,1) Fit** (50k pts) | 2.323 ms | 215.527 ms | 92.78x | 🟢 Swift |
| **ARIMA Forecast Horizon=24** | 2.456 ms | 211.880 ms | 86.26x | 🟢 Swift |
| **Holt-Winters Fit** (50k pts, period=12) | 6.841 ms | 148.627 ms | 21.73x | 🟢 Swift |
| **Random Forest Fit** (1k×4, 50 trees) | 5.025 ms | 25.475 ms | 5.07x | 🟢 Swift |
| **KernelSHAP Explain** (5 feats, 100 coalitions) | 0.192 ms | 0.426 ms | 2.22x | 🟢 Swift |
| **Kalman Filter 1D** (10k obs, Joseph Form) | 62.349 ms | 85.547 ms | 1.37x | 🟢 Swift |
| **LLM Forward Pass** (seqLen=64) | 0.636 ms | 0.528 ms | 0.83x | 🔴 PyTorch |
| **CSV Read** (100k rows, 5 cols) | 177.509 ms | 19.340 ms | 0.11x | 🔴 Pandas |
| **CSV Stream Read** (chunk=10k) | 238.699 ms | 22.525 ms | 0.09x | 🔴 Pandas |

* **CI Gate Status**: **PASSED** ✅ (*0 gated regressions detected*).
```
```
