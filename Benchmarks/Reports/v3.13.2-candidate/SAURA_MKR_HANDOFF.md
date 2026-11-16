# Передача результатів SwiftSci `v3.13.2-candidate` для Saura та Магістерської кваліфікаційної роботи (МКР)

- **Статус артефактів:** `CANDIDATE` (Stage A — перевірено на живленні від мережі AC, робоче дерево містить незакомічені зміни)
- **Базовий коміт:** `0cf99f0470d0757d0a02bd3244c62d821ab144b0` (`v3.13.1`, `dirty: true`)
- **Stage A Snapshot `tree_sha256`:** `5abd55baabb35be660ab990e97eb37bd009c9d8a955b09c447d290a083bdee30`
- **Відбиток джерел після фінальної перевірки:** `953e773629eabf731530b82edd6d74813e8d5f4b22bfec25b412ba9951eb4a9c` (чотири правки DocC та видалення кінцевих пробілів у `ARIMA.swift`).
- **Вердикт обов'язкового релізного гейту G-037:** **`FIXED AND VERIFIED`**
- **Примітка про обмеження:** Цей документ містить рекомендації для подальшого оновлення Saura та МКР; код Saura та файли МКР у межах цієї сесії **не змінювалися**.

---

## 1. Обов'язковий гейт G-037: коректність `fittedValues` та `residuals` в ARIMA / SARIMA

### 1.1 Першопричина дефекту у `v3.13.1`

У релізі `v3.13.1` методи `ARIMAModel.integrate(fittedDiff:original:d:isForecast:)` (`Sources/SwiftForecast/Core/ARIMA.swift`) та `SARIMAModel.integrate(fittedDiff:original:d:D:s:isForecast:)` (`Sources/SwiftForecast/Core/SARIMAModel.swift`) відновлювали внутрішньовибіркові (in-sample) прогнозні значення `fittedValues` при `isForecast == false` за **розімкненим контуром кумулятивного сумування (open-loop integration)**:

1. Для несезонного інтегрування `lastVal` ініціалізувався значенням `levelSeries[0]`, після чого на кожному кроці виконувалося `val = lastVal + diffVal; lastVal = val` — тобто до наступного кроку додавалося попереднє **спрогнозоване**, а не фактичне спостереження.
2. Для сезонного інтегрування в `SARIMAModel` цикл виконував `val = reconstructed[i] + current[i]`, тобто так само накопичував похибки власного попереднього прогнозу замість фактичного рівня `subLevelSeries[i]`.
3. Водночас `residuals` обчислювалися у замкненому циклі на різницевій шкалі (`diffResiduals[t] = diffSeries[t] - diffFitted[t]`). Через це `residuals` відображали істинні однокрокові інновації, а `fittedValues` дрейфували як випадкове блукання по всій історії ряду. Додатково в `ARIMAModel` при наявності екзогенних регресорів (`ARIMAX`) у `fittedValues` був пропущений компонент `withExogenousComponent(...)`.

Саме через цей дрейф Saura була змушена застосовувати тимчасовий обхідний розрахунок на рівні відображення:
`fitted[t] = actual[t] - residual[t - warmUp]`.

### 1.2 Математичне визначення та контракт у `v3.13.2-candidate`

У `v3.13.2-candidate` інтегрування для `isForecast == false` переведене на **замкнений однокроковий контур (closed-loop one-step-ahead reconstruction)** від фактичних значень `levelSeries[i]` / `subLevelSeries[i]`, а для `ARIMAX` відновлено додавання екзогенного внеску:

- Нехай $\text{warmUp} = d$ для `ARIMAModel` і $\text{warmUp} = d + D \cdot s$ для `SARIMAModel`.
- `fittedValues.count == series.count`
- `residuals.count == series.count - warmUp`
- На початковому інтервалі `t in 0..<warmUp`: `fittedValues[t] == series[t]` (граничні початкові спостереження без прогнозного зсуву).
- На інтервалі `t in warmUp..<series.count`: `fittedValues[t]` є **однокроковим умовним прогнозом у вихідній шкалі ряду** $y_t$, побудованим виключно за даними до моменту `t - 1`, так що виконується точна тотожність:

$$e_{t - \text{warmUp}} = y_t - \hat{y}_t \iff \text{residuals}[t - \text{warmUp}] = \text{series}[t] - \text{fittedValues}[t]$$

### 1.3 Верифікація G-037

1. **Модульні регресійні тести (`Tests/SwiftForecastTests/ARIMASARIMAFittedValuesTests.swift`):**
   - Перевірено строгу каузальну однокрокову рівність та тотожність `series[t] - residuals[t - warmUp] == fittedValues[t]` (допуск `< 1e-9` / `< 1e-10`) для `ARIMAModel` (`d = 0, 1, 2`), `ARIMA(0, 1, 0)`, `ARIMAX(1, 1, 0)` і `(0, 1, 0)`, `SARIMAModel` (`D = 1, 2`), `AutoARIMA` на `AirPassengers`, а також відсутність витоку майбутніх даних при збуренні `series[143] += 5000.0`.
2. **Stage A наукова валідація (`Benchmarks/Runs/v3.13.2-candidate/scientific_validation.json`):**
   - `ts_sarima_airpassengers_144_g037` (`144` точки, `warm_up=13`, `residuals_count=131`): `max_warmup_diff = 0.0`, `max_identity_diff = 5.684e-14`, `saura_formula_exact_match = true`, in-sample `RMSE = 28.3051`, `R² = 0.8554`, `late_1960_max_err = 23.7976`.
   - `sarima_ap_ro` (rolling-origin зрізи `108`, `120`, `132` на AirPassengers): на всіх 3 зрізах `max_warmup_diff = 0.0`, `max_identity_diff = 5.684e-14`, `saura_formula_exact_match = true`.
   - `ts_arima_syn500_matched_hr`, `ts_arima_dgp_syn500_matched_hr`, `arima_dgp_ro`, `ts_arima_dgp_syn50k_matched_hr`, `ts_auto_arima_valid`: усі 10 застосовних прогонів пройшли перевірку (`max_identity_diff <= 5.684e-14`, `saura_formula_exact_match = true`).

---

## 2. Практичні рекомендації для Saura (після затвердження та релізу `v3.13.2`)

1. **Вилучення тимчасового обхідного коду для `fittedValues`:**
   - Після оновлення залежності SwiftSci до `v3.13.2` у Saura можна прибрати дисплейну реконструкцію `fitted[t] = actual[t] - residual[t - warmUp]` та напряму прив'язувати графіки до `ARIMAModel.fittedValues`, `SARIMAModel.fittedValues` або `ARIMAResult.forecast.fittedValues`.
   - Зверніть увагу: перші `warmUp = series.count - residuals.count` елементів (`0..<warmUp`) містять вихідні граничні точки `series[0..<warmUp]`. Якщо UI Saura бажає приховувати граничну зону прогріву на графіку, слід просто пропускати індекси `0..<warmUp`, не перераховуючи самі значення `fittedValues`.
2. **Безпечна обробка помилок навчання (G-033):**
   - Тепер при незворотному MA/SMA-поліномі або появі `NaN`/`Infinity` методи `fit` / `forecast` / `aic` / `bic` гарантовано кидають типізовану `ForecastError`, атомарно очищують стан і залишають `isFitted == false`. Переконайтеся, що UI-стан Saura обробляє помилки `fit` як стан «модель не навчена».
3. **Узгодженість формул `Stats.skewness` / `Stats.kurtosis` та `TreeSHAP` (G-035/G-036):**
   - Поведінка `Stats.skewness` та `Stats.kurtosis` зі знаменником `ddof = 1` збережена для API-сумісності `v3.13.x`; у підказках/довідці Saura варто зазначати, що це вибіркова реалізація `ddof = 1` (наприклад, для `[1..5]` ексцес дорівнює `-1.912`).
   - Для пояснень `TreeSHAP` підтримуються `DecisionTree*` та `RandomForest*` із рівномірною вагою листків `1.0` (`exact` при `K <= 12`, seeded Monte Carlo `2048` пермутацій при `K > 12`); пояснення для `GradientBoostedTreesRegressor` та ймовірнісних класифікаторів не активовані в `v3.13.x`.

---

## 3. Рекомендації для оновлення тексту МКР (Магістерської роботи)

### 3.1 Що оновити в описі алгоритмів та верифікації

1. **Часові ряди (`SwiftForecast`):**
   - Задокументувати двохетапну умовну МНК-оцінку Ханнана–Ріссанена (`ARIMAModel`, `SARIMAModel`), перевірку стійкості/зворотності AR/MA поліномів за рекурсією Шура–Кона (`AutoARIMA.isPolynomialStable` / `isMAPolynomialInvertible`) та замкнене однокрокове відновлення `fittedValues` (виправлення G-033 і G-037).
   - Навести чесне порівняння якості поза вибіркою (rolling-origin):
     - `hw_syn500_ro` (Holt-Winters, 3 зрізи): Swift `RMSE = 1.5110` проти Python `1.5540` і Seasonal Naive `5.3479`.
     - `sarima_ap_ro` (AirPassengers, зрізи `108/120/132`, `h=12`): Swift `RMSE = 32.5422` проти Python `30.9928` та Seasonal Naive `38.9917`.
   - Чесно вказати межу застосування `TimeSeriesAnomalyDetector`: висока точність на стаціонарно-сезонних/помірних рядах (`syn144`: `F1 = 1.0`; `syn500_holdout`: `F1 = 0.9091`), але підвищена чутливість до хибних спрацьовувань на краях ряду при різкому лінійному тренді (`anomaly_syn500_trended_diagnostic`: `Precision = 0.20, Recall = 1.0, F1 = 0.3333`) через підстановку `medTrend` на краях центрованого ковзного середнього.
2. **Інтерпретованість моделей (`SwiftExplain` / `TreeSHAP`):**
   - Замінити згадки про підтримку GBDT або поліноміальний TreeSHAP $O(T L D^2)$ на фактично реалізований та верифікований алгоритм: точний перебір коаліцій $O(T \cdot 2^K \cdot L)$ при $K \le 12$ (перевірено проти коаліційного оракула з розбіжністю $\le 4.44 \times 10^{-16}$) і детермінований Monte Carlo (`2048` пермутацій) при $K > 12$ з рівномірною вагою листків `1.0` для дерев рішень і випадкових лісів. Явно віднести GBDT TreeSHAP та ймовірнісні/мультикласові пояснення (G-035, G-036) до обмежень, відкладених за межі `v3.13.2`.
3. **Описова статистика (`SwiftStats`):**
   - Явно зазначити, що `Stats.skewness` та `Stats.kurtosis` використовують центральні моменти з корекцією знаменника `ddof = 1` (збережено для сумісності `v3.13.x`), і для прямого зіставлення зі `scipy.stats(..., bias=True)` застосовується аналітичний перерахунок, перевірений з точністю до $2.13 \times 10^{-14}$ на $N = 100\,000$.
4. **Бенчмарки продуктивності (лише відтворювана методологія):**
   - Вилучити з МКР будь-які цифри з ліквідованого `--research` режиму (зокрема застарілі оцінки `174×` для Holt-Winters, `172×` для ARIMA Kalman MLE, `12.23×` для KernelSHAP, а також попередні таблиці RandomForest/GBDT `--research`).
   - Використовувати виключно результати `bench.py` (`standard`, `migration`, `public-data`, `tabular-migration`, `nist`) та accuracy-gated `scientific_validation_suite.py` із явним розділенням співставних і неспівставних оцінювачів, супресією етапів `< 1 us` та зазначенням, що Peak RSS виміряно як `ru_maxrss` усього процесу.

---

## 4. Залишкові блокери перед фінальним релізом `v3.13.2` (Stage B)

Поточні артефакти мають статус **`CANDIDATE` (Stage A)** і **не повинні** цитуватися як заморожений тег `v3.13.2` до закриття таких блокерів:

1. **Коміт та тег `v3.13.2`:** Робоче дерево наразі містить незакомічені зміни (`dirty: true`) відносно коміту `0cf99f0470d0757d0a02bd3244c62d821ab144b0`. Потрібне явне схвалення користувача на `git commit` та тегування `v3.13.2` (і `git push` за потреби).
2. **Фінальний Stage B прогін на чистому тегу:** Після створення чистого коміту/тегу необхідно повторно зібрати воркер і прогнати `bench.py` + `scientific_validation_suite.py` + `acceptance.py` у `Benchmarks/Runs/v3.13.2/`, щоб сертифікати зафіксували `dirty: false` і фінальний `HEAD` SHA для остаточних звітів без суфікса `-candidate`.
3. **Платформне обмеження та зовнішні скрипти:** Поточна перевірка охоплює лише `Apple Silicon arm64` (`Apple M4 Pro`, `macOS 27.2.1`, `AC` живлення); `x86_64` та зовнішній `/Users/oleksiichumak/Developer/MCW/Benchmarks/measure_startup.py` не є частиною цього релізного прогону `v3.13.2-candidate`.
