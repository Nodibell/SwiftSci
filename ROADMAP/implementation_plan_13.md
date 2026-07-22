# Implementation Plan 13 — v1.3.0: sklearn-parity roadmap

Список у джерельному документі писався без доступу до реального коду — половина "критичних відсутніх" речей вже є. Нижче: спершу корекція, потім план тільки на те, чого справді бракує.

---

## 0. Корекція — це вже існує, не переізобретати

Перевірено напряму в `Sources/`:

| Заявлено як "відсутнє" | Реально є |
|---|---|
| Pipeline (⭐⭐⭐⭐⭐ "найважливіша відсутня частина") | `SwiftPreprocessing/Core/Pipeline.swift` — **але** тільки для ланцюжка `PreprocessingTransformer`, не може завершуватись `ClassifierEstimator`/`RegressorEstimator` (див. п.1) |
| LabelEncoder, OrdinalEncoder, OneHotTextEncoder | `LabelEncoder`, `OrdinalEncoder`, `OneHotEncoder` |
| PolynomialFeatures | вже є |
| BinningTransformer | `KBinsDiscretizer` |
| MissingIndicator | `MissingValueIndicator`, `Imputer` |
| SHAP (Explainability, ⭐⭐⭐⭐⭐) | `SwiftExplain/Core/KernelSHAP.swift` вже реалізований |
| CountVectorizer / TFIDFVectorizer (частково) | `SwiftNLP/Core/TFIDFVectorizer.swift`, `Tokenizer.swift`, `BPETokenizer.swift`, `WordEmbeddings.swift` вже є |
| API на кшталт sklearn (`.fit()/.predict()/.transform()`, протоколи) | `SwiftML/Core/EstimatorProtocols.swift` (`ClassifierEstimator`, `RegressorEstimator`) і `SwiftPreprocessing/Core/PreprocessingTransformer.swift` (`fit/transform/fitTransform`) вже існують |
| GridSearch | `SwiftOptimize/GridSearchCV.swift` + `KFold.swift` вже є |
| Metrics (частково) | `SwiftOptimize/Metrics.swift` вже має accuracy/precision/recall/f1/classificationReport/MSE/RMSE/MAE/R² |

Справді відсутнє (нижче по пріоритету зі скорегованого списку): ColumnTransformer, RandomizedSearchCV+, розширені Metrics (ROC/PR/MCC/LogLoss/BrierScore/CohenKappa), Outlier Detection, Imbalanced Learning, Probability Calibration, Feature Selection, частина Text Preprocessing (CountVectorizer/HashingVectorizer/NGram/Stemmer/Lemmatizer/StopWords), частина Feature Engineering (Interaction/Date/Time/Hasher/VarianceThreshold), Time Series transformers (Lag/Rolling/Expanding/Holiday), Dataset Utilities, Model Persistence.

---

## 1. Pipeline ↔ Estimator unification — найвищий пріоритет, розблоковує решту

**Проблема**: `Pipeline` chains тільки `PreprocessingTransformer` (`fit(_ data: [[Double]]) throws`, sync). `ClassifierEstimator`/`RegressorEstimator` мають іншу форму (`fit(features:targets:) async throws`). Тому зараз неможливо `Pipeline { StandardScaler(); DecisionTree() }` — модель на виході pipeline не стикується.

**Файл**: новий `Sources/SwiftPreprocessing/Core/EstimatorPipeline.swift` (або в SwiftML, якщо циклічна залежність з SwiftPreprocessing небажана)

```swift
public final class ClassificationPipeline: ClassifierEstimator {
    public let steps: [any PreprocessingTransformer]
    public let finalEstimator: any ClassifierEstimator

    public init(steps: [any PreprocessingTransformer], finalEstimator: any ClassifierEstimator) {
        self.steps = steps
        self.finalEstimator = finalEstimator
    }

    public func fit(features: [[Double]], targets: [Double]) async throws {
        var current = features
        for step in steps {
            try step.fit(current)
            current = try step.transform(current)
        }
        try await finalEstimator.fit(features: current, targets: targets)
    }

    public func predict(features: [[Double]]) async throws -> [Int] {
        var current = features
        for step in steps { current = try step.transform(current) }
        return try await finalEstimator.predict(features: current)
    }
}
```

Дзеркально — `RegressionPipeline` над `RegressorEstimator`, ідентична структура з `[Double]` замість `[Int]`.

**Наслідок**: це вирішує пункт "Pipeline" з оригінального списку значно точніше, ніж "написати Pipeline з нуля" — проблема була в стикуванні, не у відсутності самого класу.

---

## 2. ColumnTransformer

**Контекст**: `PreprocessingTransformer` працює над `[[Double]]`, не над `DataFrame` з іменованими колонками — маршрутизація підмножин колонок до різних трансформерів зараз не існує на цьому рівні. Перед реалізацією перевір `Sources/SwiftPreprocessing/Core/DataFrame+Preprocessing.swift` — там уже може бути логіка DataFrame↔`[[Double]]` матеріалізації, яку варто перевикористати, а не дублювати.

Цільовий контракт:

```swift
public struct ColumnTransformer: PreprocessingTransformer {
    public struct Route {
        public let columns: [String]
        public let transformer: any PreprocessingTransformer
    }
    public init(routes: [Route], dataFrame: DataFrame) { ... }
    // fit/transform: витягнути підмножину колонок через існуючий DataFrame-міст,
    // прогнати кожен route окремо, конкатенувати результати по стовпцях назад.
}
```

---

## 3. Model Selection: RandomizedSearchCV

**Файл**: `Sources/SwiftOptimize/RandomizedSearchCV.swift`, поруч з `GridSearchCV.swift`
Семпли параметрів випадково з заданих розподілів замість повного перебору; перевикористати `KFold` для CV-розбиття. Bayesian/SuccessiveHalving — суттєво більший обсяг, відклади до v1.4+.

---

## 4. Metrics: розширення `SwiftOptimize/Metrics.swift`

Додати до існуючого `enum Metrics`: `balancedAccuracy`, `matthewsCorrelationCoefficient`, `cohenKappa`, `logLoss(yTrue: [Int], yProbPred: [Double])`, `brierScore`, `rocCurve(yTrue:yScore:) -> [(fpr: Double, tpr: Double, threshold: Double)]`, `rocAUC`, `prCurve`.

**Блокер**: ROC/PR/LogLoss/BrierScore потребують ймовірностей, а не лише класів — `ClassifierEstimator.predict` зараз повертає тільки `[Int]`. Потрібно спершу додати `predictProbability(features:) async throws -> [[Double]]` до `ClassifierEstimator` (з дефолтною реалізацією через протокол-екстеншн, що кидає `.notSupported`, щоб не ламати існуючі конформанси на кшталт `DecisionTree`/`RandomForest`).

---

## 5. Outlier Detection — новий набір у SwiftCluster

`IsolationForest`, `LocalOutlierFactor`, `EllipticEnvelope`. Це не швидкий "add" — Isolation Forest (випадкові дерева розбиття), LOF (локальна густина), EllipticEnvelope (робастна коваріація через Minimum Covariance Determinant) — реалізовувати обережно за формальним описом алгоритму, не на пам'ять.

---

## 6. Imbalanced Learning

Новий модуль `SwiftImbalance` або підмодуль `SwiftPreprocessing`: SMOTE (kNN-based синтетичний oversampling), ADASYN, RandomOversampler/RandomUndersampler. Та сама пересторога щодо коректності алгоритму, що й п.5.

---

## 7. Probability Calibration

`CalibratedClassifier` — обгортка над будь-яким `ClassifierEstimator` (Platt scaling / isotonic regression). **Залежить від п.4** (`predictProbability`) — робити після нього, не паралельно.

---

## 8. Feature Selection

`SelectKBest` — перевикористати статистичні тести з `SwiftStats`, а не переписувати кореляцію/ANOVA з нуля. `RecursiveFeatureElimination` — перевір, чи `DecisionTree`/`RandomForest` вже експонують feature importances внутрішньо, перш ніж додавати нову плумбінг-логіку для цього.

---

## 9. Text Preprocessing: залишок

`CountVectorizer`, `HashingVectorizer`, `NGramTokenizer`, `StopWords`, `Stemmer`/`Lemmatizer` у `SwiftNLP` (TF-IDF/Tokenizer/BPE вже є). Stemmer/Lemmatizer — уточнити цільову(і) мову(и) перед реалізацією: логіка для англійської й української суттєво різна.

---

## 10. Feature Engineering: залишок

`InteractionFeatures`, `DateFeatures`, `TimeFeatures`, `FeatureHasher`, `VarianceThreshold` у `SwiftPreprocessing`, як `PolynomialFeatures` — приймають `PreprocessingTransformer`.

---

## 11. Time Series transformers

`LagTransformer`, `RollingWindow`, `ExpandingWindow` у `SwiftForecast`. **Перевір спершу** `TimeSeriesDecomposition.swift` — частина "SeasonalityDetector" може вже частково покриватись ним. `HolidayEncoder` — визначитись, які календарі/локалі підтримувати, перш ніж починати.

---

## 12. Dataset Utilities

`makeRegression/makeClassification/makeClusters/makeMoons/makeCircles` — синтетичні генератори для тестів. Низький ризик, самодостатні. Кандидат на окремий testing-utilities target замість основної бібліотеки, щоб не роздувати production-поверхню API.

---

## 13. Model Persistence

`save`/`load` — Codable/JSON найпростіший шлях для Swift. Перевір, чи внутрішній стан `DecisionTree`/`RandomForest`/`LinearRegression`/`LogisticRegression` вже структурований достатньо для серіалізації — може знадобитись рефактор сховища параметрів перед цим.

---

## 14. Visualization — питання дизайну, не задача

Saura (SwiftUI-застосунок) — природне місце для рендеру через нативний Swift Charts. Рекомендація: **не** будувати plotting-рушій усередині бібліотеки; натомість SwiftSci віддає прості структури даних (`ConfusionMatrix`, точки ROC-кривої тощо), які Saura візуалізує сама. Дублювати те, що Apple вже дає нативно, немає сенсу.

---

## Release Notes — v1.3.0

```markdown
## v1.3.0

### Added
- `ClassificationPipeline` / `RegressionPipeline`: Pipeline can now terminate in a `ClassifierEstimator`/`RegressorEstimator`, not just preprocessing steps.
- `ColumnTransformer`: route DataFrame column subsets to different transformers.
- `RandomizedSearchCV` alongside existing `GridSearchCV`.
- `Metrics`: balancedAccuracy, MCC, Cohen's Kappa, LogLoss, BrierScore, ROC/PR curves + AUC.
- `ClassifierEstimator.predictProbability(features:)` (default throws `.notSupported` for existing conformances).
- Outlier detection: IsolationForest, LocalOutlierFactor, EllipticEnvelope (SwiftCluster).
- Imbalanced learning: SMOTE, ADASYN, random over/undersampling.
- `CalibratedClassifier` (Platt scaling / isotonic regression).
- Feature selection: SelectKBest, RecursiveFeatureElimination.
- Text preprocessing: CountVectorizer, HashingVectorizer, NGramTokenizer, StopWords, Stemmer/Lemmatizer.
- Feature engineering: InteractionFeatures, DateFeatures, TimeFeatures, FeatureHasher, VarianceThreshold.
- Time series: LagTransformer, RollingWindow, ExpandingWindow, HolidayEncoder.
- Dataset utilities (separate testing-utilities target): makeRegression/makeClassification/makeClusters/makeMoons/makeCircles.
- Model persistence: Codable-based save/load for SwiftML estimators.
```
