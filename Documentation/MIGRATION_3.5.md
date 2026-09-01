# SwiftSci 3.5 Migration Guide

SwiftSci version 3.5 introduces substantial performance optimizations, architectural hardening, and Swift 6 concurrency safety across all 15 modules. This guide covers breaking changes, key enhancements, and steps to upgrade existing codebases from SwiftSci 3.4 to 3.5.

---

## 1. Concurrency & Actor Isolation (Swift 6 Strict Concurrency)

### Database Connections
In SwiftSci 3.4, database connection classes (`SQLiteConnection`, `PostgreSQLConnection`, `MySQLConnection`) used `@unchecked Sendable` without guaranteed synchronization. In 3.5, all database connection classes are now **`public actor`**:

```swift
// 3.4: Synchronous / unchecked call
let db = SQLiteConnection(path: "analytics.db")
try db.connect()
let results = try db.executeQuery("SELECT * FROM users")

// 3.5: Actor-isolated async calls
let db = SQLiteConnection(path: "analytics.db")
try await db.connect()
let results = try await db.executeQuery("SELECT * FROM users")
```

### Machine Learning Estimators & Transformers
Estimators such as `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`, `MLPClassifier`, `MLPRegressor`, and `TFIDFVectorizer` are now `public actor` types for complete Swift 6 thread safety:

```swift
// 3.5 Actor API:
let tree = DecisionTreeClassifier(maxDepth: 5)
try await tree.fit(features: X, targets: y)
let predictions = try await tree.predict(features: testX)
```

---

## 2. Flat Memory Layout & Apple Accelerate SIMD Vectorization

### Preprocessing Scalers
`StandardScaler`, `MinMaxScaler`, and `RobustScaler` now operate natively on contiguous flat memory buffers `[Double]` (and `[Float]`) row-major representations accelerated with Apple Accelerate `vDSP`:

```swift
// 3.4:
let scaled = scaler.transform(matrixOfArrays) // [[Double]]

// 3.5:
let scaled = scaler.transform(flatArray, rows: numRows, cols: numCols) // [Double]
```

### Kalman Filter
`KalmanFilter` has been rewritten from pointer-based 2D arrays to 1D flat buffers backed by LAPACK (`dgesv`) and BLAS (`cblas_dgemm`), reducing memory allocations by >90% on streaming sensor inputs.

---

## 3. Machine Learning & Spatial Indexing

### DBSCAN with KD-Tree Spatial Indexing ($O(N \log N)$)
`DBSCAN` now leverages `KDTree` spatial partitioning for range searches. Clustering benchmarks scale efficiently to $100,000+$ points without $O(N^2)$ quadratic slowdowns:

```swift
let dbscan = DBSCAN(eps: 0.5, minSamples: 5)
let labels = dbscan.fit(points) // O(N log N) via KDTree
```

### Exact TreeSHAP Explainability
`TreeSHAP` now computes exact polynomial Shapley values in $O(T \cdot L \cdot D^2)$ time directly on flat tree structures:

```swift
let explainer = TreeSHAP()
let shapValues = await explainer.explain(decisionTree: tree, instance: sample)
```

### Sequential Model Selection in AutoML
`AutoML` performs real sequential 3-fold cross validation over candidate classifiers and regressors, selecting the optimal configuration based on $F_1$-score or $R^2$.

---

## 4. Natural Language Processing & Visualization

### Sparse TF-IDF Representations
`TFIDFVectorizer` now provides memory-efficient sparse vector transformations via `transformSparse`:

```swift
let vectorizer = TFIDFVectorizer(minDF: 2)
try await vectorizer.fit(documents)
let sparseMatrix: [SparseVector] = try await vectorizer.transformSparse(documents)
```

### Native SwiftUI Multi-Series Canvas Charts
`SwiftSciChartView` supports line charts, bar charts, scatter plots, histograms, and 2D heatmaps with full `ChartOptions` customization:

```swift
let chart = SwiftSciChartView(
    title: "Distribution",
    type: .histogram,
    series: [series],
    options: ChartOptions(showAxes: true, barWidthRatio: 0.85)
)
```
