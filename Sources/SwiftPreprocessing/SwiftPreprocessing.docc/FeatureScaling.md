# Feature Scaling, Normalization & Flat 1D Memory

Scale, standardize, and normalize feature matrices at memory bandwidth speeds using `StandardScaler`, `MinMaxScaler`, `RobustScaler`, and `Normalizer` powered by Apple Accelerate `vDSP`.

## Overview

Feature scaling is crucial for gradient-based estimators (Neural Networks, Logistic Regression) and distance-based estimators (KNN, DBSCAN, SVM). Without scaling, features with large numeric scales disproportionately dominate distance calculations and gradient updates.

---

## ⚡ Flat 1D Memory Layout & Hardware Vectorization

`SwiftPreprocessing` operates on contiguous 1D row-major flat buffers `[Double]` (or `[Float]`) with stride indexing `index = row * cols + col`:

```
Matrix (3 rows x 2 columns)
Row 0: [ 1.0, 200.0 ]
Row 1: [ 2.0, 300.0 ]
Row 2: [ 3.0, 400.0 ]

Contiguous Flat Memory Buffer:
┌───────┬─────────┬───────┬─────────┬───────┬─────────┐
│  1.0  │  200.0  │  2.0  │  300.0  │  3.0  │  400.0  │
└───────┴─────────┴───────┴─────────┴───────┴─────────┘
 offset:0  offset:1  offset:2  offset:3  offset:4  offset:5
```

All arithmetic operations (`vDSP_vsubD`, `vDSP_vsdivD`, `vDSP.sort`) execute SIMD instructions directly across contiguous memory without heap pointer indirections.

---

## 1. `StandardScaler` (Z-Score Standardization)

Transforms features to have zero mean ($\mu = 0$) and unit variance ($\sigma^2 = 1$):
$$z = \frac{x - \mu}{\sigma}$$

```swift
import Foundation
import SwiftPreprocessing

// 1. Prepare flat dataset: 3 rows x 2 columns
let rows = 3
let cols = 2
let flatData: [Double] = [
    1.0, 200.0,
    2.0, 300.0,
    3.0, 400.0
]

// 2. Fit and transform
var scaler = StandardScaler()
try scaler.fit(data: flatData, rows: rows, cols: cols)

let scaled = try scaler.transform(data: flatData, rows: rows, cols: cols)

print("=== Standard Scaled Values ===")
for r in 0..<rows {
    let rowVals = (0..<cols).map { c in String(format: "%+.4f", scaled[r * cols + c]) }
    print("Row \(r): [\(rowVals.joined(separator: ", "))]")
}

// 3. Inverse transform back to original scale
let reconstructed = try scaler.inverseTransform(data: scaled, rows: rows, cols: cols)
print("Reconstructed matches original: \(reconstructed == flatData)")
```

---

## 2. `MinMaxScaler` (Bounded Range Scaling)

Linearly transforms features into a bounded range $[a, b]$ (defaults to $[0, 1]$):
$$x' = a + \frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot (b - a)$$

```swift
import SwiftPreprocessing

var minMax = MinMaxScaler(featureRange: (0.0, 1.0))
let normalized = try minMax.fitTransform(data: flatData, rows: rows, cols: cols)

print("Min-Max scaled to [0, 1]: \(normalized)")
```

---

## 3. `RobustScaler` (Median & IQR Scaling)

When datasets contain extreme outliers, mean and standard deviation are distorted. `RobustScaler` scales features using the **Median** and **Interquartile Range (IQR = Q3 - Q1)**:
$$x' = \frac{x - \text{median}}{\text{IQR}}$$

```swift
import SwiftPreprocessing

// Dataset with an extreme outlier in column 1
let outlierData: [Double] = [
    1.0, 10.0,
    2.0, 11.0,
    2.1, 12.0,
    2.2, 11.5,
    3.0, 1000.0 // extreme outlier
]

var robust = RobustScaler()
let robustScaled = try robust.fitTransform(data: outlierData, rows: 5, cols: 2)

print("Robust scaled IQR data: \(robustScaled)")
```

---

## 4. `Normalizer` (Row-wise Unit Norm Vector Scaling)

Scales each sample independently to have unit $L_1$, $L_2$, or $\max$ norm:

```swift
import SwiftPreprocessing

var normalizer = Normalizer(norm: .l2)
let l2Normalized = try normalizer.fitTransform(data: flatData, rows: rows, cols: cols)
```

> **Memory & Performance Tip:**
> Use `Float` overloads for deep learning image datasets and embeddings to halve RAM usage and maximize Neural Engine throughput.
