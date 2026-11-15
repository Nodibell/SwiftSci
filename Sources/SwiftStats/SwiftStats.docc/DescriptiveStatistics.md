# Vectorized Descriptive Statistics & Linear Algebra

Compute descriptive metrics, covariance, Pearson/Spearman correlations, and matrix operations using Apple Accelerate `vDSP` and `BLAS`.

## Overview

Statistical analysis on massive datasets requires hardware vectorization to eliminate CPU branch mispredictions and heap allocations. `SwiftStats` leverages Apple Accelerate (`vDSP_meanvD`, `vDSP_measqvD`, `vDSP.sort`) to compute descriptive statistics in a single vectorized pass.

---

## 1. Vectorized Summary Statistics (`Double` & `Float`)

Compute mean, variance, standard deviation, median, and percentiles with dedicated `Float` and `Double` pathways:

```swift
import Foundation
import SwiftStats

// 1. Vector of Double precision observations
let observations: [Double] = [12.4, 15.8, 18.2, 19.5, 22.1, 25.0, 31.4, 45.2]

let mean = try Stats.mean(observations)
let variance = try Stats.variance(observations, ddof: 1) // Sample variance (ddof = 1)
let std = try Stats.standardDeviation(observations, ddof: 1)
let median = try Stats.median(observations)
let p90 = try Stats.percentile(observations, p: 90.0)

print("=== Descriptive Statistics Summary ===")
print("Mean: \(String(format: "%.2f", mean))")
print("Variance (s²): \(String(format: "%.2f", variance))")
print("Standard Deviation (s): \(String(format: "%.2f", std))")
print("Median (Q2): \(String(format: "%.2f", median))")
print("90th Percentile: \(String(format: "%.2f", p90))")

// 2. High-performance Float32 overload (50% memory savings)
let floatData: [Float] = [1.2, 2.4, 3.6, 4.8]
let floatMean = try Stats.mean(floatData)
let floatStd = try Stats.standardDeviation(floatData)
print("Float32 Mean: \(floatMean), Std: \(floatStd)")

// 3. Higher-order moments (skewness and excess kurtosis)
let momentsData = [1.0, 2.0, 3.0, 4.0, 5.0]
let skew = try Stats.skewness(momentsData)  // 0.0
let kurt = try Stats.kurtosis(momentsData)  // -1.912
```

> Note: In `SwiftSci v3.13.x`, `Stats.skewness` and `Stats.kurtosis` normalize the $1/n$ central moment numerator by the sample standard deviation `s` computed with `ddof = 1` ($\frac{m_3}{s^3}$ and $\frac{m_4}{s^4} - 3$), rather than the population-moment Fisher formula ($m_3 / m_2^{3/2}$ and $m_4 / m_2^2 - 3$). This default is preserved in `v3.13.2` for API compatibility.

---

## 2. Correlation & Covariance Matrices

Calculate Pearson linear correlation ($r$) and Spearman rank correlation ($\rho$):

```swift
let x = [1.0, 2.0, 3.0, 4.0, 5.0]
let y = [2.1, 3.9, 6.2, 8.1, 9.9]

let pearson = try Stats.pearsonCorrelation(x, y)
let spearman = try Stats.spearmanCorrelation(x, y)

print("Pearson Correlation (r): \(String(format: "%.4f", pearson))")
print("Spearman Rank Correlation (rho): \(String(format: "%.4f", spearman))")
```

---

## 3. Parametric & Non-Parametric Hypothesis Testing

Execute formal statistical hypothesis tests:

```swift
// Two-sample Independent Student's t-test
let groupA = [22.5, 24.1, 23.8, 25.0, 21.9]
let groupB = [28.2, 29.5, 27.8, 31.0, 30.2]

let tResult = try Stats.independentTTest(groupA, groupB)
print("t-Statistic: \(String(format: "%.3f", tResult.statistic)), p-value: \(String(format: "%.5f", tResult.pValue))")
```
