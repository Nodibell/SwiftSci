# Vectorized Descriptive Statistics

Learn how to compute summary statistics, variance, standard deviation, skewness, and quantiles using Accelerate vDSP acceleration.

## Overview

`SwiftStats` provides high-performance statistical primitives optimized for Apple Silicon UMA and vectorized hardware.

### 1. Vectorized Reductions

```swift
import SwiftStats

let data: [Double] = [1.5, 2.7, 3.1, 4.8, 5.2, 6.0, 7.4]

let mean = data.mean()
let stdDev = data.stdDev()
let variance = data.variance()

print("Mean: \(mean), StdDev: \(stdDev), Variance: \(variance)")
```

### 2. Quantiles and Percentiles

```swift
let median = data.median()
let p75 = data.percentile(75)

print("Median: \(median), 75th Percentile: \(p75)")
```
