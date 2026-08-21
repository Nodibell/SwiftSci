# Non-Parametric Hypothesis Testing & Rank Correlation

Perform statistical hypothesis tests without distributional assumptions using accelerated rank routines.

## Overview

When dataset distributions violate normality assumptions, non-parametric statistics provide distribution-free tests based on ranks and order statistics.

## 1. Supported Non-Parametric Tests

| Test | Parametric Counterpart | Null Hypothesis ($H_0$) |
| :--- | :--- | :--- |
| **Mann-Whitney U** | Independent Two-Sample t-test | Distributions of both groups are equal |
| **Wilcoxon Signed-Rank** | Paired Sample t-test | Median difference between pairs is zero |
| **Kruskal-Wallis** | One-Way ANOVA | All group medians are equal |
| **Spearman Rank ($\\rho$)** | Pearson Correlation ($r$) | No monotonic relationship between variables |

## 2. Example: Mann-Whitney U Test

```swift
import SwiftStats

let groupA = [12.4, 15.2, 18.1, 14.9, 16.5, 17.0]
let groupB = [22.1, 24.5, 19.8, 26.2, 21.0, 25.3]

let testResult = SwiftStats.mannWhitneyUTest(groupA, groupB)
print("U-Statistic: \(testResult.statistic), p-value: \(testResult.pValue)")
```

## Topics

### Non-Parametric Functions
- ``SwiftStats/mannWhitneyUTest(_:_:)``
