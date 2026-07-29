# Hypothesis Testing & ANOVA

Perform Student's t-tests, paired t-tests, Chi-Square independence tests, and Analysis of Variance (ANOVA).

## Overview

`SwiftStats` supports complete inferential statistics with exact p-value and confidence interval calculations.

### 1. Two-Sample Student's t-Test

```swift
import SwiftStats

let groupA: [Double] = [12.1, 14.3, 11.8, 13.5, 15.0]
let groupB: [Double] = [9.8, 10.5, 11.2, 10.1, 9.4]

let result = Stats.ttest(groupA, groupB)
print("t-statistic: \(result.statistic), p-value: \(result.pValue)")
if result.isSignificant {
    print("Statistically significant difference (p < 0.05)")
}
```

### 2. One-Way ANOVA

```swift
let samples: [[Double]] = [groupA, groupB, [8.5, 9.0, 8.8]]
let anova = Stats.anova(samples)
print("F-statistic: \(anova.statistic), p-value: \(anova.pValue)")
```
