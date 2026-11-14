# Hypothesis Testing & ANOVA

Perform Student's/Welch's t-tests, paired t-tests, Chi-Square goodness-of-fit tests, normality tests, and one-way ANOVA.

## Overview

`SwiftStats` supports complete inferential statistics with numerical stability across extreme sample offsets, large degrees of freedom (`df >= 2e5` via Temme uniform asymptotic expansion), and sub-machine-epsilon upper tails.

### 1. Two-Sample Welch / Student t-Test

```swift
import SwiftStats

let groupA: [Double] = [12.1, 14.3, 11.8, 13.5, 15.0]
let groupB: [Double] = [9.8, 10.5, 11.2, 10.1, 9.4]

let result = try Stats.tTest(sample1: groupA, sample2: groupB)
print("t-statistic: \(result.statistic), p-value: \(result.pValue)")
if result.isSignificant {
    print("Statistically significant difference (p < 0.05)")
}
```

### 2. One-Way ANOVA

```swift
let samples: [[Double]] = [groupA, groupB, [8.5, 9.0, 8.8]]
let anova = try Stats.oneWayANOVA(groups: samples)
print("F-statistic: \(anova.fStatistic), p-value: \(anova.pValue)")
```

### 3. Chi-Square Goodness-of-Fit

`Stats.chiSquareGoodnessOfFit(observed:expected:)` requires at least two categories, finite non-negative observed frequencies, and strictly positive finite expected frequencies. Upper-tail survival probabilities use direct Lentz continued fractions and Temme asymptotics (`a = df / 2 >= 1e5`).

```swift
let chi2 = try Stats.chiSquareGoodnessOfFit(
    observed: [20, 30, 50],
    expected: [25, 25, 50]
)
print("chi2 = \(chi2.statistic), p = \(chi2.pValue), df = \(chi2.degreesOfFreedom)")
```

### 4. Normality Tests (Shapiro–Wilk & Lilliefors KS)

- `Stats.shapiroWilk(_:)` implements Royston AS R94 for `3 <= n <= 5000` with shift/scale invariance (`1e12` offsets, subnormal to `1e308` dynamic range) and rejects non-finite or constant samples.
- `Stats.kolmogorovSmirnov(_:)` implements the Lilliefors normality test against sample-fitted normal parameters (`n >= 5`) using Dallal–Wilkinson and Stephens calibrated p-values.

```swift
let sample: [Double] = [2.1, 2.5, 2.7, 3.0, 3.1, 3.3, 3.8]
let sw = try Stats.shapiroWilk(sample)
let ks = try Stats.kolmogorovSmirnov(sample)
print("Shapiro-Wilk W = \(sw.statistic), p = \(sw.pValue)")
print("Lilliefors D = \(ks.statistic), p = \(ks.pValue)")
```
