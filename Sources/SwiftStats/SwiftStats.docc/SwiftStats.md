# ``SwiftStats``

Vectorized Statistical Computing & Inferential Analytics.

## Overview

`SwiftStats` provides high-performance statistical primitives and hypothesis testing functions powered by Accelerate vDSP hardware acceleration.

### Key Capabilities

- **Vectorized Reductions**: Accelerated mean, variance, standard deviation, skewness, kurtosis, and quantiles.
- **Probability Distributions**: Student-t, Chi-Square, F-distribution, and Normal cumulative density functions.
- **Hypothesis Testing**: Two-sample Student's t-test, Paired t-test, Welch's t-test, and One-Way ANOVA.
- **Survival Analysis**: Kaplan-Meier survival curve estimation and Cox Proportional Hazards regression.

### Example Usage

```swift
import SwiftStats

let data: [Double] = [12.5, 14.2, 15.8, 11.0, 16.3]
let mean = data.mean()
let tTest = Stats.ttest(groupA, groupB)
```

## Topics

### Guides & Tutorials
- <doc:DescriptiveStatistics>
- <doc:HypothesisTesting>
