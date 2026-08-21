# Continuous & Discrete Probability Distributions

Compute probability density (PDF), cumulative distribution (CDF), and quantiles across statistical distributions using Apple Accelerate vForce.

## Overview

`SwiftStats` provides mathematically rigorous probability distributions essential for statistical modeling, Monte Carlo simulations, and hypothesis testing on Apple Silicon.

## 1. Supported Distributions

| Distribution | Type | Primary Applications |
| :--- | :--- | :--- |
| **Normal (Gaussian)** | Continuous | Central Limit Theorem, error modeling, z-tests |
| **Student's t** | Continuous | Small-sample inference, t-tests, regression coefficients |
| **Chi-Squared ($\\chi^2$)** | Continuous | Goodness-of-fit, contingency tables, variance testing |
| **F-Distribution** | Continuous | ANOVA, regression variance ratio tests |
| **Poisson** | Discrete | Arrival rates, count data modeling |
| **Binomial** | Discrete | Success/failure trials, A/B testing |

## 2. Normal Distribution Example

```swift
import SwiftStats

let normal = NormalDistribution(mean: 0.0, standardDeviation: 1.0)

// Probability Density Function (PDF)
let pdfVal = normal.pdf(x: 1.96) // ~0.0584

// Cumulative Distribution Function (CDF)
let cdfVal = normal.cdf(x: 1.96) // ~0.9750

// Quantile / Inverse CDF (Percent Point Function)
let zCritical = normal.quantile(p: 0.95) // ~1.6449
```

## 3. Student's t-Distribution & Confidence Intervals

```swift
let tDist = StudentTDistribution(degreesOfFreedom: 29)
let tCrit = tDist.quantile(p: 0.975) // Critical value for 95% two-tailed test
```

## Topics

### Distribution Types
- ``NormalDistribution``
- ``StudentTDistribution``
- ``ChiSquaredDistribution``
- ``FDistribution``
