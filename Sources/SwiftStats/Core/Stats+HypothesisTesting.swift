import Accelerate
import Foundation
import SwiftDataFrame

// MARK: – Hypothesis Testing

extension Stats {

    // MARK: One-sample t-test

    /// Tests whether the sample mean equals `mu`.
    /// - Parameters:
    ///   - sample: Sample observation array.
    ///   - mu: Hypothesized population mean scalar.
    /// - Throws: `StatsError` if input is empty, contains NaNs when forbidden, or dimensions mismatch.
    /// - Returns: `TTestResult` structure containing t-statistic, p-value, and degrees of freedom.
    public static func tTest(sample: [Double], populationMean mu: Double) throws -> TTestResult {
        try requireNonEmpty(sample, minimum: 2)
        try requireNoNaN(sample)

        let n    = Double(sample.count)
        let xBar = vDSP.mean(sample)
        let s    = try standardDeviation(sample, ddof: 1)
        guard s > 0 else { throw StatsError.divisionByZero(context: "one-sample tTest") }

        let se = s / n.squareRoot()
        let t  = (xBar - mu) / se
        let df = n - 1

        let pValue = 2.0 * tDistributionCDF(-abs(t), df: df)
        let margin = criticalT(df: df, alpha: 0.05) * se
        let d      = (xBar - mu) / s

        return TTestResult(
            statistic:          t,
            pValue:             pValue,
            degreesOfFreedom:   df,
            confidenceInterval: ConfidenceInterval(lower: xBar - margin, upper: xBar + margin, confidence: 0.95),
            effectSize:         d
        )
    }

    // MARK: Two-sample t-test (Welch by default)

    /// Two-sample t-test.
    /// - Parameters:
    ///   - sample1: First numeric sample array.
    ///   - sample2: Second numeric sample array.
    ///   - equalVariances: If true, uses pooled variance (Student). Default false (Welch).
    /// - Throws: `StatsError.emptyInput` if input is insufficient or contains NaN.
    /// - Returns: `TTestResult` struct containing t-statistic, degrees of freedom, and p-value.
    public static func tTest(sample1: [Double], sample2: [Double],
                             equalVariances: Bool = false) throws -> TTestResult {
        try requireNonEmpty(sample1, minimum: 2)
        try requireNonEmpty(sample2, minimum: 2)
        try requireNoNaN(sample1)
        try requireNoNaN(sample2)

        let n1   = Double(sample1.count)
        let n2   = Double(sample2.count)
        let x1   = vDSP.mean(sample1)
        let x2   = vDSP.mean(sample2)
        let var1 = try variance(sample1, ddof: 1)
        let var2 = try variance(sample2, ddof: 1)

        let t: Double
        let df: Double
        let se: Double

        if equalVariances {
            // Student's t (pooled variance)
            let sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            guard sp2 > 0 else { throw StatsError.divisionByZero(context: "pooled tTest") }
            se = (sp2 * (1/n1 + 1/n2)).squareRoot()
            t  = (x1 - x2) / se
            df = n1 + n2 - 2
        } else {
            // Welch's t-test
            let se1 = var1 / n1
            let se2 = var2 / n2
            se = (se1 + se2).squareRoot()
            guard se > 0 else { throw StatsError.divisionByZero(context: "Welch tTest") }
            t  = (x1 - x2) / se
            // Welch-Satterthwaite degrees of freedom
            df = pow(se1 + se2, 2) / (pow(se1, 2)/(n1-1) + pow(se2, 2)/(n2-1))
        }

        let pValue = 2.0 * tDistributionCDF(-abs(t), df: df)
        let margin = criticalT(df: df, alpha: 0.05) * se
        let d      = (x1 - x2) / (try pooledSD(sample1, sample2))

        return TTestResult(
            statistic:          t,
            pValue:             pValue,
            degreesOfFreedom:   df,
            confidenceInterval: ConfidenceInterval(lower: (x1-x2) - margin, upper: (x1-x2) + margin, confidence: 0.95),
            effectSize:         d
        )
    }

    // MARK: Paired t-test

    /// Tests whether the mean difference between paired samples equals zero.
    /// - Parameters:
    ///   - before: Sample observations prior to treatment or intervention.
    ///   - after: Sample observations following treatment or intervention.
    /// - Throws: `StatsError` if input is empty, contains NaNs when forbidden, or dimensions mismatch.
    /// - Returns: `TTestResult` structure containing t-statistic, p-value, and degrees of freedom.
    public static func pairedTTest(before: [Double], after: [Double]) throws -> TTestResult {
        try requireSameSize(before, after)
        // zip subtraction: diffs[i] = after[i] - before[i]
        let diffs = zip(before, after).map { $1 - $0 }
        return try tTest(sample: diffs, populationMean: 0)
    }

    // MARK: One-way ANOVA

    /// Tests whether the means of 2+ groups are equal.
    /// - Parameters:
    ///   - groups: Array of sample observation groups for comparison.
    /// - Throws: `StatsError` if input is empty, contains NaNs when forbidden, or dimensions mismatch.
    /// - Returns: ANOVA test result containing F-statistic, p-value, and group variance metrics.
    public static func oneWayANOVA(groups: [[Double]]) throws -> ANOVAResult {
        guard groups.count >= 2 else {
            throw StatsError.invalidGroupCount(minimum: 2, got: groups.count)
        }
        for g in groups { try requireNonEmpty(g, minimum: 1); try requireNoNaN(g) }

        let k = groups.count
        let n = groups.reduce(0) { $0 + $1.count }  // total observations

        func compensatedSum<S: Sequence>(_ values: S) -> Double where S.Element == Double {
            var sum = 0.0
            var correction = 0.0
            for value in values {
                let next = sum + value
                correction += abs(sum) >= abs(value) ? (sum - next) + value : (value - next) + sum
                sum = next
            }
            return sum + correction
        }

        // Keep means relative to an observed value so small differences survive large offsets.
        let origin = groups[0][0]
        let means = groups.map { group in
            compensatedSum(group.lazy.map { $0 - origin }) / Double(group.count)
        }
        let grandMean = compensatedSum(zip(groups, means).lazy.map {
            Double($0.0.count) * $0.1
        }) / Double(n)
        let ssBetween = compensatedSum(zip(groups, means).lazy.map { group, mean in
            let difference = mean - grandMean
            return Double(group.count) * difference * difference
        })
        let ssWithin = compensatedSum(groups.lazy.map { group in
            if let centeredSquares = CenteredMoments.sumOfSquares(group) {
                return centeredSquares
            }
            let groupOrigin = group[0]
            let groupMean = compensatedSum(group.lazy.map { $0 - groupOrigin }) / Double(group.count)
            return compensatedSum(group.lazy.map {
                let difference = ($0 - groupOrigin) - groupMean
                return difference * difference
            })
        })

        let dfBetween = k - 1
        let dfWithin  = n - k
        let msBetween = ssBetween / Double(dfBetween)
        let msWithin  = ssWithin  / Double(dfWithin)

        guard msWithin > 0 else { throw StatsError.divisionByZero(context: "oneWayANOVA") }
        let f       = msBetween / msWithin
        let pValue  = fDistributionSurvival(f: f, d1: Double(dfBetween), d2: Double(dfWithin))
        let eta2    = ssBetween / (ssBetween + ssWithin)

        return ANOVAResult(
            fStatistic: f,
            pValue:     pValue,
            dfBetween:  dfBetween,
            dfWithin:   dfWithin,
            etaSquared: eta2
        )
    }

    // MARK: Chi-square Goodness of Fit

    /// Tests whether observed frequencies match expected frequencies.
    /// - Parameters:
    ///   - observed: Array of observed sample frequencies.
    ///   - expected: Array of expected theoretical frequencies.
    /// - Throws: `StatsError` if input is empty, contains NaNs when forbidden, or dimensions mismatch.
    /// - Returns: `ChiSquareResult` containing chi-squared test statistic, p-value, and degrees of freedom.
    public static func chiSquareGoodnessOfFit(observed: [Double],
                                               expected: [Double]) throws -> ChiSquareResult {
        try requireNonEmpty(observed, minimum: 2)
        try requireSameSize(observed, expected)
        try requireNoNaN(observed)
        try requireNoNaN(expected)

        var chi2 = 0.0
        for (o, e) in zip(observed, expected) {
            guard o.isFinite, e.isFinite else { throw StatsError.containsInfinity }
            guard o >= 0 else {
                throw StatsError.invalidInput("chiSquare observed frequency must be non-negative")
            }
            guard e > 0 else { throw StatsError.divisionByZero(context: "chiSquare expected=0") }
            let diff = o - e
            chi2 += diff * (diff / e)
        }

        let df     = observed.count - 1
        let pValue = chiSquareSurvival(chi2: chi2, df: Double(df))

        return ChiSquareResult(statistic: chi2, pValue: pValue, degreesOfFreedom: df)
    }

    // MARK: Normality Tests

    /// Shapiro-Wilk normality test (Royston AS R94, n ∈ [3, 5000]).
    /// Constant samples and nonfinite observations are rejected.
    /// - Parameters:
    ///   - values: Numeric values array to be evaluated.
    /// - Throws: `StatsError` if input is empty, contains NaNs when forbidden, or dimensions mismatch.
    /// - Returns: Normality test diagnostic result evaluating distribution adherence.
    public static func shapiroWilk(_ values: [Double]) throws -> NormalityTestResult {
        try requireNonEmpty(values, minimum: 3)
        try requireNoNaN(values)
        guard values.count <= 5000 else {
            throw StatsError.tooManySamples(limit: 5000, got: values.count)
        }
        let (w, p) = shapiroWilkImpl(try scaledNormalitySample(values))
        return NormalityTestResult(statistic: w, pValue: p)
    }

    /// Lilliefors normality test: KS distance from a normal distribution fitted to the sample.
    /// Uses Dallal-Wilkinson's lower-tail approximation and Stephens' modified-statistic
    /// approximation above p = 0.1. These are approximate p-values, not fixed-CDF KS values.
    /// Requires at least five finite, nonconstant observations.
    public static func kolmogorovSmirnov(_ values: [Double]) throws -> NormalityTestResult {
        try requireNonEmpty(values, minimum: 5)
        try requireNoNaN(values)
        let sorted = try scaledNormalitySample(values)
        let mu = vDSP.mean(sorted)
        let s = try standardDeviation(sorted, ddof: 1)

        let n        = Double(sorted.count)
        var dPlus    = 0.0
        var dMinus   = 0.0

        for (i, x) in sorted.enumerated() {
            let z    = (x - mu) / s
            let cdf  = normalCDF(z)
            let fi   = Double(i + 1) / n
            let fim1 = Double(i) / n
            dPlus  = Swift.max(dPlus,  fi - cdf)
            dMinus = Swift.max(dMinus, cdf - fim1)
        }

        let d      = Swift.max(dPlus, dMinus)
        let pValue = lillieforsPValue(d: d, n: Int(n))

        return NormalityTestResult(statistic: d, pValue: pValue)
    }

    // MARK: – Private numerical routines

    /// Student t CDF using regularised incomplete beta function.
    /// P(T ≤ t) where T ~ t(df).
    internal static func tDistributionCDF(_ t: Double, df: Double) -> Double {
        // Using: P(T ≤ t) = Ix(df/2, 1/2) / 2  where x = df/(df+t²), t ≤ 0
        // For t > 0: P(T ≤ t) = 1 - Ix(df/2, 1/2)/2
        let x = df / (df + t * t)
        let ib = regularisedIncompleteBeta(x: x, a: df/2, b: 0.5) / 2.0
        return t < 0 ? ib : 1.0 - ib
    }

    /// Two-tailed critical t value at given alpha.
    internal static func criticalT(df: Double, alpha: Double) -> Double {
        // Binary search on the t-CDF
        var lo = 0.0, hi = 100.0
        for _ in 0..<64 {
            let mid = (lo + hi) / 2
            let p   = 2.0 * (1.0 - tDistributionCDF(mid, df: df))
            if p > alpha { lo = mid } else { hi = mid }
        }
        return (lo + hi) / 2
    }

    /// F-distribution survival function P(F > f).
    internal static func fDistributionSurvival(f: Double, d1: Double, d2: Double) -> Double {
        guard f > 0 else { return 1.0 }
        let x = d2 / (d2 + d1 * f)
        return regularisedIncompleteBeta(x: x, a: d2/2, b: d1/2)
    }

    /// Chi-square survival function P(χ² > chi2).
    /// Uses Temme's uniform asymptotic expansion for large shape parameters (df >= 2e5)
    /// to avoid series iteration caps and lgamma cancellation, and Lentz CF in the upper tail.
    internal static func chiSquareSurvival(chi2: Double, df: Double) -> Double {
        guard df > 0, df.isFinite, !chi2.isNaN else { return .nan }
        guard chi2 > 0 else { return 1.0 }
        if chi2 == .infinity { return 0 }
        let a = df / 2, x = chi2 / 2
        if a >= 1e5 {
            return temmeIncompleteGamma(a: a, x: x).q
        }
        // Evaluate the upper tail directly to retain probabilities smaller than machine epsilon.
        return x < a + 1 ? 1 - regularisedIncompleteGamma(a: a, x: x)
                         : regularisedIncompleteGammaCF(a: a, x: x)
    }

    /// Dallal & Wilkinson (1986) / Stephens (1974) normal Lilliefors calibration.
    /// The polynomial coefficients approximate the distribution of the modified statistic.
    private static func lillieforsPValue(d: Double, n: Int) -> Double {
        let size = Double(n)
        let effectiveN = Swift.min(size, 100)
        let adjustedD = size > 100 ? d * pow(size / 100, 0.49) : d
        var p = exp(-7.01256 * adjustedD * adjustedD * (effectiveN + 2.78019)
            + 2.99587 * adjustedD * sqrt(effectiveN + 2.78019) - 0.122119
            + 0.974598 / sqrt(effectiveN) + 1.67997 / effectiveN)
        if p > 0.1 {
            let k = (sqrt(size) - 0.01 + 0.85 / sqrt(size)) * d
            switch k {
            case ...0.302: p = 1
            case ...0.5: p = poly([2.76773, -19.828315, 80.709644, -138.55152, 81.218052], k)
            case ...0.9: p = poly([-4.901232, 40.662806, -97.490286, 94.029866, -32.355711], k)
            case ...1.31: p = poly([6.198765, -19.558097, 23.186922, -12.234627, 2.423045], k)
            default: p = 0
            }
        }
        return Swift.max(0, Swift.min(1, p))
    }

    /// Pooled standard deviation for Cohen's d.
    private static func pooledSD(_ a: [Double], _ b: [Double]) throws -> Double {
        let na = Double(a.count), nb = Double(b.count)
        let va = try variance(a, ddof: 1)
        let vb = try variance(b, ddof: 1)
        return (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)).squareRoot()
    }

    // MARK: – Regularised Incomplete Beta (Numerical Recipes betacf)

    /// Regularised incomplete beta function Iₓ(a, b).
    /// Uses the Lentz continued fraction from Numerical Recipes (Press et al. 2007).
    internal static func regularisedIncompleteBeta(x: Double, a: Double, b: Double) -> Double {
        guard x > 0 && x < 1 else { return x <= 0 ? 0.0 : 1.0 }

        // Symmetry relation for faster convergence
        if x > (a + 1.0) / (a + b + 2.0) {
            return 1.0 - regularisedIncompleteBeta(x: 1.0 - x, a: b, b: a)
        }

        let lbeta = lgamma(a + b) - lgamma(a) - lgamma(b)
        let front = exp(a * log(x) + b * log(1.0 - x) + lbeta) / a

        return front * betacf(a: a, b: b, x: x)
    }

    /// Continued fraction evaluation for betai (Numerical Recipes).
    private static func betacf(a: Double, b: Double, x: Double) -> Double {
        let maxIter = 200
        let eps     = 3.0e-15
        let fpmin   = 1.0e-300

        let qab = a + b
        let qap = a + 1.0
        let qam = a - 1.0

        var c = 1.0
        var d = 1.0 - qab * x / qap
        if abs(d) < fpmin { d = fpmin }
        d = 1.0 / d
        var h = d

        for m in 1...maxIter {
            let dm  = Double(m)
            let m2  = 2.0 * dm

            // Even step
            var aa = dm * (b - dm) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin { d = fpmin }
            c = 1.0 + aa / c
            if abs(c) < fpmin { c = fpmin }
            d = 1.0 / d
            h *= d * c

            // Odd step
            aa = -(a + dm) * (qab + dm) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin { d = fpmin }
            c = 1.0 + aa / c
            if abs(c) < fpmin { c = fpmin }
            d = 1.0 / d
            let delta = d * c
            h *= delta

            if abs(delta - 1.0) <= eps { break }
        }
        return h
    }

    /// Regularised lower incomplete gamma P(a, x).
    /// Delegates `a >= 1e5` to Temme's uniform asymptotic expansion to eliminate
    /// series iteration truncation and `lgamma(a)` cancellation at large shapes.
    internal static func regularisedIncompleteGamma(a: Double, x: Double) -> Double {
        guard a > 0, a.isFinite, x >= 0 else { return .nan }
        if x == 0 { return 0 }
        if x == .infinity { return 1 }
        if a >= 1e5 {
            return temmeIncompleteGamma(a: a, x: x).p
        }

        // Series expansion for x < a + 1 (converges in < 2400 iterations for a < 1e5)
        if x < a + 1 {
            var term = 1.0 / a
            var sum  = term
            for n in 1...10000 {
                term *= x / (a + Double(n))
                sum  += term
                if abs(term) < 2e-15 * abs(sum) { break }
            }
            return sum * exp(-x + a * log(x) - lgamma(a))
        }

        // Continued fraction for x >= a + 1 (complement)
        return 1.0 - regularisedIncompleteGammaCF(a: a, x: x)
    }

    private static func regularisedIncompleteGammaCF(a: Double, x: Double) -> Double {
        // Modified Lentz evaluation of Q(a,x), valid in the x >= a + 1 region.
        let tiny = 1e-300
        var b = x + 1 - a
        var c = 1 / tiny
        var d = 1 / b
        var h = d
        for i in 1...10000 {
            let coefficient = Double(i) * (a - Double(i))
            b += 2
            d = b + coefficient * d
            if abs(d) < tiny { d = d < 0 ? -tiny : tiny }
            c = b + coefficient / c
            if abs(c) < tiny { c = c < 0 ? -tiny : tiny }
            d = 1 / d
            let delta = c * d
            h *= delta
            if abs(delta - 1) <= 2e-15 { break }
        }
        return exp(a * log(x) - x - lgamma(a)) * h
    }

    /// Temme's uniform asymptotic expansion for P(a, x) and Q(a, x) when a >= 1e5.
    private static func temmeIncompleteGamma(a: Double, x: Double) -> (p: Double, q: Double) {
        let mu = (x - a) / a
        let eta: Double
        let etaSqHalf: Double
        if abs(mu) < 0.05 {
            var s = 0.5
            var t = 1.0
            for k in 3..<35 {
                t *= -mu
                s += t / Double(k)
            }
            eta = mu * sqrt(2.0 * s)
            etaSqHalf = mu * mu * s
        } else {
            etaSqHalf = mu > -0.5 ? mu - log1p(mu) : mu - log(x / a)
            eta = copysign(sqrt(2.0 * etaSqHalf), mu)
        }

        let c0: Double
        let c1: Double
        let c2: Double
        let c3: Double
        if abs(eta) < 0.1 {
            c0 = poly([
                -1.0 / 3.0, 1.0 / 12.0, -2.0 / 135.0, 1.0 / 864.0, 1.0 / 2835.0,
                -139.0 / 777600.0, 39.19263178522438e-6, -2.185448510679992e-6
            ], eta)
            c1 = poly([
                -1.0 / 540.0, -1.0 / 288.0, 1.0 / 378.0, -77.0 / 77760.0, 1.0 / 4860.0,
                -1.0 / 2488320.0, -1.8098550334489977e-5, 7.64916091608111e-6
            ], eta)
            c2 = poly([
                25.0 / 6048.0, -139.0 / 51840.0, 1.0 / 1296.0, 1.0 / 497664.0,
                -6199.0 / 57736800.0, 5531.0 / 104509440.0
            ], eta)
            c3 = poly([
                101.0 / 155520.0, 571.0 / 2488320.0, -54179.0 / 115473600.0, 41969.0 / 156764160.0
            ], eta)
        } else {
            @inline(__always)
            func jetMul(_ p: [Double], _ q: [Double]) -> [Double] {
                var r = [Double](repeating: 0, count: 5)
                for i in 0..<5 {
                    for j in 0..<(5 - i) { r[i + j] += p[i] * q[j] }
                }
                return r
            }
            @inline(__always)
            func jetInv(_ p: [Double]) -> [Double] {
                var r = [Double](repeating: 0, count: 5)
                let inv0 = 1.0 / p[0]
                r[0] = inv0
                for k in 1..<5 {
                    var s = 0.0
                    for j in 1...k { s += p[j] * r[k - j] }
                    r[k] = -s * inv0
                }
                return r
            }

            let etaJet = [eta, 1.0, 0.0, 0.0, 0.0]
            var muJet = [mu, 0.0, 0.0, 0.0, 0.0]
            for k in 0..<4 {
                let gK = jetInv(muJet)
                let qK = jetMul(etaJet, gK)
                muJet[k + 1] = (etaJet[k] + qK[k]) / Double(k + 1)
            }
            let gJet = jetInv(muJet)
            let vJet = jetInv(etaJet)
            let lJet = jetMul(etaJet, gJet)
            var c0Jet = [Double](repeating: 0, count: 5)
            for k in 0..<5 { c0Jet[k] = gJet[k] - vJet[k] }

            var num1 = [Double](repeating: 0, count: 5)
            for k in 0..<4 { num1[k] = Double(k + 1) * c0Jet[k + 1] - (1.0 / 12.0) * lJet[k] }
            let c1Jet = jetMul(num1, vJet)

            var num2 = [Double](repeating: 0, count: 5)
            for k in 0..<3 { num2[k] = Double(k + 1) * c1Jet[k + 1] + (1.0 / 288.0) * lJet[k] }
            let c2Jet = jetMul(num2, vJet)

            var num3 = [Double](repeating: 0, count: 5)
            for k in 0..<2 { num3[k] = Double(k + 1) * c2Jet[k + 1] + (139.0 / 51840.0) * lJet[k] }
            let c3Jet = jetMul(num3, vJet)

            c0 = c0Jet[0]; c1 = c1Jet[0]; c2 = c2Jet[0]; c3 = c3Jet[0]
        }

        let series = c0 + (c1 + (c2 + c3 / a) / a) / a
        let correction = exp(-a * etaSqHalf) / sqrt(2.0 * Double.pi * a) * series
        let y = eta * sqrt(0.5 * a)
        if eta >= 0 {
            let q = Swift.min(1.0, Swift.max(0.0, 0.5 * erfc(y) + correction))
            return (1.0 - q, q)
        } else {
            let p = Swift.min(1.0, Swift.max(0.0, 0.5 * erfc(-y) - correction))
            return (p, 1.0 - p)
        }
    }

    /// Standard normal CDF via complementary error function.
    internal static func normalCDF(_ z: Double) -> Double {
        0.5 * erfc(-z / 2.0.squareRoot())
    }

    // MARK: – Shapiro-Wilk implementation (Royston AS R94, 1995)

    /// Affine rescaling preserves both normality statistics while avoiding square overflow.
    private static func scaledNormalitySample(_ values: [Double]) throws -> [Double] {
        guard values.allSatisfy(\.isFinite) else { throw StatsError.containsInfinity }
        let sorted = values.sorted()
        let lo = sorted[0], hi = sorted[sorted.count - 1]
        guard lo < hi else { throw StatsError.divisionByZero(context: "normality test on a constant sample") }
        let range = hi - lo
        if range.isFinite, range >= Double.leastNormalMagnitude {
            return sorted.map { ($0 - lo) / range }
        }
        let scale = Swift.max(abs(lo), abs(hi))
        let scaledLo = lo / scale
        let scaledRange = hi / scale - scaledLo
        return sorted.map { ($0 / scale - scaledLo) / scaledRange }
    }

    private static func shapiroWilkImpl(_ sorted: [Double]) -> (W: Double, p: Double) {
        let n = sorted.count
        let a = shapiroWilkCoefficients(n: n)
        var numerator = 0.0
        for i in a.indices {
            numerator += a[i] * (sorted[n - 1 - i] - sorted[i])
        }
        let mean = vDSP.mean(sorted)
        let ssq = vDSP.sumOfSquares(sorted.map { $0 - mean })
        let norm = 2 * vDSP.sumOfSquares(a)
        let denominator = sqrt(norm * ssq)
        // Factor 1-W before subtraction to improve the log-tail calculation near W=1.
        let oneMinusW = Swift.max(0, (denominator - numerator) * (denominator + numerator) / (norm * ssq))
        let w = 1 - oneMinusW
        if n == 3 {
            return (Swift.max(0.75, w), Swift.max(0, 1 - 6 / Double.pi * acos(sqrt(Swift.max(0.75, w)))))
        }
        guard oneMinusW > 0 else { return (1.0, 1.0) }
        var y = log(oneMinusW)
        let mu: Double
        let sigma: Double
        if n <= 11 {
            let size = Double(n)
            let gamma = -2.273 + 0.459 * size
            if y >= gamma { return (w, 1e-19) }
            y = -log(gamma - y)
            mu = poly([0.5440, -0.39978, 0.025054, -0.0006714], size)
            sigma = exp(poly([1.3822, -0.77857, 0.062767, -0.0020322], size))
        } else {
            let logN = log(Double(n))
            mu = poly([-1.5861, -0.31082, -0.083751, 0.0038915], logN)
            sigma = exp(poly([-0.4803, -0.082676, 0.0030302], logN))
        }
        return (w, 0.5 * erfc((y - mu) / (sigma * sqrt(2))))
    }

    private static func shapiroWilkCoefficients(n: Int) -> [Double] {
        if n == 3 { return [sqrt(0.5)] }
        let size = Double(n)
        var a = (0..<(n / 2)).map { normalQuantile((Double($0 + 1) - 0.375) / (size + 0.25)) }
        let sumSquares = 2 * vDSP.sumOfSquares(a)
        let normalization = sqrt(sumSquares)
        let inverseRootN = 1 / sqrt(size)
        let first = -a[0] / normalization
            + poly([0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056], inverseRootN)
        let adjustedCount = n > 5 ? 2 : 1
        var originalSquares = a[0] * a[0]
        var adjustedSquares = first * first
        if n > 5 {
            let second = -a[1] / normalization
                + poly([0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633], inverseRootN)
            originalSquares += a[1] * a[1]
            adjustedSquares += second * second
            a[1] = second
        }
        let factor = sqrt((sumSquares - 2 * originalSquares) / (1 - 2 * adjustedSquares))
        a[0] = first
        for i in adjustedCount..<a.count { a[i] /= -factor }
        return a
    }

    /// Normal quantile function (probit) via rational approximation (Beasley-Springer-Moro).
    private static func normalQuantile(_ p: Double) -> Double {
        guard p > 0 && p < 1 else { return p <= 0 ? -.infinity : .infinity }
        let a: [Double] = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
        let b: [Double] = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
        let c: [Double] = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
                           0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
                           0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
        let y = p - 0.5
        if abs(y) < 0.42 {
            let r = y * y
            return y * poly(a, r) / (1 + r * poly(b, r))
        }
        let r = p < 0.5 ? log(-log(p)) : log(-log(1 - p))
        let x = poly(c, r)
        return p < 0.5 ? -x : x
    }

    private static func poly(_ coeffs: [Double], _ x: Double) -> Double {
        coeffs.reversed().reduce(0) { $0 * x + $1 }
    }
}
