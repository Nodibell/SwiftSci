import Accelerate
import Foundation

// MARK: – Hypothesis Testing

extension Stats {

    // MARK: One-sample t-test

    /// Tests whether the sample mean equals `mu`.
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
    /// - Parameter equalVariances: If true, uses pooled variance (Student). Default false (Welch).
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

        if equalVariances {
            // Student's t (pooled variance)
            let sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            guard sp2 > 0 else { throw StatsError.divisionByZero(context: "pooled tTest") }
            let se = (sp2 * (1/n1 + 1/n2)).squareRoot()
            t  = (x1 - x2) / se
            df = n1 + n2 - 2
        } else {
            // Welch's t-test
            let se1 = var1 / n1
            let se2 = var2 / n2
            let se  = (se1 + se2).squareRoot()
            guard se > 0 else { throw StatsError.divisionByZero(context: "Welch tTest") }
            t  = (x1 - x2) / se
            // Welch-Satterthwaite degrees of freedom
            df = pow(se1 + se2, 2) / (pow(se1, 2)/(n1-1) + pow(se2, 2)/(n2-1))
        }

        let pValue = 2.0 * tDistributionCDF(-abs(t), df: df)
        let margin = criticalT(df: df, alpha: 0.05) * ((var1/n1 + var2/n2).squareRoot())
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
    public static func pairedTTest(before: [Double], after: [Double]) throws -> TTestResult {
        try requireSameSize(before, after)
        // zip subtraction: diffs[i] = after[i] - before[i]
        let diffs = zip(before, after).map { $1 - $0 }
        return try tTest(sample: diffs, populationMean: 0)
    }

    // MARK: One-way ANOVA

    /// Tests whether the means of 2+ groups are equal.
    public static func oneWayANOVA(groups: [[Double]]) throws -> ANOVAResult {
        guard groups.count >= 2 else {
            throw StatsError.invalidGroupCount(minimum: 2, got: groups.count)
        }
        for g in groups { try requireNonEmpty(g, minimum: 1); try requireNoNaN(g) }

        let k = groups.count
        let n = groups.reduce(0) { $0 + $1.count }  // total observations

        let grandMean = try mean(groups.flatMap { $0 })

        // SS_between = Σ nᵢ (x̄ᵢ - x̄)²
        var ssBetween = 0.0
        for g in groups {
            let ni   = Double(g.count)
            let xBar = vDSP.mean(g)
            ssBetween += ni * pow(xBar - grandMean, 2)
        }

        // SS_within = Σ Σ (xᵢⱼ - x̄ᵢ)²
        var ssWithin = 0.0
        for g in groups {
            let xBar = vDSP.mean(g)
            var centered = [Double](repeating: 0, count: g.count)
            vDSP.add(-xBar, g, result: &centered)
            ssWithin += vDSP.sumOfSquares(centered)
        }

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
    public static func chiSquareGoodnessOfFit(observed: [Double],
                                               expected: [Double]) throws -> ChiSquareResult {
        try requireNonEmpty(observed)
        try requireSameSize(observed, expected)

        var chi2 = 0.0
        for (o, e) in zip(observed, expected) {
            guard e > 0 else { throw StatsError.divisionByZero(context: "chiSquare expected=0") }
            chi2 += pow(o - e, 2) / e
        }

        let df     = observed.count - 1
        let pValue = chiSquareSurvival(chi2: chi2, df: Double(df))

        return ChiSquareResult(statistic: chi2, pValue: pValue, degreesOfFreedom: df)
    }

    // MARK: Normality Tests

    /// Shapiro-Wilk normality test (Royston 1992, n ∈ [3, 5000]).
    public static func shapiroWilk(_ values: [Double]) throws -> NormalityTestResult {
        try requireNonEmpty(values, minimum: 3)
        try requireNoNaN(values)
        guard values.count <= 5000 else {
            throw StatsError.tooManySamples(limit: 5000, got: values.count)
        }
        let (w, p) = shapiroWilkImpl(values)
        return NormalityTestResult(statistic: w, pValue: p)
    }

    /// Kolmogorov-Smirnov test against standard normal (Lilliefors variant for unknown μ/σ).
    public static func kolmogorovSmirnov(_ values: [Double]) throws -> NormalityTestResult {
        try requireNonEmpty(values, minimum: 3)
        try requireNoNaN(values)

        let mu  = vDSP.mean(values)
        let s   = try standardDeviation(values, ddof: 1)
        guard s > 0 else { return NormalityTestResult(statistic: 0, pValue: 1.0) }

        let sorted   = values.sorted()
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
        let pValue = ksTestPValue(d: d, n: Int(n))

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
    internal static func chiSquareSurvival(chi2: Double, df: Double) -> Double {
        guard chi2 > 0 else { return 1.0 }
        return 1.0 - regularisedIncompleteGamma(a: df / 2, x: chi2 / 2)
    }

    /// KS test p-value approximation.
    private static func ksTestPValue(d: Double, n: Int) -> Double {
        let lambda = (Double(n).squareRoot() + 0.12 + 0.11 / Double(n).squareRoot()) * d
        guard lambda > 0 else { return 1.0 }
        var p = 0.0
        for k in 1...100 {
            let term = pow(-1.0, Double(k-1)) * exp(-2 * pow(Double(k) * lambda, 2))
            p += term
        }
        return Swift.max(0, Swift.min(1, 2 * p))
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
    internal static func regularisedIncompleteGamma(a: Double, x: Double) -> Double {
        guard x >= 0 else { return 0 }
        if x == 0 { return 0 }

        // Series expansion for x < a + 1
        if x < a + 1 {
            var term = 1.0 / a
            var sum  = term
            for n in 1...300 {
                term *= x / (a + Double(n))
                sum  += term
                if abs(term) < 1e-12 * abs(sum) { break }
            }
            return sum * exp(-x + a * log(x) - lgamma(a))
        }

        // Continued fraction for x >= a + 1 (complement)
        return 1.0 - regularisedIncompleteGammaCF(a: a, x: x)
    }

    private static func regularisedIncompleteGammaCF(a: Double, x: Double) -> Double {
        var f   = 1.0 + (1 - a) / x
        var C   = f
        var D   = 1.0 / f
        let eps = 1e-15

        for i in 1...300 {
            let di = Double(i)
            // Even step: b_n = 1 + (2n - a) / x
            let b = 1 + (2*di - a) / x
            D = b + di * (a - di) * D
            if abs(D) < eps { D = eps }
            D = 1 / D

            C = b + di * (a - di) / C
            if abs(C) < eps { C = eps }

            let delta = C * D
            f *= delta
            if abs(delta - 1) < 1e-12 { break }
        }
        return exp(-x + a * log(x) - lgamma(a)) / f
    }

    /// Standard normal CDF via complementary error function.
    internal static func normalCDF(_ z: Double) -> Double {
        0.5 * erfc(-z / 2.0.squareRoot())
    }

    // MARK: – Shapiro-Wilk implementation (Royston 1992)

    private static func shapiroWilkImpl(_ x: [Double]) -> (W: Double, p: Double) {
        let sorted = x.sorted()
        let n      = sorted.count

        // Compute a-coefficients (polynomial approximation for n ≤ 2000)
        let a = shapiroWilkCoefficients(n: n)

        // Compute W statistic
        var b = 0.0
        let m  = n / 2
        for i in 0..<m {
            b += a[i] * (sorted[n - 1 - i] - sorted[i])
        }

        let centred = sorted.map { $0 - vDSP.mean(sorted) }
        let ssq = vDSP.sumOfSquares(centred)
        let W   = ssq > 0 ? (b * b) / ssq : 1.0

        // p-value via log transform (Royston 1992)
        let mu: Double
        let sigma: Double
        let gamma: Double
        let lnN = log(Double(n))

        if n <= 11 {
            let gamma_c: [Double] = [-2.273, 0.459]
            gamma = gamma_c[0] + gamma_c[1] * Double(n)
            let mu_c: [Double]    = [-0.0006714, 0.025054, -0.6714, 0.7240]
            let sigma_c: [Double] = [-0.0020322, 0.04981, -0.1358, -0.2031]
            mu    = poly(mu_c,    lnN)
            sigma = exp(poly(sigma_c, lnN))
        } else {
            let mu_c: [Double]    = [-1.2725, 1.0521]
            let sigma_c: [Double] = [-1.1098, 1.5198]
            mu    = poly(mu_c,    lnN)
            sigma = exp(poly(sigma_c, lnN))
            gamma = 0
        }

        let lnW = log(1.0 - W)
        let z   = (lnW - mu) / sigma - gamma
        let p   = 1.0 - normalCDF(z)
        return (W, Swift.max(0, Swift.min(1, p)))
    }

    private static func shapiroWilkCoefficients(n: Int) -> [Double] {
        // Approximate half-normal order statistics (Blom 1958 formula)
        var a = [Double](repeating: 0, count: n / 2)
        let m   = n / 2
        let nd  = Double(n)
        var mtilde = [Double](repeating: 0, count: n)
        for i in 0..<n {
            // approximate expected order statistics of N(0,1)
            let p = (Double(i + 1) - 0.375) / (nd + 0.25)
            mtilde[i] = normalQuantile(p)
        }
        // Compute normalising constant
        var c = [Double](repeating: 0, count: n)
        vDSP.multiply(mtilde, mtilde, result: &c)
        let cn = vDSP.sumOfSquares(mtilde).squareRoot()
        for i in 0..<m {
            a[i] = mtilde[n - 1 - i] / cn
        }
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
            return y * poly(a, r) / poly(b, r)
        }
        let r = p < 0.5 ? log(-log(p)) : log(-log(1 - p))
        let x = poly(c, r)
        return p < 0.5 ? -x : x
    }

    private static func poly(_ coeffs: [Double], _ x: Double) -> Double {
        coeffs.reversed().reduce(0) { $0 * x + $1 }
    }
}
