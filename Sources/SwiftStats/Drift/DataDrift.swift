import Foundation
import SwiftDataFrame

// MARK: - Drift Severity

/// Qualitative drift classification based on empirical metric thresholds.
public enum DriftSeverity: String, Sendable, Codable, CaseIterable {
    /// Negligible or no detectable distribution shift (PSI < 0.1).
    case stable
    /// Moderate distribution shift; monitoring recommended (0.1 <= PSI < 0.2).
    case moderate
    /// Significant distribution drift; model retraining or recalibration advised (PSI >= 0.2).
    case significant
}

// MARK: - PSI Bucket

/// Detailed drift metric breakdown for an individual distribution bin.
public struct PSIBucket: Sendable, Codable, Equatable {
    /// Lower bound of the bin interval (inclusive).
    public let lowerBound: Double
    /// Upper bound of the bin interval (inclusive for the final bucket, exclusive otherwise).
    public let upperBound: Double
    /// Proportion of reference / expected samples falling into this bucket.
    public let expectedPercentage: Double
    /// Proportion of target / actual samples falling into this bucket.
    public let actualPercentage: Double
    /// Individual contribution of this bucket to the total Population Stability Index.
    public let contribution: Double
    
    /// Initializes a new bucket record.
    /// - Parameters:
    ///   - lowerBound: Lower boundary of the interval.
    ///   - upperBound: Upper boundary of the interval.
    ///   - expectedPercentage: Fraction of expected samples in this bucket.
    ///   - actualPercentage: Fraction of actual samples in this bucket.
    ///   - contribution: Individual bucket contribution to overall PSI.
    public init(
        lowerBound: Double,
        upperBound: Double,
        expectedPercentage: Double,
        actualPercentage: Double,
        contribution: Double
    ) {
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.expectedPercentage = expectedPercentage
        self.actualPercentage = actualPercentage
        self.contribution = contribution
    }
}

// MARK: - PSI Result

/// Aggregated Population Stability Index (PSI) assessment across dataset partitions.
public struct PSIResult: Sendable, Codable, Equatable {
    /// Total Population Stability Index computed over all bins.
    public let psi: Double
    /// Qualitative interpretation of the computed stability index.
    public let severity: DriftSeverity
    /// Individual bin calculations and empirical frequency comparisons.
    public let buckets: [PSIBucket]
    
    /// Initializes a PSI assessment result.
    /// - Parameters:
    ///   - psi: Total Population Stability Index value.
    ///   - severity: Qualitative severity classification.
    ///   - buckets: Per-bucket empirical frequencies and contributions.
    public init(psi: Double, severity: DriftSeverity, buckets: [PSIBucket]) {
        self.psi = psi
        self.severity = severity
        self.buckets = buckets
    }
}

// MARK: - Wasserstein Distance

/// Computes the 1-Wasserstein metric (Earth Mover's Distance) between two continuous empirical distributions.
public enum WassersteinDistance {
    /// Computes the first Wasserstein distance (L1 Earth Mover's Distance) between two empirical 1D samples.
    ///
    /// For two empirical continuous samples $u$ and $v$, the 1-Wasserstein distance corresponds to the area
    /// between their empirical cumulative distribution functions (ECDFs):
    /// $$W_1(u, v) = \int_{-\infty}^{\infty} |F_u(x) - F_v(x)| \, dx$$
    ///
    /// - Parameters:
    ///   - u: Reference sample distribution.
    ///   - v: Target sample distribution to compare against.
    /// - Returns: Non-negative scalar representing the minimal work required to transform distribution $u$ into $v$.
    /// - Throws: `SwiftMLError.emptyInput` if either sample array is empty.
    public static func compute(_ u: [Double], _ v: [Double]) throws -> Double {
        guard !u.isEmpty && !v.isEmpty else {
            throw SwiftMLError.emptyInput
        }
        
        let sortedU = u.sorted()
        let sortedV = v.sorted()
        
        let n = sortedU.count
        let m = sortedV.count
        
        var allValues = Set<Double>()
        allValues.reserveCapacity(n + m)
        for x in sortedU { allValues.insert(x) }
        for x in sortedV { allValues.insert(x) }
        let sortedX = allValues.sorted()
        
        guard sortedX.count > 1 else {
            return 0.0
        }
        
        var distance = 0.0
        var idxU = 0
        var idxV = 0
        
        let doubleN = Double(n)
        let doubleM = Double(m)
        
        for i in 0..<(sortedX.count - 1) {
            let xCurrent = sortedX[i]
            let xNext = sortedX[i + 1]
            let dx = xNext - xCurrent
            
            while idxU < n && sortedU[idxU] <= xCurrent {
                idxU += 1
            }
            while idxV < m && sortedV[idxV] <= xCurrent {
                idxV += 1
            }
            
            let cdfU = Double(idxU) / doubleN
            let cdfV = Double(idxV) / doubleM
            distance += abs(cdfU - cdfV) * dx
        }
        
        return distance
    }
}

// MARK: - Population Stability Index

/// Evaluates population drift between expected (baseline) and actual (target) data partitions.
public enum PopulationStabilityIndex {
    /// Computes the Population Stability Index (PSI) by binning the continuous distributions.
    ///
    /// The metric bins the reference distribution into quantiles and evaluates the divergence
    /// across partitions:
    /// $$\text{PSI} = \sum_{i=1}^B (A_i - E_i) \cdot \ln\left(\frac{A_i}{E_i}\right)$$
    ///
    /// - Parameters:
    ///   - expected: Baseline or training reference values.
    ///   - actual: Target or validation/test evaluation values.
    ///   - buckets: Number of quantile bins to partition the reference range into (defaults to 10).
    ///   - epsilon: Small positive offset added to zero bins to guarantee numerical stability (defaults to 1e-4).
    /// - Returns: A `PSIResult` containing total PSI score, severity rating, and bucket diagnostics.
    /// - Throws: `SwiftMLError.emptyInput` if either input is empty, or `SwiftMLError.invalidParameter` if buckets < 2.
    public static func compute(
        expected: [Double],
        actual: [Double],
        buckets: Int = 10,
        epsilon: Double = 1e-4
    ) throws -> PSIResult {
        guard !expected.isEmpty && !actual.isEmpty else {
            throw SwiftMLError.emptyInput
        }
        guard buckets >= 2 else {
            throw SwiftMLError.invalidParameter("Number of buckets must be at least 2, got \(buckets)")
        }
        
        let sortedExp = expected.sorted()
        let nExp = sortedExp.count
        
        // Derive bucket cutpoints from expected quantiles
        var cutpoints: [Double] = []
        cutpoints.reserveCapacity(buckets + 1)
        cutpoints.append(-Double.infinity)
        for b in 1..<buckets {
            let q = Double(b) / Double(buckets)
            let idx = Int((Double(nExp - 1) * q).rounded())
            cutpoints.append(sortedExp[idx])
        }
        cutpoints.append(Double.infinity)
        
        // Remove duplicate cutpoints if data is discrete or heavily repeated
        var uniqueCutpoints: [Double] = []
        for cp in cutpoints {
            if let last = uniqueCutpoints.last {
                if cp > last {
                    uniqueCutpoints.append(cp)
                }
            } else {
                uniqueCutpoints.append(cp)
            }
        }
        
        if uniqueCutpoints.count < 2 {
            uniqueCutpoints = [-Double.infinity, Double.infinity]
        }
        
        let numBins = uniqueCutpoints.count - 1
        var expCounts = [Int](repeating: 0, count: numBins)
        var actCounts = [Int](repeating: 0, count: numBins)
        
        for val in expected {
            let bin = findBin(val: val, cutpoints: uniqueCutpoints)
            expCounts[bin] += 1
        }
        for val in actual {
            let bin = findBin(val: val, cutpoints: uniqueCutpoints)
            actCounts[bin] += 1
        }
        
        let totalExp = Double(expected.count)
        let totalAct = Double(actual.count)
        
        var totalPSI = 0.0
        var bucketResults: [PSIBucket] = []
        bucketResults.reserveCapacity(numBins)
        
        for b in 0..<numBins {
            let pExp = Double(expCounts[b]) / totalExp
            let pAct = Double(actCounts[b]) / totalAct
            
            let e = max(pExp, epsilon)
            let a = max(pAct, epsilon)
            let contribution = (a - e) * log(a / e)
            totalPSI += contribution
            
            bucketResults.append(PSIBucket(
                lowerBound: uniqueCutpoints[b],
                upperBound: uniqueCutpoints[b + 1],
                expectedPercentage: pExp,
                actualPercentage: pAct,
                contribution: contribution
            ))
        }
        
        let severity: DriftSeverity
        if totalPSI < 0.1 {
            severity = .stable
        } else if totalPSI < 0.2 {
            severity = .moderate
        } else {
            severity = .significant
        }
        
        return PSIResult(psi: totalPSI, severity: severity, buckets: bucketResults)
    }
    
    private static func findBin(val: Double, cutpoints: [Double]) -> Int {
        let numBins = cutpoints.count - 1
        for i in 0..<numBins {
            let lower = cutpoints[i]
            let upper = cutpoints[i + 1]
            if i == numBins - 1 {
                if val >= lower && val <= upper {
                    return i
                }
            } else {
                if val >= lower && val < upper {
                    return i
                }
            }
        }
        return numBins - 1
    }
}

// MARK: - Stats Extension

extension Stats {
    /// Computes the 1-Wasserstein (Earth Mover's Distance) between two continuous empirical samples.
    ///
    /// - Parameters:
    ///   - u: Reference sample distribution.
    ///   - v: Target sample distribution.
    /// - Returns: Scalar distance representing minimal transportation cost between distributions.
    /// - Throws: `SwiftMLError.emptyInput` if either input array is empty.
    public static func wassersteinDistance(_ u: [Double], _ v: [Double]) throws -> Double {
        try WassersteinDistance.compute(u, v)
    }
    
    /// Computes the Population Stability Index (PSI) between baseline and evaluation samples.
    ///
    /// - Parameters:
    ///   - expected: Baseline reference distribution.
    ///   - actual: Target comparison distribution.
    ///   - buckets: Number of quantile bins (defaults to 10).
    /// - Returns: A `PSIResult` with total score, severity classification, and bucket breakdown.
    /// - Throws: `SwiftMLError.emptyInput` if either sample is empty.
    public static func populationStabilityIndex(
        expected: [Double],
        actual: [Double],
        buckets: Int = 10
    ) throws -> PSIResult {
        try PopulationStabilityIndex.compute(expected: expected, actual: actual, buckets: buckets)
    }
}
