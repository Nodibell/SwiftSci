import Foundation
import Accelerate
import SwiftStats

/// Result of an anomaly detection evaluation.
public struct TimeSeriesAnomaly: Sendable, Equatable {
    public let index: Int
    public let value: Double
    public let expectedValue: Double
    public let score: Double
    public let isAnomaly: Bool

    public init(index: Int, value: Double, expectedValue: Double, score: Double, isAnomaly: Bool) {
        self.index = index
        self.value = value
        self.expectedValue = expectedValue
        self.score = score
        self.isAnomaly = isAnomaly
    }
}

public struct AnomalyDetectionResult: Sendable {
    public let anomalies: [TimeSeriesAnomaly]
    public let threshold: Double
    public let anomalyIndices: [Int]

    public init(anomalies: [TimeSeriesAnomaly], threshold: Double) {
        self.anomalies = anomalies
        self.threshold = threshold
        self.anomalyIndices = anomalies.filter(\.isAnomaly).map(\.index)
    }
}

/// Seasonal Extreme Studentized Deviate (S-ESD) & Residual-based Time Series Anomaly Detector.
public enum TimeSeriesAnomalyDetector {
    /// Detects temporal anomalies in a time series using Seasonal-ESD / STL residual analysis.
    /// - Parameters:
    ///   - series: Input temporal series.
    ///   - period: Seasonal period (optional; auto-estimated if nil).
    ///   - maxAnomaliesRatio: Maximum fraction of points to flag as anomalies (default: 0.10).
    ///   - thresholdZ: Robust Z-score threshold (default: 3.0 for ~99.7% confidence).
    /// - Returns: AnomalyDetectionResult with flagged anomalies and test statistics.
    public static func detectAnomalies(
        series: [Double],
        period: Int? = nil,
        maxAnomaliesRatio: Double = 0.10,
        thresholdZ: Double = 3.0
    ) -> AnomalyDetectionResult {
        let n = series.count
        guard n >= 4 else {
            let pts = series.enumerated().map { TimeSeriesAnomaly(index: $0.offset, value: $0.element, expectedValue: $0.element, score: 0.0, isAnomaly: false) }
            return AnomalyDetectionResult(anomalies: pts, threshold: thresholdZ)
        }

        let p: Int = {
            if let userPeriod = period, userPeriod >= 2 { return userPeriod }
            if n >= 28 { return 7 }
            if n >= 12 { return 4 }
            return 2
        }()

        // 1. Compute expected baseline & residuals via decomposition or robust median
        var expectedValues = [Double](repeating: 0.0, count: n)
        var residuals = [Double](repeating: 0.0, count: n)

        if n >= p * 2, let decomp = try? TimeSeriesDecomposition.decompose(series: series, period: p, model: .additive) {
            let medTrend = median(decomp.trend.filter { !$0.isNaN }) ?? 0.0
            for i in 0..<n {
                let t = decomp.trend[i].isNaN ? medTrend : decomp.trend[i]
                let s = decomp.seasonal[i]
                expectedValues[i] = t + s
                residuals[i] = series[i] - expectedValues[i]
            }
        } else {
            // Running median baseline for shorter series
            let window = min(7, max(3, n / 4))
            for i in 0..<n {
                let start = max(0, i - window / 2)
                let end = min(n, i + window / 2 + 1)
                let winSlice = Array(series[start..<end])
                let exp = median(winSlice) ?? series[i]
                expectedValues[i] = exp
                residuals[i] = series[i] - exp
            }
        }

        // 2. Compute Median Absolute Deviation (MAD) of residuals
        let validRes = residuals.filter { !$0.isNaN }
        let medResidual = median(validRes) ?? 0.0
        let absDevs = validRes.map { abs($0 - medResidual) }
        let mad = median(absDevs) ?? 1.0
        let normalScale = 1.4826 * max(mad, 1e-6)

        // 3. Compute robust Z-scores
        var scores = [Double](repeating: 0.0, count: n)
        for i in 0..<n {
            scores[i] = abs(residuals[i] - medResidual) / normalScale
        }

        // 4. Determine anomalies respecting maxAnomaliesRatio
        let maxAnomalies = max(1, Int(Double(n) * maxAnomaliesRatio))
        let sortedCandidateIndices = (0..<n)
            .filter { scores[$0] >= thresholdZ }
            .sorted { scores[$0] > scores[$1] }
        let flaggedSet = Set(sortedCandidateIndices.prefix(maxAnomalies))

        var anomalies: [TimeSeriesAnomaly] = []
        anomalies.reserveCapacity(n)
        for i in 0..<n {
            let isAnom = flaggedSet.contains(i)
            anomalies.append(TimeSeriesAnomaly(
                index: i,
                value: series[i],
                expectedValue: expectedValues[i],
                score: scores[i],
                isAnomaly: isAnom
            ))
        }

        return AnomalyDetectionResult(anomalies: anomalies, threshold: thresholdZ)
    }

    private static func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let count = sorted.count
        if count % 2 == 0 {
            return (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            return sorted[count / 2]
        }
    }
}
