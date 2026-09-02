import Testing
@testable import SwiftForecast

// MARK: - TimeSeriesAnomaly

@Suite("TimeSeriesAnomaly")
struct TimeSeriesAnomalyTests {

    @Test("init stores all fields correctly")
    func initStoresFields() {
        let a = TimeSeriesAnomaly(index: 3, value: 10.0, expectedValue: 5.0, score: 3.5, isAnomaly: true)
        #expect(a.index == 3)
        #expect(a.value == 10.0)
        #expect(a.expectedValue == 5.0)
        #expect(a.score == 3.5)
        #expect(a.isAnomaly == true)
    }

    @Test("Equatable: equal instances are equal")
    func equatable() {
        let a = TimeSeriesAnomaly(index: 0, value: 1.0, expectedValue: 1.0, score: 0.0, isAnomaly: false)
        let b = TimeSeriesAnomaly(index: 0, value: 1.0, expectedValue: 1.0, score: 0.0, isAnomaly: false)
        #expect(a == b)
    }

    @Test("Equatable: different isAnomaly flag makes them unequal")
    func notEquatable() {
        let a = TimeSeriesAnomaly(index: 0, value: 1.0, expectedValue: 1.0, score: 0.0, isAnomaly: false)
        let b = TimeSeriesAnomaly(index: 0, value: 1.0, expectedValue: 1.0, score: 0.0, isAnomaly: true)
        #expect(a != b)
    }
}

// MARK: - AnomalyDetectionResult

@Suite("AnomalyDetectionResult")
struct AnomalyDetectionResultTests {

    @Test("anomalyIndices contains only flagged indices")
    func anomalyIndices() {
        let pts = [
            TimeSeriesAnomaly(index: 0, value: 1.0, expectedValue: 1.0, score: 0.5, isAnomaly: false),
            TimeSeriesAnomaly(index: 1, value: 9.0, expectedValue: 1.0, score: 4.0, isAnomaly: true),
            TimeSeriesAnomaly(index: 2, value: 1.5, expectedValue: 1.0, score: 0.3, isAnomaly: false),
        ]
        let result = AnomalyDetectionResult(anomalies: pts, threshold: 3.0)
        #expect(result.anomalyIndices == [1])
        #expect(result.threshold == 3.0)
        #expect(result.anomalies.count == 3)
    }

    @Test("no anomalies yields empty anomalyIndices")
    func noAnomalies() {
        let pts = [
            TimeSeriesAnomaly(index: 0, value: 1.0, expectedValue: 1.0, score: 0.1, isAnomaly: false),
            TimeSeriesAnomaly(index: 1, value: 1.1, expectedValue: 1.0, score: 0.2, isAnomaly: false),
        ]
        let result = AnomalyDetectionResult(anomalies: pts, threshold: 3.0)
        #expect(result.anomalyIndices.isEmpty)
    }
}

// MARK: - TimeSeriesAnomalyDetector

@Suite("TimeSeriesAnomalyDetector")
struct TimeSeriesAnomalyDetectorTests {

    @Test("flat series produces no anomalies")
    func flatSeries() {
        let series = [Double](repeating: 1.0, count: 50)
        let result = TimeSeriesAnomalyDetector.detectAnomalies(series: series)
        #expect(result.anomalyIndices.isEmpty)
    }

    @Test("single spike is detected as anomaly")
    func singleSpike() {
        var series = [Double](repeating: 1.0, count: 40)
        series[20] = 100.0   // obvious outlier
        let result = TimeSeriesAnomalyDetector.detectAnomalies(series: series, thresholdZ: 2.5)
        #expect(result.anomalyIndices.contains(20))
    }

    @Test("series shorter than 4 returns all non-anomaly points")
    func tooShortSeries() {
        let result = TimeSeriesAnomalyDetector.detectAnomalies(series: [1.0, 2.0, 3.0])
        #expect(result.anomalies.count == 3)
        #expect(result.anomalyIndices.isEmpty)
    }

    @Test("explicit period is respected")
    func explicitPeriod() {
        var series = (0..<28).map { Double($0 % 7) }
        series[14] = 50.0  // outlier
        let result = TimeSeriesAnomalyDetector.detectAnomalies(series: series, period: 7, thresholdZ: 2.5)
        #expect(result.anomalyIndices.contains(14))
    }

    @Test("maxAnomaliesRatio caps number of anomalies")
    func maxAnomaliesRatio() {
        // All values are outliers relative to baseline, but ratio caps at 10%
        var series = (0..<50).map { _ in Double.random(in: 1...2) }
        for i in stride(from: 0, to: 50, by: 5) { series[i] = 1000.0 }
        let result = TimeSeriesAnomalyDetector.detectAnomalies(series: series, maxAnomaliesRatio: 0.10, thresholdZ: 2.0)
        #expect(result.anomalyIndices.count <= 5)
    }

    @Test("result contains one entry per input point")
    func resultCountMatchesInput() {
        let series = (0..<30).map { Double($0) }
        let result = TimeSeriesAnomalyDetector.detectAnomalies(series: series)
        #expect(result.anomalies.count == 30)
    }

    @Test("all non-anomaly points have isAnomaly == false")
    func nonAnomalyFlag() {
        let series = [Double](repeating: 5.0, count: 20)
        let result = TimeSeriesAnomalyDetector.detectAnomalies(series: series)
        #expect(result.anomalies.allSatisfy { !$0.isAnomaly })
    }
}
