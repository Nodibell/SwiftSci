import Testing
import Foundation
@testable import SwiftForecast

/// Deterministic anomaly fixture (S-04). Metadata:
/// - n = 144, period = 12, seeded LCG (seed 2026), noise amplitude ±0.5, trend 0.05/step, seasonal amplitude 5
/// - 21 point anomalies, amplitudes ±12…±20, at the indices listed in `truth`
/// The detector's default cap is `max(1, Int(n * 0.10)) = 14`, so with default parameters it can
/// never flag more than 14 points; flagging 22 points (e.g. TP=20, FP=2) would need `maxAnomaliesRatio >= 22/144 ≈ 0.1528`
/// *and* enough candidates above `thresholdZ`.
@Suite("TimeSeriesAnomalyDetector fixture and contract")
struct AnomalyFixtureTests {
    static let n = 144, period = 12
    static let truth: [Int] = (0..<20).map { 3 + 7 * $0 } + [141]   // phases mod 12 are spread over the whole cycle

    static func fixture() -> [Double] {
        var s: UInt64 = 2026
        func rnd() -> Double { s = s &* 6364136223846793005 &+ 1442695040888963407; return Double(s >> 11) / Double(1 << 53) - 0.5 }
        var y = (0..<n).map { 0.05 * Double($0) + 5 * sin(2 * .pi * Double($0) / Double(period)) + rnd() }
        for (k, i) in truth.enumerated() { y[i] += (k % 2 == 0 ? 1 : -1) * (12 + Double(k % 5) * 2) }
        return y
    }

    private func metrics(_ flagged: [Int]) -> (tp: Int, fp: Int, fn: Int) {
        let t = Set(Self.truth), f = Set(flagged)
        return (t.intersection(f).count, f.subtracting(t).count, t.subtracting(f).count)
    }

    @Test func fixtureHas144PointsAnd21Truths() {
        #expect(Self.fixture().count == 144)
        #expect(Set(Self.truth).count == 21)
    }

    @Test func defaultRatioCapsAt14() {
        let r = TimeSeriesAnomalyDetector.detectAnomalies(series: Self.fixture(), period: Self.period)
        let m = metrics(r.anomalyIndices)
        // Observed on this fixture: the cap of 14 is reached, all flags are true positives.
        #expect(r.anomalyIndices.count == 14)
        #expect(m.tp == 14 && m.fp == 0 && m.fn == 7)
    }

    @Test func explicitRatioRecoversMoreAndReportsConfusion() {
        let r = TimeSeriesAnomalyDetector.detectAnomalies(series: Self.fixture(), period: Self.period, maxAnomaliesRatio: 0.20)
        let m = metrics(r.anomalyIndices)
        // With the cap lifted the detector stops at the z >= 3 threshold: 15 flags, TP=15, FP=0, FN=6
        // (precision 1.0, recall 15/21 ≈ 0.714). Raising the ratio alone does not reach TP=20, FP=2.
        #expect(r.anomalyIndices.count == 15)
        #expect(m.tp == 15 && m.fp == 0 && m.fn == 6)
        let precision = Double(m.tp) / Double(max(1, m.tp + m.fp)), recall = Double(m.tp) / 21
        #expect(precision == 1.0 && abs(recall - 15.0 / 21.0) < 1e-12)
    }

    @Test func flaggedCountNeverExceedsCapForVariousRatios() {
        let y = Self.fixture()
        for ratio in [0.001, 0.05, 0.10, 0.1458, 0.20, 0.5, 0.9] {
            let cap = max(1, Int(Double(y.count) * max(0.001, min(0.5, ratio))))
            let r = TimeSeriesAnomalyDetector.detectAnomalies(series: y, period: Self.period, maxAnomaliesRatio: ratio)
            #expect(r.anomalyIndices.count <= cap)
        }
    }

    @Test func constantSeriesHasNoAnomalies() {
        let r = TimeSeriesAnomalyDetector.detectAnomalies(series: [Double](repeating: 3, count: 60))
        #expect(r.anomalyIndices.isEmpty)
    }

    @Test func invalidPeriodAndIrregularInputDoNotCrash() {
        let y = Self.fixture()
        for p in [-3, 0, 1, 500] {
            let r = TimeSeriesAnomalyDetector.detectAnomalies(series: y, period: p)
            #expect(r.anomalies.count == y.count)
        }
        let r = TimeSeriesAnomalyDetector.detectAnomalies(series: [], period: 7)
        #expect(r.anomalies.isEmpty)
    }

    @Test func nanAndInfInputsDoNotCrash() {
        var y = Self.fixture()
        y[10] = .nan; y[20] = .infinity; y[30] = -.infinity
        let r = TimeSeriesAnomalyDetector.detectAnomalies(series: y, period: Self.period)
        #expect(r.anomalies.count == y.count)
    }
}
