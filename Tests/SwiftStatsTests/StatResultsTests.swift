import Testing
import Foundation
@testable import SwiftStats

// MARK: - StatResults Coverage Tests

@Suite("StatResults Description Tests")
struct StatResultsDescriptionTests {

    // MARK: DescriptiveStats

    @Test("DescriptiveStats description contains count")
    func testDescriptiveStatsDescriptionContainsCount() {
        let stats = DescriptiveStats(
            count: 50,
            mean: 5.0,
            standardDeviation: 1.5,
            variance: 2.25,
            min: 1.0,
            q1: 3.5,
            median: 5.0,
            q3: 6.5,
            max: 10.0,
            skewness: 0.1,
            kurtosis: -0.2,
            nullCount: 0
        )
        let desc = stats.description
        #expect(desc.contains("50"))
        #expect(desc.contains("5.000000"))
        #expect(desc.contains("1.500000"))
        #expect(desc.contains("10.000000"))
    }

    @Test("DescriptiveStats description contains null count")
    func testDescriptiveStatsNullCount() {
        let stats = DescriptiveStats(
            count: 10,
            mean: 0.0,
            standardDeviation: 0.0,
            variance: 0.0,
            min: 0.0,
            q1: 0.0,
            median: 0.0,
            q3: 0.0,
            max: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            nullCount: 3
        )
        #expect(stats.description.contains("3"))
        #expect(stats.nullCount == 3)
    }

    @Test("DescriptiveStats description contains skewness and kurtosis")
    func testDescriptiveStatsSkewnessKurtosis() {
        let stats = DescriptiveStats(
            count: 100,
            mean: 2.5,
            standardDeviation: 0.8,
            variance: 0.64,
            min: 0.5,
            q1: 2.0,
            median: 2.5,
            q3: 3.0,
            max: 5.5,
            skewness: 0.45678,
            kurtosis: -1.2345,
            nullCount: 0
        )
        let desc = stats.description
        #expect(desc.contains("0.456780"))
        #expect(desc.contains("-1.234500"))
    }

    // MARK: TTestResult

    @Test("TTestResult description formats correctly")
    func testTTestResultDescription() {
        let ci = ConfidenceInterval(lower: 1.23, upper: 4.56, confidence: 0.95)
        let result = TTestResult(
            statistic: 3.1415,
            pValue: 0.0234,
            degreesOfFreedom: 28.0,
            confidenceInterval: ci,
            effectSize: 0.7890
        )
        let desc = result.description
        #expect(desc.contains("28.00"))
        #expect(desc.contains("3.1415"))
        #expect(desc.contains("0.0234"))
        #expect(desc.contains("0.7890"))
    }

    @Test("TTestResult isSignificant is true when p < 0.05")
    func testTTestIsSignificantTrue() {
        let ci = ConfidenceInterval(lower: 0.1, upper: 2.0, confidence: 0.95)
        let result = TTestResult(
            statistic: 3.0,
            pValue: 0.01,
            degreesOfFreedom: 20.0,
            confidenceInterval: ci,
            effectSize: 0.5
        )
        #expect(result.isSignificant == true)
    }

    @Test("TTestResult isSignificant is false when p >= 0.05")
    func testTTestIsSignificantFalse() {
        let ci = ConfidenceInterval(lower: -0.5, upper: 1.5, confidence: 0.95)
        let result = TTestResult(
            statistic: 1.2,
            pValue: 0.25,
            degreesOfFreedom: 18.0,
            confidenceInterval: ci,
            effectSize: 0.2
        )
        #expect(result.isSignificant == false)
    }

    @Test("TTestResult isSignificant is false exactly at boundary p == 0.05")
    func testTTestIsSignificantBoundary() {
        let ci = ConfidenceInterval(lower: 0.0, upper: 1.0, confidence: 0.95)
        let result = TTestResult(
            statistic: 2.0,
            pValue: 0.05,
            degreesOfFreedom: 30.0,
            confidenceInterval: ci,
            effectSize: 0.3
        )
        #expect(result.isSignificant == false)
    }

    // MARK: ANOVAResult

    @Test("ANOVAResult description formats F-statistic and p-value correctly")
    func testANOVAResultDescription() {
        let result = ANOVAResult(
            fStatistic: 12.3456,
            pValue: 0.0012,
            dfBetween: 2,
            dfWithin: 57,
            etaSquared: 0.3021
        )
        let desc = result.description
        #expect(desc.contains("12.3456"))
        #expect(desc.contains("0.0012"))
        #expect(desc.contains("0.3021"))
        #expect(desc.contains("2"))
        #expect(desc.contains("57"))
    }

    @Test("ANOVAResult isSignificant when p < 0.05")
    func testANOVAIsSignificant() {
        let result = ANOVAResult(fStatistic: 10.0, pValue: 0.001, dfBetween: 2, dfWithin: 30, etaSquared: 0.4)
        #expect(result.isSignificant == true)
    }

    @Test("ANOVAResult isNotSignificant when p >= 0.05")
    func testANOVAIsNotSignificant() {
        let result = ANOVAResult(fStatistic: 1.5, pValue: 0.2, dfBetween: 2, dfWithin: 30, etaSquared: 0.05)
        #expect(result.isSignificant == false)
    }

    // MARK: ChiSquareResult

    @Test("ChiSquareResult description formats statistic and p-value")
    func testChiSquareResultDescription() {
        let result = ChiSquareResult(statistic: 9.876, pValue: 0.0421, degreesOfFreedom: 3)
        let desc = result.description
        #expect(desc.contains("9.8760"))
        #expect(desc.contains("0.0421"))
        #expect(desc.contains("3"))
    }

    @Test("ChiSquareResult isSignificant when p < 0.05")
    func testChiSquareIsSignificant() {
        let result = ChiSquareResult(statistic: 10.0, pValue: 0.04, degreesOfFreedom: 2)
        #expect(result.isSignificant == true)
    }

    @Test("ChiSquareResult isNotSignificant when p >= 0.05")
    func testChiSquareIsNotSignificant() {
        let result = ChiSquareResult(statistic: 2.0, pValue: 0.37, degreesOfFreedom: 2)
        #expect(result.isSignificant == false)
    }

    // MARK: NormalityTestResult

    @Test("NormalityTestResult isNormal when p >= 0.05")
    func testNormalityIsNormal() {
        let result = NormalityTestResult(statistic: 0.95, pValue: 0.15)
        #expect(result.isNormal == true)
    }

    @Test("NormalityTestResult isNotNormal when p < 0.05")
    func testNormalityIsNotNormal() {
        let result = NormalityTestResult(statistic: 0.80, pValue: 0.03)
        #expect(result.isNormal == false)
    }

    @Test("NormalityTestResult description contains W and p")
    func testNormalityResultDescription() {
        let result = NormalityTestResult(statistic: 0.9234, pValue: 0.1567)
        let desc = result.description
        #expect(desc.contains("0.9234"))
        #expect(desc.contains("0.1567"))
        #expect(desc.contains("normal"))
    }

    @Test("NormalityTestResult description shows 'not normal' when not normal")
    func testNormalityDescriptionNotNormal() {
        let result = NormalityTestResult(statistic: 0.70, pValue: 0.01)
        #expect(result.description.contains("not normal"))
    }

    // MARK: ConfidenceInterval

    @Test("ConfidenceInterval description formats bounds")
    func testConfidenceIntervalDescription() {
        let ci = ConfidenceInterval(lower: 1.2345, upper: 3.6789, confidence: 0.95)
        let desc = ci.description
        #expect(desc.contains("1.2345"))
        #expect(desc.contains("3.6789"))
    }

    @Test("ConfidenceInterval stores all fields")
    func testConfidenceIntervalFields() {
        let ci = ConfidenceInterval(lower: -1.0, upper: 2.5, confidence: 0.99)
        #expect(ci.lower == -1.0)
        #expect(ci.upper == 2.5)
        #expect(ci.confidence == 0.99)
    }

    // MARK: NormOrder

    @Test("NormOrder cases exist")
    func testNormOrderCases() {
        let l1 = NormOrder.l1
        let l2 = NormOrder.l2
        let inf = NormOrder.infinity
        // Just ensure they compile and are distinct
        #expect("\(l1)" != "\(l2)")
        #expect("\(l2)" != "\(inf)")
    }
}
