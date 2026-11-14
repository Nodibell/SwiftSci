import Foundation
import Testing
import SwiftDataFrame
@testable import SwiftStats

struct CorrectnessRegressionStatsTests {
    @Test func chiSquareClosedForms() {
        for x in [0.1, 2.0, 6.0, 80.0, 200.0] {
            let expected = exp(-x / 2)
            #expect(abs(Stats.chiSquareSurvival(chi2: x, df: 2) / expected - 1) < 1e-11)
        }
        #expect(abs(Stats.regularisedIncompleteGamma(a: 1, x: 3) - (1 - exp(-3))) < 1e-12)
    }
    @Test func shapiro3() throws {
        let result = try Stats.shapiroWilk([0.0, 1.13, 4.26])
        #expect(abs(result.statistic - 0.931557009588863) < 2e-7)
        #expect(abs(result.pValue - 0.4944673939076983) < 2e-6)
    }
    @Test func shapiro4() throws {
        let result = try Stats.shapiroWilk([0.0, 1.13, 4.26, 9.39])
        #expect(abs(result.statistic - 0.9179533298759962) < 2e-7)
        #expect(abs(result.pValue - 0.5255353974781583) < 2e-6)
    }
    @Test func shapiro5() throws {
        let result = try Stats.shapiroWilk([0.0, 1.13, 4.26, 9.39, 16.52])
        #expect(abs(result.statistic - 0.9119138861470799) < 2e-7)
        #expect(abs(result.pValue - 0.4791767767507542) < 2e-6)
    }
    @Test func shapiro6() throws {
        let result = try Stats.shapiroWilk([0.0, 1.13, 4.26, 9.39, 16.52, 8.65])
        #expect(abs(result.statistic - 0.9382369167753445) < 2e-7)
        #expect(abs(result.pValue - 0.6450273010870822) < 2e-6)
    }
    @Test func shapiro11() throws {
        let result = try Stats.shapiroWilk([0.0, 1.13, 4.26, 9.39, 16.52, 8.65, 2.7800000000000002, 15.91, 14.04, 14.17, 16.3])
        #expect(abs(result.statistic - 0.879166734142718) < 2e-7)
        #expect(abs(result.pValue - 0.10154013543075664) < 2e-6)
    }
    @Test func shapiro12() throws {
        let result = try Stats.shapiroWilk([0.0, 1.13, 4.26, 9.39, 16.52, 8.65, 2.7800000000000002, 15.91, 14.04, 14.17, 16.3, 3.43])
        #expect(abs(result.statistic - 0.8811686855683523) < 2e-7)
        #expect(abs(result.pValue - 0.09072191488127236) < 2e-6)
    }
    @Test func shapiro30() throws {
        let result = try Stats.shapiroWilk([0.0, 1.13, 4.26, 9.39, 16.52, 8.65, 2.7800000000000002, 15.91, 14.04, 14.17, 16.3, 3.43, 9.56, 17.69, 10.82, 5.95, 3.08, 2.21, 3.34, 6.470000000000001, 11.6, 18.73, 10.86, 4.99, 18.12, 16.25, 16.38, 18.51, 5.640000000000001, 11.77])
        #expect(abs(result.statistic - 0.9256939636533592) < 2e-7)
        #expect(abs(result.pValue - 0.037779988703926626) < 2e-6)
    }
    @Test func shapiro100() throws {
        let result = try Stats.shapiroWilk([0.0, 1.13, 4.26, 9.39, 16.52, 8.65, 2.7800000000000002, 15.91, 14.04, 14.17, 16.3, 3.43, 9.56, 17.69, 10.82, 5.95, 3.08, 2.21, 3.34, 6.470000000000001, 11.6, 18.73, 10.86, 4.99, 18.12, 16.25, 16.38, 18.51, 5.640000000000001, 11.77, 19.9, 13.030000000000001, 8.16, 5.29, 4.42, 5.55, 8.68, 13.81, 20.94, 13.07, 7.2, 20.33, 18.46, 18.59, 20.72, 7.8500000000000005, 13.98, 22.11, 15.24, 10.370000000000001, 7.5, 6.63, 7.76, 10.89, 16.02, 23.15, 15.280000000000001, 9.41, 22.54, 20.67, 20.8, 22.93, 10.06, 16.189999999999998, 24.32, 17.450000000000003, 12.58, 9.71, 8.84, 9.97, 13.1, 18.23, 25.36, 17.490000000000002, 11.620000000000001, 24.75, 22.880000000000003, 23.009999999999998, 25.14, 12.27, 18.4, 26.53, 19.66, 14.790000000000001, 11.92, 11.05, 12.18, 15.31, 20.44, 27.57, 19.700000000000003, 13.83, 26.96, 25.09, 25.22, 27.35, 14.48, 20.61, 28.740000000000002, 21.87])
        #expect(abs(result.statistic - 0.9828193102078345) < 2e-7)
        #expect(abs(result.pValue - 0.21900603835203736) < 2e-6)
    }
    @Test func lillieforsRejectsSkewedSample() throws {
        let result = try Stats.kolmogorovSmirnov([0, 0, 0, 0, 0, 1, 1, 1, 2, 10])
        #expect(abs(result.statistic - 0.3648110483523105) < 1e-12)
        #expect(result.pValue < 0.01)
    }
    @Test func invalidNormalityInputs() {
        #expect(throws: (any Error).self) { try Stats.shapiroWilk([1, 1, 1]) }
        #expect(throws: (any Error).self) { try Stats.shapiroWilk([1, 2, .infinity]) }
        #expect(throws: (any Error).self) { try Stats.kolmogorovSmirnov([1, 1, 1, 1, 1]) }
        #expect(throws: (any Error).self) { try Stats.kolmogorovSmirnov([1, 2, 3, 4, .infinity]) }
        #expect(throws: (any Error).self) { try Stats.kolmogorovSmirnov([1, 2, 3, 4]) }
    }
    @Test func shapiro500() throws {
        let values = (0..<500).map { Double(($0 * $0) % 17) + Double($0) * 0.13 }
        let result = try Stats.shapiroWilk(values)
        #expect(abs(result.statistic - 0.9745164588433908) < 2e-7)
        #expect(abs(result.pValue / 1.2035663636213464e-07 - 1) < 0.001)
    }
    @Test func shapiro5000() throws {
        let values = (0..<5000).map { Double(($0 * $0) % 17) + Double($0) * 0.13 }
        let result = try Stats.shapiroWilk(values)
        #expect(abs(result.statistic - 0.9555109318958607) < 2e-7)
        #expect(abs(result.pValue / 1.4746770400765329e-36 - 1) < 0.001)
    }
    @Test func chiSquareNonIntegerReference() {
        #expect(abs(Stats.chiSquareSurvival(chi2: 0.1, df: 1) - 0.7518296340458492) < 1e-12)
        #expect(abs(Stats.chiSquareSurvival(chi2: 6, df: 1) - 0.014305878435429645) < 1e-12)
        #expect(abs(Stats.chiSquareSurvival(chi2: 20, df: 5) - 0.0012497305630313773) < 1e-12)
        #expect(abs(Stats.chiSquareSurvival(chi2: 3, df: 7.5) - 0.9125687876353294) < 1e-12)
        #expect(abs(Stats.chiSquareSurvival(chi2: 30, df: 7.5) - 0.00014301587882904854) < 1e-12)
        #expect(abs(Stats.chiSquareSurvival(chi2: 100, df: 100) - 0.48119168452795674) < 1e-12)
        #expect(abs(Stats.chiSquareSurvival(chi2: 1000, df: 1000) - 0.49405285382923964) < 1e-12)
        #expect(abs(Stats.chiSquareSurvival(chi2: .infinity, df: 2) - 0.0) < 1e-12)
    }

    @Test func chiSquareExtremeTailsAndLargeShapeParameters() {
        // Sub-machine-epsilon upper tails evaluated via direct CF without 1 - P cancellation
        let pSub1 = Stats.chiSquareSurvival(chi2: 1400, df: 2)
        #expect(pSub1 > 0 && abs(pSub1 / 9.85967654375939032e-305 - 1) < 1e-12)
        let pSub2 = Stats.chiSquareSurvival(chi2: 1400, df: 10)
        #expect(pSub2 > 0 && abs(pSub2 / 9.92039147980045937e-295 - 1) < 1e-12)

        // Large df >= 2e5 (a >= 1e5) where naive 10_000-term series truncates badly
        let p1e7Center = Stats.chiSquareSurvival(chi2: 1e7, df: 1e7)
        #expect(abs(p1e7Center - 0.499940529196062189) < 1e-12)

        let p1e7Upper = Stats.chiSquareSurvival(chi2: 1e7 + 6.0 * sqrt(2e7), df: 1e7)
        #expect(abs(p1e7Upper / 1.01873743176101734e-09 - 1) < 1e-10)

        let cdf1e7Lower = Stats.regularisedIncompleteGamma(a: 5e6, x: (1e7 - 6.0 * sqrt(2e7)) / 2.0)
        // Verified against 60-digit Decimal series + Stirling reference (scipy.special.gammainc loses ~0.27% at a=5e6 lower tail)
        #expect(abs(cdf1e7Lower / 9.5532861671054559e-10 - 1) < 1e-11)

        let p2e9Center = Stats.chiSquareSurvival(chi2: 2e9, df: 2e9)
        #expect(abs(p2e9Center - 0.499995794779129921) < 1e-11)

        // Boundary / invalid gamma & chi-square arguments
        #expect(Stats.chiSquareSurvival(chi2: 0, df: 2) == 1.0)
        #expect(Stats.chiSquareSurvival(chi2: -1, df: 2) == 1.0)
        #expect(Stats.chiSquareSurvival(chi2: 1, df: 0).isNaN)
        #expect(Stats.chiSquareSurvival(chi2: 1, df: .infinity).isNaN)
        #expect(Stats.chiSquareSurvival(chi2: .nan, df: 2).isNaN)
        #expect(Stats.regularisedIncompleteGamma(a: 0, x: 1).isNaN)
        #expect(Stats.regularisedIncompleteGamma(a: 2, x: .nan).isNaN)
        #expect(Stats.regularisedIncompleteGamma(a: 2, x: .infinity) == 1.0)
    }

    @Test func chiSquareGoodnessOfFitValidationAndScaling() throws {
        let fit = try Stats.chiSquareGoodnessOfFit(observed: [20, 30, 50], expected: [25, 25, 50])
        #expect(fit.degreesOfFreedom == 2)
        #expect(abs(fit.statistic - 2.0) < 1e-12)
        #expect(abs(fit.pValue - 0.36787944117144245) < 1e-12)

        // Large-magnitude frequency scaling: avoids intermediate pow(o - e, 2) overflow
        let scaled = try Stats.chiSquareGoodnessOfFit(observed: [2e200, 1e200], expected: [1e200, 1e200])
        #expect(abs(scaled.statistic / 1e200 - 1) < 1e-12)
        #expect(scaled.pValue == 0.0)

        #expect(throws: StatsError.insufficientData(minimum: 2, got: 1)) {
            try Stats.chiSquareGoodnessOfFit(observed: [5], expected: [5])
        }
        #expect(throws: StatsError.containsNaN) {
            try Stats.chiSquareGoodnessOfFit(observed: [1, .nan], expected: [1, 1])
        }
        #expect(throws: StatsError.containsInfinity) {
            try Stats.chiSquareGoodnessOfFit(observed: [1, 1], expected: [1, .infinity])
        }
        #expect(throws: StatsError.invalidInput("chiSquare observed frequency must be non-negative")) {
            try Stats.chiSquareGoodnessOfFit(observed: [-1, 5], expected: [2, 2])
        }
        #expect(throws: StatsError.divisionByZero(context: "chiSquare expected=0")) {
            try Stats.chiSquareGoodnessOfFit(observed: [1, 1], expected: [1, 0])
        }
    }

    @Test func normalityExtremeScaleAndBoundaryHandling() throws {
        // Large offset invariance
        let base = [0.0, 1.13, 4.26, 9.39, 16.52, 8.65, 2.78, 15.91, 14.04, 14.17, 16.3, 3.43]
        let shifted = base.map { 1e12 + $0 }
        let swShifted = try Stats.shapiroWilk(shifted)
        #expect(abs(swShifted.statistic - 0.881168610179303) < 2e-6)
        #expect(abs(swShifted.pValue - 0.09072171295050327) < 2e-5)

        // Extreme magnitude without square overflow in numerator/denominator or KS variance
        let huge = [-1e200, -0.3e200, 0.3e200, 1e200]
        let swHuge = try Stats.shapiroWilk(huge)
        #expect(abs(swHuge.statistic - 0.9970872391283413) < 2e-7)
        #expect(abs(swHuge.pValue - 0.9901843527338619) < 2e-6)

        let extreme308 = [-1e308, -0.3e308, 0.3e308, 1e308]
        let swExtreme = try Stats.shapiroWilk(extreme308)
        #expect(abs(swExtreme.statistic - swHuge.statistic) < 1e-12)
        #expect(swExtreme.pValue.isFinite)

        let ksHuge = try Stats.kolmogorovSmirnov([-1e308, -0.5e308, 0.0, 0.5e308, 1e308])
        #expect(ksHuge.statistic.isFinite && ksHuge.pValue.isFinite && ksHuge.pValue > 0.1)

        let swSubnormal = try Stats.shapiroWilk([1e-320, 2e-320, 3e-320, 5e-320])
        #expect(swSubnormal.statistic.isFinite && swSubnormal.pValue.isFinite)

        #expect(throws: StatsError.self) { try Stats.shapiroWilk([1, 2, .nan]) }
        #expect(throws: StatsError.self) { try Stats.kolmogorovSmirnov([1, 2, 3, 4, .nan]) }
        #expect(throws: StatsError.self) { try Stats.shapiroWilk(Array(repeating: 1.0, count: 5001).enumerated().map { Double($0.offset) }) }
    }
}
