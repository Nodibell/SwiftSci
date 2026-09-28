import Testing
import SwiftStats
import SwiftDataFrame

@Suite("Original-decimal ANOVA ingestion")
struct DecimalANOVATests {
    @Test("Preserves sub-ULP decimal differences at a large origin")
    func beyondBinary64() throws {
        let groups = [["10000000000000000.1", "10000000000000000.2", "10000000000000000.3"],
                      ["10000000000000000.4", "10000000000000000.5", "10000000000000000.6"]]
        let result = try Stats.oneWayANOVA(decimalGroups: groups)
        #expect(abs(result.fStatistic - 13.5) < 1e-12)
        #expect(result.dfBetween == 1 && result.dfWithin == 4)
        #expect(abs(result.etaSquared - 27.0 / 35) < 1e-14)
        #expect(throws: StatsError.self) {
            try Stats.oneWayANOVA(groups: groups.map { $0.map { Double($0)! } })
        }
    }

    @Test("Uses a common origin across unequal groups, signs and scientific notation")
    func commonOrigin() throws {
        let expected = try Stats.oneWayANOVA(groups: [[-4, -2], [1, 2, 3], [5, 8, 11, 12]])
        let actual = try Stats.oneWayANOVA(decimalGroups: [["-4e-1", "-.2"], [".1", "+0.20", "3E-1"], [".5", ".8", "1.1", "1.2"]])
        #expect(abs(actual.fStatistic - expected.fStatistic) < 1e-12)
        #expect(abs(actual.pValue - expected.pValue) < 1e-12)
        #expect(abs(actual.etaSquared - expected.etaSquared) < 1e-14)
        let reversed = try Stats.oneWayANOVA(decimalGroups: [["1.2", "1.1", ".8", ".5"], [".3", ".2", ".1"], ["-.2", "-.4"]])
        #expect(abs(reversed.fStatistic - expected.fStatistic) < 1e-12)
    }

    @Test("Rejects malformed or unsupported decimals without rounding",
          arguments: ["", " ", " 1", "1 ", "1x", "1,2", "1_000", "0x1", "NaN", "inf", "1e", "1e+", ".", "--1", "١", "1e128", "1e-129", "123456789012345678901234567890123456789", "1e999999999999999999999", "1e-9223372036854775808"])
    func invalidInput(text: String) {
        #expect(throws: StatsError.self) {
            try Stats.oneWayANOVA(decimalGroups: [["1", "2"], [text, "4"]])
        }
    }

    @Test("Accepts 38 significant digits, redundant zeros and signed zero")
    func exactLimits() throws {
        let result = try Stats.oneWayANOVA(decimalGroups: [
            ["12345678901234567890123456789012345671", "12345678901234567890123456789012345672", "12345678901234567890123456789012345673"],
            ["12345678901234567890123456789012345674", "12345678901234567890123456789012345675", "12345678901234567890123456789012345676"]])
        #expect(abs(result.fStatistic - 13.5) < 1e-12)
        let zero = try Stats.oneWayANOVA(decimalGroups: [["-0.000", "+000.1000", "0.2"], [".3", ".4", ".5"]])
        #expect(abs(zero.fStatistic - 13.5) < 1e-12)
    }

    @Test("Rejects inexact decimal subtraction")
    func subtractionRange() {
        #expect(throws: StatsError.self) {
            try Stats.oneWayANOVA(decimalGroups: [["1e127", "2e127"], ["1e-128", "2e-128"]])
        }
    }

    @Test("Preserves scale invariance at the supported exponent boundaries",
          arguments: [-128, -60, -1, 0, 60, 127])
    func exponentBounds(exponent: Int) throws {
        let groups = [[1, 2, 3], [4, 5, 6]].map { $0.map { "\($0)e\(exponent)" } }
        let result = try Stats.oneWayANOVA(decimalGroups: groups)
        #expect(abs(result.fStatistic - 13.5) < 1e-12)
    }

    @Test("Checks full-precision carries, borrows and opposite signs")
    func coefficientArithmetic() throws {
        for prefix in ["99999999999999999999999999999999999", "-99999999999999999999999999999999999"] {
            let groups = [[1, 2, 3], [4, 5, 6]].map { $0.map { prefix + String($0) } }
            #expect(abs(try Stats.oneWayANOVA(decimalGroups: groups).fStatistic - 13.5) < 1e-12)
        }
        let groups = [["-2", "-1", "0"], ["1", "2", "3"]]
        #expect(abs(try Stats.oneWayANOVA(decimalGroups: groups).fStatistic - 13.5) < 1e-12)
        #expect(throws: StatsError.self) {
            try Stats.oneWayANOVA(decimalGroups: [["99999999999999999999999999999999999999", "0"], ["-2", "-1"]])
        }
    }

    @Test("Retains the existing group and variance validation")
    func dimensions() {
        for groups: [[String]] in [[], [["1", "2"]], [[], ["1", "2"]], [["1", "1"], ["2", "2"]]] {
            #expect(throws: StatsError.self) { try Stats.oneWayANOVA(decimalGroups: groups) }
        }
    }
}
