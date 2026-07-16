import Testing
import SwiftStats
import SwiftDataFrame

@Suite("Descriptive Statistics")
struct DescriptiveStatsTests {

    // MARK: – Basic

    @Test("mean of [1,2,3,4,5] == 3.0")
    func meanBasic() throws {
        let result = try Stats.mean([1.0, 2.0, 3.0, 4.0, 5.0] as [Double])
        #expect(result == 3.0)
    }

    @Test("mean empty throws")
    func meanEmptyThrows() throws {
        #expect(throws: StatsError.emptyInput) {
            try Stats.mean([] as [Double])
        }
    }

    @Test("variance [2,4,4,4,5,5,7,9] ddof=1 ≈ 4.571")
    func varianceSample() throws {
        let result = try Stats.variance([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] as [Double], ddof: 1)
        #expect(abs(result - 4.571428571) < 1e-8)
    }

    @Test("variance constant array == 0.0 (not NaN)")
    func varianceConstant() throws {
        let result = try Stats.variance([5.0, 5.0, 5.0, 5.0] as [Double], ddof: 1)
        #expect(result == 0.0)
        #expect(!result.isNaN)
    }

    @Test("median odd count")
    func medianOdd() throws {
        #expect(try Stats.median([1.0, 3.0, 5.0] as [Double]) == 3.0)
    }

    @Test("median even count")
    func medianEven() throws {
        #expect(try Stats.median([1.0, 2.0, 3.0, 4.0] as [Double]) == 2.5)
    }

    @Test("percentile q=0.0 returns min")
    func percentileMin() throws {
        let vals = [3.0, 1.0, 2.0, 5.0, 4.0]
        #expect(try Stats.percentile(vals, q: 0.0) == 1.0)
    }

    @Test("percentile q=1.0 returns max")
    func percentileMax() throws {
        let vals = [3.0, 1.0, 2.0]
        #expect(try Stats.percentile(vals, q: 1.0) == 3.0)
    }

    @Test("percentile invalid q throws")
    func percentileInvalidQ() throws {
        #expect(throws: StatsError.self) { try Stats.percentile([1.0], q: -0.1) }
        #expect(throws: StatsError.self) { try Stats.percentile([1.0], q: 1.5) }
    }

    @Test("sum matches manual")
    func sumMatches() throws {
        #expect(try Stats.sum([1.0, 2.0, 3.0, 4.0] as [Double]) == 10.0)
    }

    @Test("min and max")
    func minMax() throws {
        let vals = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]
        #expect(try Stats.min(vals) == 1.0)
        #expect(try Stats.max(vals) == 9.0)
    }

    @Test("containsNaN throws")
    func nanThrows() throws {
        #expect(throws: StatsError.containsNaN) {
            try Stats.mean([1.0, Double.nan, 3.0])
        }
    }
}

@Suite("Stats Edge Cases")
struct EdgeCaseTests {

    @Test("tTest with n<2 throws insufficientData")
    func tTestInsufficientData() throws {
        #expect(throws: StatsError.self) {
            try Stats.tTest(sample: [42.0], populationMean: 0)
        }
    }

    @Test("pearsonCorrelation size mismatch throws")
    func pearsonSizeMismatch() throws {
        #expect(throws: StatsError.self) {
            try Stats.pearsonCorrelation([1, 2, 3], [1, 2])
        }
    }

    @Test("pearsonCorrelation perfect linear == 1.0")
    func pearsonPerfectLinear() throws {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0]
        let y = x.map { $0 * 2 + 1 }
        let r = try Stats.pearsonCorrelation(x, y)
        #expect(abs(r - 1.0) < 1e-10)
    }
}

@Suite("Hypothesis Tests")
struct TTestTests {

    @Test("one-sample t-test fails to reject H0 (p > 0.05)")
    func oneSampleNotSignificant() throws {
        let sample = [2.9, 3.0, 2.5, 2.8, 3.2, 3.0]
        let result = try Stats.tTest(sample: sample, populationMean: 3.0)
        #expect(result.pValue > 0.05)
        #expect(!result.isSignificant)
    }

    @Test("one-sample t-test rejects H0 for clearly different mean (p < 0.05)")
    func oneSampleSignificant() throws {
        let sample = [10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1]
        let result = try Stats.tTest(sample: sample, populationMean: 0.0)
        #expect(result.pValue < 0.05)
    }

    @Test("paired t-test equivalent to one-sample on differences")
    func pairedEquivalentToOneSample() throws {
        let before = [5.0, 6.0, 7.0, 4.0, 8.0]
        let after  = [6.0, 7.5, 8.0, 5.0, 9.0]
        let paired = try Stats.pairedTTest(before: before, after: after)
        let diffs  = zip(before, after).map { $0.1 - $0.0 }
        let manual = try Stats.tTest(sample: diffs, populationMean: 0)
        #expect(abs(paired.statistic - manual.statistic) < 1e-10)
    }

    @Test("chi-square GoF: observed == expected → p ≈ 1.0")
    func chiSquarePerfectFit() throws {
        let obs: [Double] = [10, 20, 30]
        let result = try Stats.chiSquareGoodnessOfFit(observed: obs, expected: obs)
        #expect(result.statistic == 0.0)
        #expect(result.pValue > 0.99)
    }

    @Test("ANOVA with equal groups: F ≈ 0, p ≈ 1")
    func anovaEqualGroups() throws {
        let g = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        let r = try Stats.oneWayANOVA(groups: g)
        #expect(r.fStatistic < 1e-10)
        #expect(r.pValue > 0.99)
    }

    @Test("ANOVA with clearly different groups: p < 0.05")
    func anovaDifferentGroups() throws {
        let g = [[1.0, 1.1, 0.9], [5.0, 5.1, 4.9], [10.0, 10.1, 9.9]]
        let r = try Stats.oneWayANOVA(groups: g)
        #expect(r.isSignificant)
    }
}

@Suite("Correlation Matrix & Spearman Tests")
struct CorrelationTests {
    @Test("Spearman Rank Correlation handles linear and monotonic data")
    func testSpearman() throws {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0]
        let y = [2.0, 4.0, 6.0, 8.0, 10.0]
        let r = try Stats.spearmanCorrelation(x, y)
        #expect(abs(r - 1.0) < 1e-10)

        // Monotonic but non-linear: y = x^3
        let y2 = x.map { $0 * $0 * $0 }
        let r2 = try Stats.spearmanCorrelation(x, y2)
        #expect(abs(r2 - 1.0) < 1e-10)
    }

    @Test("correlationMatrix computes full variable correlation matrix")
    func testCorrelationMatrix() throws {
        let x = [1.0, 2.0, 3.0, 4.0]
        let y = [2.0, 4.0, 6.0, 8.0]
        let z = [4.0, 3.0, 2.0, 1.0] // perfect negative correlation with x/y
        let matrix = try Stats.correlationMatrix([x, y, z])

        #expect(matrix.count == 3)
        #expect(matrix[0].count == 3)
        #expect(abs(matrix[0][1] - 1.0) < 1e-10)
        #expect(abs(matrix[0][2] - (-1.0)) < 1e-10)
    }
}

@Suite("Normality Tests")
struct NormalityTests {
    @Test("Shapiro-Wilk normality test on normal data passes normality check")
    func testShapiroWilkNormal() throws {
        // A symmetric-ish, small sample that approximates normal
        let normalSample = [2.0, 2.5, 3.0, 3.5, 4.0]
        let result = try Stats.shapiroWilk(normalSample)
        #expect(result.pValue >= 0.05)
        #expect(result.isNormal)
    }

    @Test("Kolmogorov-Smirnov normality test")
    func testKSNormal() throws {
        let sample = [2.0, 2.5, 3.0, 3.5, 4.0]
        let result = try Stats.kolmogorovSmirnov(sample)
        #expect(result.pValue >= 0.05)
        #expect(result.isNormal)
    }
}

@Suite("Linear Algebra Tests")
struct LinearAlgebraTests {
    @Test("Vector operations and similarity")
    func testVectorOps() throws {
        let a = [1.0, 2.0, 3.0]
        let b = [4.0, 5.0, 6.0]

        let dot = try Stats.dotProduct(a, b)
        #expect(dot == 4.0 + 10.0 + 18.0) // 32.0

        let normL1 = try Stats.norm(a, order: .l1)
        #expect(normL1 == 6.0)

        let normL2 = try Stats.norm(a, order: .l2)
        #expect(abs(normL2 - 14.0.squareRoot()) < 1e-10)

        let normLinf = try Stats.norm(a, order: .infinity)
        #expect(normLinf == 3.0)

        let sim = try Stats.cosineSimilarity(a, a)
        #expect(abs(sim - 1.0) < 1e-10)

        let added = try Stats.add(a, b)
        #expect(added == [5.0, 7.0, 9.0])

        let subtracted = try Stats.subtract(a, b)
        #expect(subtracted == [3.0, 3.0, 3.0]) // b - a

        let scaled = Stats.scale(a, by: 2.0)
        #expect(scaled == [2.0, 4.0, 6.0])

        let normed = try Stats.normalise(a)
        let normedL2 = try Stats.norm(normed, order: .l2)
        #expect(abs(normedL2 - 1.0) < 1e-10)

        let stdVal = try Stats.standardise([1.0, 2.0, 3.0])
        #expect(abs(try Stats.mean(stdVal)) < 1e-10)
        #expect(abs(try Stats.standardDeviation(stdVal) - 1.0) < 1e-10)
    }
}

@Suite("DataFrame Stats Extensions")
struct DataFrameStatsTests {
    @Test("DataFrame describe and correlation extensions")
    func testDataFrameStats() throws {
        let colA = TypedColumn<Double>(name: "A", values: [1.0, 2.0, 3.0, 4.0])
        let colB = TypedColumn<Double>(name: "B", values: [2.0, 4.0, 6.0, 8.0])
        let colC = TypedColumn<String>(name: "C", values: ["x", "y", "z", "w"])
        let df = try DataFrame(columns: [colA, colB, colC])

        // stats(for:)
        let descA = try df.stats(for: "A")
        #expect(descA.mean == 2.5)

        // describe()
        let summary = try df.describe()
        #expect(summary.shape.rows == 8)
        #expect(summary.shape.columns == 3) // stat, A, B
        #expect(summary.columnNames == ["stat", "A", "B"])

        // correlationMatrix()
        let corr = try df.correlationMatrix()
        #expect(corr.shape.rows == 2)
        #expect(corr.shape.columns == 3) // variable, A, B
        #expect(corr.columnNames == ["variable", "A", "B"])
        
        let aCorrB = corr[column: "B"]?.value(at: 0) as? Double // Row 0 is variable A
        #expect(aCorrB != nil)
        #expect(abs((aCorrB ?? 0.0) - 1.0) < 1e-10)
    }
}

