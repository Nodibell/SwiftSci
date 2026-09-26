import Testing
import SwiftDataFrame

@Suite("Numeric filter comparison semantics")
struct NumericFilterDispatchTests {
    private func check<T: SupportedType>(
        _ values: [T?],
        _ cases: [(FilterCondition, [Int])]
    ) throws {
        let frame = try DataFrame(columns: [
            TypedColumn<T>(name: "value", values: values),
            TypedColumn<Int>(name: "row", values: Array(values.indices))
        ])
        for (condition, expected) in cases {
            let results = [
                try frame.filter(column: "value", where: condition),
                try frame.filterFast(column: "value", where: condition)
            ]
            for result in results {
                #expect(result.rowCount == expected.count)
                if !expected.isEmpty {
                    let rows = try #require(result[column: "row", as: Int.self])
                    #expect(rows.values == expected.map { Optional($0) })
                }
            }
        }
    }

    private func ordinaryComparisons<T: SupportedType>(_ values: [T?]) throws {
        try check(values, [
            (.equals(Int64(2)), [3, 4]),
            (.notEquals(Int64(2)), [0, 2]),
            (.greaterThan(Int64(2)), []),
            (.lessThan(Int64(2)), [0, 2]),
            (.greaterThanOrEqual(Int64(2)), [3, 4]),
            (.lessThanOrEqual(Int64(2)), [0, 2, 3, 4]),
            (.isNull, [1]),
            (.isNotNull, [0, 2, 3, 4])
        ])
    }

    @Test func numericWidthsPreserveRowsAndNulls() throws {
        try ordinaryComparisons([-2, nil, 0, 2, 2] as [Float?])
        try ordinaryComparisons([-2, nil, 0, 2, 2] as [Int32?])
        try ordinaryComparisons([-2, nil, 0, 2, 2] as [Int?])
    }

    @Test func floatDoesNotRoundDoubleThresholdDown() throws {
        let threshold = 1.0 + 1.0 / 33_554_432.0
        try check([Float(1), Float(1).nextUp, nil], [
            (.equals(threshold), []),
            (.notEquals(threshold), [0, 1]),
            (.greaterThan(threshold), [1]),
            (.lessThan(threshold), [0]),
            (.greaterThanOrEqual(threshold), [1]),
            (.lessThanOrEqual(threshold), [0])
        ])
    }

    @Test func floatNaNAndSignedZeroFollowIEEEComparisons() throws {
        let values: [Float?] = [.nan, -.infinity, -0.0, 0.0, .infinity, nil]
        try check(values, [
            (.equals(0.0), [2, 3]),
            (.notEquals(0.0), [0, 1, 4]),
            (.greaterThan(0.0), [4]),
            (.lessThan(0.0), [1]),
            (.greaterThanOrEqual(0.0), [2, 3, 4]),
            (.lessThanOrEqual(0.0), [1, 2, 3]),
            (.equals(Double.nan), []),
            (.notEquals(Double.nan), [0, 1, 2, 3, 4]),
            (.greaterThan(Double.nan), []),
            (.lessThan(Double.nan), []),
            (.greaterThanOrEqual(Double.nan), []),
            (.lessThanOrEqual(Double.nan), [])
        ])
    }

    private func fractionalThresholds<T: SupportedType>(_ values: [T?]) throws {
        try check(values, [
            (.equals(-0.5), []),
            (.notEquals(-0.5), [0, 1, 2, 3, 4]),
            (.greaterThan(-0.5), [2, 3, 4]),
            (.lessThan(-0.5), [0, 1]),
            (.greaterThanOrEqual(-0.5), [2, 3, 4]),
            (.lessThanOrEqual(-0.5), [0, 1]),
            (.equals(0.5), []),
            (.notEquals(0.5), [0, 1, 2, 3, 4]),
            (.greaterThan(0.5), [3, 4]),
            (.lessThan(0.5), [0, 1, 2]),
            (.greaterThanOrEqual(0.5), [3, 4]),
            (.lessThanOrEqual(0.5), [0, 1, 2])
        ])
    }

    @Test func integerComparisonsDoNotTruncateFractionalThresholds() throws {
        try fractionalThresholds([-2, -1, 0, 1, 2, nil] as [Int32?])
        try fractionalThresholds([-2, -1, 0, 1, 2, nil] as [Int64?])
        try fractionalThresholds([-2, -1, 0, 1, 2, nil] as [Int?])
    }

    private func largeIntegerComparisons<T: SupportedType>(_ values: [T?]) throws {
        let threshold = 9_007_199_254_740_992.0
        try check(values, [
            (.equals(threshold), [0]),
            (.notEquals(threshold), [1, 2, 4, 5]),
            (.greaterThan(threshold), [1, 2]),
            (.lessThan(threshold), [4, 5]),
            (.greaterThanOrEqual(threshold), [0, 1, 2]),
            (.lessThanOrEqual(threshold), [0, 4, 5]),
            (.equals(-threshold), [4]),
            (.notEquals(-threshold), [0, 1, 2, 5]),
            (.greaterThan(-threshold), [0, 1, 2]),
            (.lessThan(-threshold), [5]),
            (.greaterThanOrEqual(-threshold), [0, 1, 2, 4]),
            (.lessThanOrEqual(-threshold), [4, 5])
        ])
    }

    @Test func integerComparisonsPreserveBitsBeyondDoublePrecision() throws {
        let p = 9_007_199_254_740_992
        try largeIntegerComparisons([p, p + 1, p + 2, nil, -p, -p - 1] as [Int?])
        let q = Int64(p)
        try largeIntegerComparisons([q, q + 1, q + 2, nil, -q, -q - 1])
    }

    @Test func floatingColumnsPreserveInexactIntegerThresholds() throws {
        let threshold: Int64 = 9_007_199_254_740_993
        let cases: [(FilterCondition, [Int])] = [
            (.equals(threshold), []),
            (.notEquals(threshold), [0, 1, 3, 4]),
            (.greaterThan(threshold), [1]),
            (.lessThan(threshold), [0, 3, 4]),
            (.greaterThanOrEqual(threshold), [1]),
            (.lessThanOrEqual(threshold), [0, 3, 4]),
            (.equals(-threshold), []),
            (.notEquals(-threshold), [0, 1, 3, 4]),
            (.greaterThan(-threshold), [0, 1, 3]),
            (.lessThan(-threshold), [4]),
            (.greaterThanOrEqual(-threshold), [0, 1, 3]),
            (.lessThanOrEqual(-threshold), [4])
        ]
        try check([
            9_007_199_254_740_992, 9_007_199_254_740_994, nil,
            -9_007_199_254_740_992, -9_007_199_254_740_994
        ] as [Double?], cases)
        let positive: Float = 9_007_199_254_740_992
        try check([positive, positive.nextUp, nil, -positive, -positive.nextUp], cases)

        let upperCases: [(FilterCondition, [Int])] = [
            (.equals(Int64.max), []),
            (.notEquals(Int64.max), [0, 1, 3]),
            (.greaterThan(Int64.max), [0]),
            (.lessThan(Int64.max), [1, 3]),
            (.greaterThanOrEqual(Int64.max), [0]),
            (.lessThanOrEqual(Int64.max), [1, 3])
        ]
        let upperDouble = 9_223_372_036_854_775_808.0
        try check([upperDouble, upperDouble.nextDown, nil, -upperDouble], upperCases)
        let upperFloat: Float = 9_223_372_036_854_775_808
        try check([upperFloat, upperFloat.nextDown, nil, -upperFloat], upperCases)
    }

    @Test func int32AcceptsThresholdOutsideItsOwnRange() throws {
        let values: [Int32?] = [.min, -1, 0, .max, nil]
        let threshold = Int64(2_147_483_648)
        try check(values, [
            (.equals(threshold), []),
            (.notEquals(threshold), [0, 1, 2, 3]),
            (.greaterThan(threshold), []),
            (.lessThan(threshold), [0, 1, 2, 3]),
            (.greaterThanOrEqual(threshold), []),
            (.lessThanOrEqual(threshold), [0, 1, 2, 3]),
            (.greaterThan(Int64(-2_147_483_649)), [0, 1, 2, 3])
        ])
    }

    @Test func int64LimitsCompareExactlyToFloatingBoundaries() throws {
        let values: [Int64?] = [.min, .min + 1, -1, 0, .max - 1, .max, nil]
        let upper = 9_223_372_036_854_775_808.0
        try check(values, [
            (.equals(upper), []),
            (.notEquals(upper), [0, 1, 2, 3, 4, 5]),
            (.greaterThan(upper), []),
            (.lessThan(upper), [0, 1, 2, 3, 4, 5]),
            (.greaterThanOrEqual(upper), []),
            (.lessThanOrEqual(upper), [0, 1, 2, 3, 4, 5]),
            (.equals(-upper), [0]),
            (.notEquals(-upper), [1, 2, 3, 4, 5]),
            (.greaterThan(-upper), [1, 2, 3, 4, 5]),
            (.lessThan(-upper), []),
            (.greaterThanOrEqual(-upper), [0, 1, 2, 3, 4, 5]),
            (.lessThanOrEqual(-upper), [0])
        ])
    }

    @Test func integerComparisonsHandleNonFiniteThresholds() throws {
        let values: [Int64?] = [.min, 0, .max, nil]
        try check(values, [
            (.equals(Double.nan), []),
            (.notEquals(Double.nan), [0, 1, 2]),
            (.greaterThan(Double.nan), []),
            (.lessThan(Double.nan), []),
            (.greaterThanOrEqual(Double.nan), []),
            (.lessThanOrEqual(Double.nan), []),
            (.equals(Double.infinity), []),
            (.notEquals(Double.infinity), [0, 1, 2]),
            (.greaterThan(Double.infinity), []),
            (.lessThan(Double.infinity), [0, 1, 2]),
            (.greaterThanOrEqual(Double.infinity), []),
            (.lessThanOrEqual(Double.infinity), [0, 1, 2]),
            (.equals(-Double.infinity), []),
            (.notEquals(-Double.infinity), [0, 1, 2]),
            (.greaterThan(-Double.infinity), [0, 1, 2]),
            (.lessThan(-Double.infinity), []),
            (.greaterThanOrEqual(-Double.infinity), [0, 1, 2]),
            (.lessThanOrEqual(-Double.infinity), [])
        ])
    }

    @Test func emptyAndAllNullInputsRemainEmptyForComparisons() throws {
        let cases: [(FilterCondition, [Int])] = [
            (.equals(0.0), []), (.notEquals(0.0), []),
            (.greaterThan(0.0), []), (.lessThan(0.0), []),
            (.greaterThanOrEqual(0.0), []), (.lessThanOrEqual(0.0), [])
        ]
        try check([] as [Float?], cases)
        try check([nil, nil] as [Float?], cases)
        try check([] as [Int?], cases)
        try check([nil, nil] as [Int32?], cases)
    }
}
