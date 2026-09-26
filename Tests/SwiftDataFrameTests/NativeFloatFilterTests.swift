import Testing
import SwiftDataFrame

@Suite("Native Float filter boundaries")
struct NativeFloatFilterTests {
    private func conditions(_ rhs: Double) -> [FilterCondition] {
        [.equals(rhs), .notEquals(rhs), .lessThan(rhs), .lessThanOrEqual(rhs),
         .greaterThan(rhs), .greaterThanOrEqual(rhs)]
    }

    private func doublePredicates(_ rhs: Double) -> [(Double) -> Bool] {
        [{ $0 == rhs }, { $0 != rhs }, { $0 < rhs }, { $0 <= rhs },
         { $0 > rhs }, { $0 >= rhs }]
    }

    @Test(arguments: [false, true])
    func floatMatchesExactWideningAcrossBitPatterns(includeNulls: Bool) {
        var state: UInt32 = 0x12345678
        var values: [Float?] = [nil, .nan, -.infinity, .infinity, -0.0, 0.0,
                               .leastNonzeroMagnitude, -.leastNonzeroMagnitude,
                               .greatestFiniteMagnitude, -.greatestFiniteMagnitude]
        for _ in 0..<2048 {
            state = state &* 1664525 &+ 1013904223
            values.append(Float(bitPattern: state))
        }
        if !includeNulls { values.removeAll { $0 == nil } }
        let column = TypedColumn<Float>(name: "value", values: values)
        let boundaries: [Float] = [0, 1, -1, .leastNonzeroMagnitude,
                                   .leastNormalMagnitude, .greatestFiniteMagnitude,
                                   -.greatestFiniteMagnitude, 16_777_216]
        var thresholds: [Double] = [.nan, -.infinity, .infinity, 0.1, -0.1]
        for value in boundaries {
            let exact = Double(value)
            thresholds.append(contentsOf: [exact.nextDown, exact, exact.nextUp])
        }
        for rhs in thresholds {
            for (condition, predicate) in zip(conditions(rhs), doublePredicates(rhs)) {
                let expected = values.indices.filter { index in
                    values[index].map { predicate(Double($0)) } ?? false
                }
                #expect(column.filteredIndices(matching: condition) == expected)
            }
        }
    }

    @Test(arguments: [false, true])
    func doubleSpecialValuesKeepTheirComparisonSemantics(includeNulls: Bool) {
        var values: [Double?] = [nil, .nan, -.infinity, .infinity, -0.0, 0.0,
                                .leastNonzeroMagnitude, -.leastNonzeroMagnitude,
                                .leastNormalMagnitude, -.leastNormalMagnitude,
                                .greatestFiniteMagnitude, -.greatestFiniteMagnitude,
                                1.0.nextDown, 1.0, 1.0.nextUp]
        if !includeNulls { values.removeAll { $0 == nil } }
        let column = TypedColumn<Double>(name: "value", values: values)
        for rhs in values.compactMap({ $0 }) {
            for (condition, predicate) in zip(conditions(rhs), doublePredicates(rhs)) {
                let expected = values.indices.filter { index in
                    values[index].map(predicate) ?? false
                }
                #expect(column.filteredIndices(matching: condition) == expected)
            }
        }
    }

    private func checkMixedIntegerThreshold<T: SupportedType>(
        _ values: [T],
        cases: [(FilterCondition, [Int])]
    ) {
        let column = TypedColumn<T>(name: "value", values: values)
        #expect(column.nullCount == 0)
        for (condition, expected) in cases {
            #expect(column.filteredIndices(matching: condition) == expected)
        }
    }

    @Test func nullFreeFloatingColumnsPreserveExactIntegerThresholds() {
        let threshold: Int64 = 9_007_199_254_740_993
        let precisionCases: [(FilterCondition, [Int])] = [
            (.equals(threshold), []), (.notEquals(threshold), [0, 1, 2, 3]),
            (.lessThan(threshold), [0, 2, 3]), (.lessThanOrEqual(threshold), [0, 2, 3]),
            (.greaterThan(threshold), [1]), (.greaterThanOrEqual(threshold), [1]),
            (.equals(-threshold), []), (.notEquals(-threshold), [0, 1, 2, 3]),
            (.lessThan(-threshold), [3]), (.lessThanOrEqual(-threshold), [3]),
            (.greaterThan(-threshold), [0, 1, 2]), (.greaterThanOrEqual(-threshold), [0, 1, 2])
        ]
        let preciseDouble = 9_007_199_254_740_992.0
        checkMixedIntegerThreshold(
            [preciseDouble, preciseDouble.nextUp, -preciseDouble, -preciseDouble.nextUp],
            cases: precisionCases
        )
        let preciseFloat: Float = 9_007_199_254_740_992
        checkMixedIntegerThreshold(
            [preciseFloat, preciseFloat.nextUp, -preciseFloat, -preciseFloat.nextUp],
            cases: precisionCases
        )

        let upperCases: [(FilterCondition, [Int])] = [
            (.equals(Int64.max), []), (.notEquals(Int64.max), [0, 1, 2]),
            (.lessThan(Int64.max), [1, 2]), (.lessThanOrEqual(Int64.max), [1, 2]),
            (.greaterThan(Int64.max), [0]), (.greaterThanOrEqual(Int64.max), [0])
        ]
        let upperDouble = 9_223_372_036_854_775_808.0
        checkMixedIntegerThreshold([upperDouble, upperDouble.nextDown, -upperDouble], cases: upperCases)
        let upperFloat: Float = 9_223_372_036_854_775_808
        checkMixedIntegerThreshold([upperFloat, upperFloat.nextDown, -upperFloat], cases: upperCases)

        let lowerCases: [(FilterCondition, [Int])] = [
            (.equals(Int64.min + 1), []), (.notEquals(Int64.min + 1), [0, 1]),
            (.lessThan(Int64.min + 1), [0]), (.lessThanOrEqual(Int64.min + 1), [0]),
            (.greaterThan(Int64.min + 1), [1]), (.greaterThanOrEqual(Int64.min + 1), [1])
        ]
        let lowerDouble = -upperDouble
        checkMixedIntegerThreshold([lowerDouble, lowerDouble.nextUp], cases: lowerCases)
        let lowerFloat = -upperFloat
        checkMixedIntegerThreshold([lowerFloat, lowerFloat.nextUp], cases: lowerCases)
    }
}
