import Testing
import SwiftDataFrame

@Suite("Exact checked integer group sums")
struct CheckedIntegerSumTests {
    @Test func integerConversionLosesContribution() throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Int64>(name: "key", values: [0, 0]),
            TypedColumn<Int64>(name: "value", values: [9_007_199_254_740_993, -9_007_199_254_740_992])])
        #expect(try frame.groupBy("key").sumChecked()[column: "value", as: Int64.self]?.values == [1])
    }

    @Test func exactLargeTotalsAndNullGroups() throws {
        let large: Int64 = 9_007_199_254_740_993
        let frame = try DataFrame(columns: [
            TypedColumn<Int64>(name: "key", values: [7, nil, 7, nil, -4, -4]),
            TypedColumn<Int64>(name: "value", values: [large, .min, 2, nil, nil, nil])])
        let result = try frame.groupBy("key").sumChecked()
        #expect(result[column: "key", as: String.self]?.values == ["7", nil, "-4"])
        #expect(result[column: "value", as: Int64.self]?.values == [large + 2, .min, nil])
    }

    @Test func temporaryOverflowCanCancel() throws {
        let cases: [([Int64], Int64)] = [
            ([.max, 1, -1], .max),
            ([.min, -1, 1], .min),
            ([.max, .max, .min], .max - 1),
            ([.min, .min, .max, .max], -2),
            ([.max, .max, .max, -.max, -.max], .max),
            ([.min, .min, .min, .max, .max, .max], -3),
            ([.max, .min, 1], 0)
        ]
        for (values, expected) in cases {
            let frame = try DataFrame(columns: [
                TypedColumn<String>(name: "key", values: Array(repeating: "a", count: values.count)),
                TypedColumn<Int64>(name: "value", values: values)])
            #expect(try frame.groupBy("key").sumChecked()[column: "value", as: Int64.self]?.values == [expected])
        }
    }

    @Test(arguments: [[Int64.max, 1], [Int64.min, -1], [Int64.max, Int64.max]])
    func finalOverflowThrows(values: [Int64]) throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Int64>(name: "key", values: [4, 99, 99]),
            TypedColumn<Int64>(name: "value", values: [5] + values)])
        #expect(throws: SwiftMLError.integerOverflow(column: "value", group: 1)) {
            try frame.groupBy("key").sumChecked()
        }
    }

    @Test func promotesIntegerWidthsAndPreservesFloatingSums() throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Int32>(name: "key", values: [0, 0, 0]),
            TypedColumn<Int32>(name: "int32", values: [.max, .max, nil]),
            TypedColumn<Int>(name: "native", values: [9_007_199_254_740_993, -9_007_199_254_740_992, nil]),
            TypedColumn<Double>(name: "double", values: [1e16, 1, -1e16]),
            TypedColumn<Float>(name: "float", values: [1e30, 1, -1e30])])
        let result = try frame.groupBy("key").sumChecked()
        #expect(result[column: "int32", as: Int64.self]?.values == [4_294_967_294])
        #expect(result[column: "native", as: Int64.self]?.values == [1])
        #expect(result[column: "double", as: Double.self]?.values == [1])
        #expect(result[column: "float", as: Double.self]?.values == [1])
        #expect(frame.groupBy("key").sum()[column: "int32", as: Double.self] != nil)
        #expect(frame.groupBy("key").agg(["int32": .sum])[column: "int32_sum", as: Double.self] != nil)
    }

    @Test func emptyAndMultiKeyGroups() throws {
        let empty = try DataFrame(columns: [
            TypedColumn<Int64>(name: "key", values: [Int64]()),
            TypedColumn<Int64>(name: "value", values: [Int64]())])
        #expect(try empty.groupBy("key").sumChecked()[column: "value", as: Int64.self]?.values == [])
        let frame = try DataFrame(columns: [
            TypedColumn<String>(name: "key", values: ["a", "a", "b"]),
            TypedColumn<Int32>(name: "kind", values: [1, 1, 1]),
            TypedColumn<Int64>(name: "value", values: [9_007_199_254_740_993, 2, nil])])
        #expect(try frame.groupBy("key", "kind").sumChecked()[column: "value", as: Int64.self]?.values == [9_007_199_254_740_995, nil])
        #expect(try frame.groupBy().sumChecked().rowCount == 0)
    }

    @Test func customIntegerStorage() throws {
        let column = DataFrameRelease35CoverageTests.CustomGenericColumn(
            name: "value", dtype: .int64,
            rawValues: [Int64(9_007_199_254_740_993), Int32(2), Int(-1), nil])
        let key = TypedColumn<Int64>(name: "key", values: [0, 0, 0, 0])
        let frame = try DataFrame(columns: [key, column])
        #expect(try frame.groupBy("key").sumChecked()[column: "value", as: Int64.self]?.values == [9_007_199_254_740_994])
        let invalid = try DataFrame(columns: [
            TypedColumn<Int64>(name: "key", values: [0]),
            DataFrameRelease35CoverageTests.CustomGenericColumn(name: "value", dtype: .int64, rawValues: [1.5])])
        #expect(throws: SwiftMLError.typeMismatch(column: "value", expected: "Int32, Int64 or Int", got: "Double")) {
            try invalid.groupBy("key").sumChecked()
        }
    }

    @Test func wideAccumulatorMatchesIndependentInt128Reference() throws {
        if #available(macOS 15.0, *) {
            var seed: UInt64 = 42
            for _ in 0..<200 {
                var values: [Int64] = []
                for _ in 0..<40 {
                    seed = seed &* 6_364_136_223_846_793_005 &+ 1
                    values.append(Int64(bitPattern: seed))
                }
                // Cancellation restores a small exact result after arbitrary wide prefixes.
                values += values.reversed().map { ~$0 }
                let expected = values.reduce(Int128(0)) { $0 + Int128($1) }
                let frame = try DataFrame(columns: [
                    TypedColumn<Int64>(name: "key", values: Array(repeating: 0, count: values.count)),
                    TypedColumn<Int64>(name: "value", values: values)])
                #expect(try frame.groupBy("key").sumChecked()[column: "value", as: Int64.self]?.values == [Int64(exactly: expected)])
                let partial = try DataFrame(columns: [
                    TypedColumn<Int64>(name: "key", values: Array(repeating: 0, count: 40)),
                    TypedColumn<Int64>(name: "value", values: Array(values.prefix(40)))])
                let partialSum = values.prefix(40).reduce(Int128(0)) { $0 + Int128($1) }
                if let expected = Int64(exactly: partialSum) {
                    #expect(try partial.groupBy("key").sumChecked()[column: "value", as: Int64.self]?.values == [expected])
                } else {
                    #expect(throws: SwiftMLError.integerOverflow(column: "value", group: 0)) {
                        try partial.groupBy("key").sumChecked()
                    }
                }
            }
        }
    }
}
