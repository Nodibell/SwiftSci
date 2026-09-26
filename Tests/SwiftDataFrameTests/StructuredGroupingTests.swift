import Foundation
import Testing
import SwiftDataFrame

@Suite("Structured group key equality")
struct StructuredGroupingTests {
    private func frame(_ keys: [any AnyColumn]) throws -> DataFrame {
        let count = keys.first?.count ?? 0
        return try DataFrame(columns: keys + [TypedColumn<Double>(
            name: "value", values: (1...max(count, 1)).prefix(count).map(Double.init))])
    }

    @Test func stringSentinelsAreValues() throws {
        let data = try frame([TypedColumn<String>(
            name: "key", values: [nil, "__null__", "null", "", nil, "__null__"])])
        let result = data.groupBy("key").sum()
        #expect(result[column: "key", as: String.self]?.values == [nil, "__null__", "null", ""])
        #expect(result[column: "value", as: Double.self]?.values == [6, 8, 3, 4])
    }

    @Test func tupleBoundariesAndNullsCannotCollide() throws {
        let data = try frame([
            TypedColumn<String>(name: "a", values: ["a||b", "a", nil, "null", "a||b", "a"]),
            TypedColumn<String>(name: "b", values: ["c", "b||c", "x", "x", "c", "b||c"])])
        #expect(data.groupBy("a", "b").sum()[column: "value", as: Double.self]?.values == [6, 8, 3, 4])
        #expect(data.groupBy("b", "a").sum()[column: "value", as: Double.self]?.values == [6, 8, 3, 4])
        #expect(data.groupBy("a", "b").transform(["value": .sum])[column: "value_group_sum", as: Double.self]?.values == [6, 8, 3, 4, 6, 8])
    }

    @Test func threeKeysPreserveExactIntegersAndOrder() throws {
        let high = Int64(9_007_199_254_740_992)
        let data = try frame([
            TypedColumn<Int64>(name: "a", values: [high, high + 1, high, high, high + 1]),
            TypedColumn<Int32>(name: "b", values: [1, 1, 1, 2, 1]),
            TypedColumn<String>(name: "c", values: ["x", "x", "x", "x", "x"])])
        let result = data.groupBy("a", "b", "c").sum()
        #expect(result[column: "a", as: String.self]?.values == [String(high), String(high + 1), String(high)])
        #expect(result[column: "value", as: Double.self]?.values == [4, 7, 4])
    }

    @Test func floatingKeysHaveExplicitEquality() throws {
        let values: [Double?] = [-0.0, 0.0, .nan, Double(bitPattern: 0x7ff8000000000001), nil, .infinity, -.infinity, nil]
        for key in [
            TypedColumn<Double>(name: "key", values: values) as any AnyColumn,
            TypedColumn<Float>(name: "key", values: values.map { $0.map(Float.init) })
        ] {
            let data = try frame([key, TypedColumn<Bool>(name: "kind", values: Array(repeating: true, count: values.count))])
            for result in [data.groupBy("key").sum(), data.groupBy("key", "kind").sum()] {
                #expect(result[column: "value", as: Double.self]?.values == [3, 7, 13, 6, 7])
                #expect(result[column: "key", as: String.self]?.values.first == "-0.0")
            }
        }
    }

    @Test func datesKeepSubsecondIdentityAndBoolsKeepNull() throws {
        let date = Date(timeIntervalSince1970: 1_000)
        let later = date.addingTimeInterval(0.001)
        let data = try frame([
            TypedColumn<Date>(name: "date", values: [date, later, date, date, date]),
            TypedColumn<Bool>(name: "flag", values: [true, true, true, false, nil])])
        #expect(data.groupBy("date", "flag").sum()[column: "value", as: Double.self]?.values == [4, 2, 4, 5])
        #expect(data.groupBy("date").sum()[column: "value", as: Double.self]?.values == [13, 2])
    }

    private struct CollisionKey: Hashable, CustomStringConvertible {
        let value: Int
        var description: String { "same text" }
        func hash(into hasher: inout Hasher) { hasher.combine(0) }
    }

    private struct TextKey: CustomStringConvertible {
        let value: Int
        var description: String { "text\(value)" }
    }

    @Test func erasedHashableKeysUseEqualityAfterHashCollision() throws {
        let key = DataFrameRelease35CoverageTests.CustomGenericColumn(
            name: "key", dtype: .utf8,
            rawValues: [CollisionKey(value: 2), CollisionKey(value: 1), CollisionKey(value: 2), nil])
        let data = try frame([key])
        #expect(data.groupBy("key").sum()[column: "value", as: Double.self]?.values == [4, 2, 4])
    }

    @Test func erasedKeysKeepRuntimeTypesDistinct() throws {
        let key = DataFrameRelease35CoverageTests.CustomGenericColumn(
            name: "key", dtype: .utf8,
            rawValues: [Int(1), Int64(1), Double(1), "1", true, Int(1)])
        #expect(try frame([key]).groupBy("key").sum()[column: "value", as: Double.self]?.values == [7, 2, 3, 4, 5])
    }

    @Test func erasedFloatingKeysMatchTypedRules() throws {
        let key = DataFrameRelease35CoverageTests.CustomGenericColumn(
            name: "key", dtype: .float64,
            rawValues: [-0.0, 0.0, Double.nan, Double(bitPattern: 0x7ff8000000000001), nil])
        #expect(try frame([key]).groupBy("key").sum()[column: "value", as: Double.self]?.values == [3, 7, 5])
    }

    @Test func nonHashableFallbackRemainsPerComponent() throws {
        let key = DataFrameRelease35CoverageTests.CustomGenericColumn(
            name: "key", dtype: .utf8,
            rawValues: [TextKey(value: 1), TextKey(value: 2), TextKey(value: 1), nil])
        let data = try frame([key, TypedColumn<String>(name: "kind", values: ["x", "x", "x", "text1"])])
        #expect(data.groupBy("key", "kind").sum()[column: "value", as: Double.self]?.values == [4, 2, 4])
    }
}
