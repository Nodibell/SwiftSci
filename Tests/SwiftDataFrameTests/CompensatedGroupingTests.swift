import Testing
import SwiftDataFrame

@Suite("Compensated grouped sums and means")
struct CompensatedGroupingTests {
    @Test(arguments: [
        [1e16, 1, 1, -1e16],
        [1, 1e16, 1, -1e16],
        [1e16, 1, -1e16, 1],
        [-1e16, 1, 1, 1e16]
    ])
    func cancellation(values: [Double]) throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Int64>(name: "key", values: Array(repeating: 0, count: values.count)),
            TypedColumn<Double>(name: "value", values: values.map { Optional($0) })])
        let group = frame.groupBy("key")
        #expect(group.sum()[column: "value", as: Double.self]?.values == [2])
        #expect(group.mean()[column: "value", as: Double.self]?.values == [0.5])
        #expect(group.agg(["value": .sum])[column: "value_sum", as: Double.self]?.values == [2])
        #expect(group.transform(["value": .mean])[column: "value_group_mean", as: Double.self]?.values == [0.5, 0.5, 0.5, 0.5])
    }

    @Test func floatValuesAndMissingValues() throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Int64>(name: "key", values: [0, 1_000_000, 0, 1_000_000, 0, 0]),
            TypedColumn<Float>(name: "value", values: [1e30, nil, 1, nil, nil, -1e30])])
        #expect(frame.groupBy("key").sum()[column: "value", as: Double.self]?.values == [1, nil])
        #expect(frame.groupBy("key").mean()[column: "value", as: Double.self]?.values == [1.0 / 3, nil])
    }

    @Test func repeatedSmallContributions() throws {
        let values: [Double?] = [1e16] + Array(repeating: 0.25, count: 4096) + [-1e16]
        let frame = try DataFrame(columns: [
            TypedColumn<Int64>(name: "key", values: Array(repeating: 0, count: values.count)),
            TypedColumn<Double>(name: "value", values: values)])
        #expect(frame.groupBy("key").sum()[column: "value", as: Double.self]?.values == [1024])
        #expect(frame.groupBy("key").mean()[column: "value", as: Double.self]?.values == [1024.0 / 4098])
    }

    @Test(arguments: [
        [Double.infinity, 1, 2],
        [1, Double.infinity, 2],
        [Double.infinity, Double.infinity, 2],
        [Double.greatestFiniteMagnitude, Double.greatestFiniteMagnitude, 1]
    ])
    func positiveInfinityStaysInfinite(values: [Double]) throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Int64>(name: "key", values: [0, 0, 0]),
            TypedColumn<Double>(name: "value", values: values.map { Optional($0) })])
        #expect(frame.groupBy("key").sum()[column: "value", as: Double.self]?.values == [.infinity])
        #expect(frame.groupBy("key").mean()[column: "value", as: Double.self]?.values == [.infinity])
    }

    @Test func nonfiniteGroupsStayIndependent() throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Int64>(name: "key", values: [0, 1, 2, 3, 0, 1, 2, 3, 3]),
            TypedColumn<Double>(name: "value", values: [-.infinity, .infinity, .nan, 1e16, -1, -.infinity, 1, 1, -1e16])])
        let result = try #require(frame.groupBy("key").sum()[column: "value", as: Double.self])
        #expect(result.values[0] == -.infinity)
        #expect(result.values[1]?.isNaN == true)
        #expect(result.values[2]?.isNaN == true)
        #expect(result.values[3] == 1)
    }

    @Test func emptyAndSubnormalValues() throws {
        let empty = try DataFrame(columns: [
            TypedColumn<Int64>(name: "key", values: []),
            TypedColumn<Double>(name: "value", values: [])])
        #expect(empty.groupBy("key").sum().rowCount == 0)
        let tiny = Double.leastNonzeroMagnitude
        let frame = try DataFrame(columns: [
            TypedColumn<Int64>(name: "key", values: [0, 0, 0]),
            TypedColumn<Double>(name: "value", values: [1, tiny, -1])])
        #expect(frame.groupBy("key").sum()[column: "value", as: Double.self]?.values == [tiny])
    }
}
