import Testing
@testable import SwiftDataFrame

@Suite("Compact grouping numerical parity")
struct CompactGroupingTests {
    @Test func compactKeysPreserveFirstSeenGroups() throws {
        let keys: [Double?] = [-0.0, 0.0, .nan, Double(bitPattern: 0x7ff8000000000001), nil, .infinity, -.infinity, nil]
        let key = dense(keys, name: "key")
        let kind = TypedColumn<Bool>(name: "kind", values: Array(repeating: true, count: keys.count))
        let values = TypedColumn<Double>(name: "value", values: [1, 2, 3, 4, 5, 6, 7, 8])
        let frame = try DataFrame(columns: [key, kind, values])
        for result in [frame.groupBy("key").sum(), frame.groupBy("key", "kind").sum()] {
            #expect(result[column: "value", as: Double.self]?.values == [3, 7, 13, 6, 7])
            #expect(result[column: "key", as: String.self]?.values.first == "-0.0")
        }
    }

    @Test(arguments: [Aggregation.sum, .mean, .min, .max, .first, .last, .count])
    func compactPayloadMatchesOptionalAggregation(aggregation: Aggregation) throws {
        let labels: [String] = ["cancel", "cancel", "cancel", "null", "null", "ieee", "ieee", "infinity", "infinity"]
        let values: [Double?] = [1e16, 1, -1e16, nil, nil, .nan, 2, .infinity, -.infinity]
        let key = TypedColumn<String>(name: "key", values: labels)
        let compact = try DataFrame(columns: [key, dense(values, name: "value")])
        let optional = try DataFrame(columns: [key, TypedColumn<Double>(name: "value", values: values)])
        let actual = compact.groupBy("key").agg(["value": aggregation])
        let expected = optional.groupBy("key").agg(["value": aggregation])
        #expect(actual.columnNames == expected.columnNames)
        let name = try #require(actual.columnNames.last)
        let lhs = try #require(actual[column: name, as: Double.self])
        let rhs = try #require(expected[column: name, as: Double.self])
        #expect(lhs.count == rhs.count)
        for row in 0..<lhs.count {
            if let a = lhs[row], let b = rhs[row] {
                #expect(a.bitPattern == b.bitPattern || (a.isNaN && b.isNaN))
            } else { #expect(lhs[row] == nil && rhs[row] == nil) }
        }
        if aggregation == .sum {
            #expect(lhs[0] == 1)
            #expect(lhs[1] == nil)
            #expect(lhs[2]?.isNaN == true && lhs[3]?.isNaN == true)
        }
    }

    private func dense(_ values: [Double?], name: String) -> TypedColumn<Double> {
        var builder = CSVCompactDoubleBuilder(capacity: values.count)
        for value in values { builder.append(value) }
        return builder.column(named: name)
    }
}
