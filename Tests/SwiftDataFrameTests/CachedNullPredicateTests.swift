import Foundation
import Testing
import SwiftDataFrame

@Suite("Cached column null predicates")
struct CachedNullPredicateTests {
    private func check<T: SupportedType>(_ column: TypedColumn<T>) throws {
        let values = column.values
        let missing = values.indices.filter { values[$0] == nil }
        let present = values.indices.filter { values[$0] != nil }
        #expect(column.nullCount == missing.count)
        let frame = try DataFrame(columns: [
            column,
            TypedColumn<Int>(name: "position", values: Array(values.indices))
        ])
        for (condition, expected) in [(FilterCondition.isNull, missing), (.isNotNull, present)] {
            if let selected = column.filteredIndices(matching: condition) {
                #expect(selected == expected)
            } else {
                #expect(!missing.isEmpty && !present.isEmpty)
            }
            for result in [
                try frame.filter(column: column.name, where: condition),
                try frame.filterFast(column: column.name, where: condition)
            ] {
                #expect(result.rowCount == expected.count)
                if !expected.isEmpty {
                    #expect(result[column: "position", as: Int.self]?.values == expected.map { Optional($0) })
                    let output = try #require(result[column: column.name, as: T.self])
                    #expect(output.nullCount == expected.filter { values[$0] == nil }.count)
                }
            }
        }
        #expect(column.nullCount == missing.count)
    }

    private func checkStates<T: SupportedType>(_ examples: [T]) throws {
        try check(TypedColumn<T>(name: "value", values: [T?]()))
        try check(TypedColumn<T>(name: "value", values: [T?](repeating: nil, count: 5)))
        try check(TypedColumn<T>(name: "value", values: examples))
        let mixed: [T?] = [nil, examples[0], examples[1], nil, examples[2]]
        let column = TypedColumn<T>(name: "value", values: mixed)
        try check(column)
        for indices in [[4, 0, 4, 3, 1], [1, 2, 4, 1], [3, 0, 3], []] {
            let gathered = try #require(column.gathered(at: indices) as? TypedColumn<T>)
            try check(gathered)
        }
        for mask in [
            [true, true, false, false, true],
            [false, true, true, false, true],
            [true, false, false, true, false],
            [false, false, false, false, false]
        ] {
            let filtered = try #require(column.filtered(by: mask) as? TypedColumn<T>)
            try check(filtered)
        }
    }

    @Test func nullPredicatesAcrossAllBuiltInTypesAndDerivedColumns() throws {
        try checkStates([Int32.min, 0, Int32.max])
        try checkStates([Int64.min, 9_007_199_254_740_993, Int64.max])
        try checkStates([Int.min, 0, Int.max])
        try checkStates([Float.nan, -.infinity, .infinity])
        try checkStates([Double.nan, -0.0, .infinity])
        try checkStates([false, true, false])
        try checkStates(["", "a retained long string value", "third"])
        try checkStates([Date(timeIntervalSince1970: -1), Date(timeIntervalSince1970: 0), Date(timeIntervalSince1970: 1)])
    }
}
