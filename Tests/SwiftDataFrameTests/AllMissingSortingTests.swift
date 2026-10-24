import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("All-missing stable sorting")
struct AllMissingSortingTests {
    @Test(arguments: [0, 1, 63, 64, 65, 1024, 4097])
    func missingKeysPreserveNativeRowIdentities(count: Int) throws {
        let columns: [any AnyColumn] = [
            TypedColumn<Double>(name: "x", values: Array(repeating: nil, count: count)),
            TypedColumn<Float>(name: "x", values: Array(repeating: nil, count: count)),
            TypedColumn<Int64>(name: "x", values: Array(repeating: nil, count: count)),
            TypedColumn<Int32>(name: "x", values: Array(repeating: nil, count: count)),
            TypedColumn<Int>(name: "x", values: Array(repeating: nil, count: count)),
            TypedColumn<String>(name: "x", values: Array(repeating: nil, count: count)),
            TypedColumn<Date>(name: "x", values: Array(repeating: nil, count: count)),
            TypedColumn<Bool>(name: "x", values: Array(repeating: nil, count: count))
        ]
        let ids = TypedColumn<Int64>(name: "id",
            values: (0..<count).map { Int64($0) + 9_007_199_254_740_993 })
        for column in columns {
            let frame = try DataFrame(columns: [column, ids])
            for ascending in [false, true] {
                #expect(column.sortedIndices(ascending: ascending) == Array(0..<count))
                let sorted = try frame.sortBy("x", ascending: ascending)
                #expect(sorted[column: "id", as: Int64.self]?.values == ids.values)
                #expect(sorted.columnNames == frame.columnNames)
                #expect(sorted.dtypes == frame.dtypes)
                #expect(sorted[column: "x"]?.nullCount == count)
            }
        }
    }

    @Test(arguments: [0, 1, 63, 64, 65, 1024, 4097])
    func compactMissingKeysKeepValidNaNsDistinct(count: Int) {
        var builder = CSVCompactDoubleBuilder(capacity: count)
        for _ in 0..<count { builder.append(nil) }
        let missing = builder.column(named: "x")
        for ascending in [false, true] {
            #expect(missing.sortedIndices(ascending: ascending) == Array(0..<count))
        }
        guard count > 0 else { return }
        for position in [0, count / 2, count - 1] {
            let nan = Double(bitPattern: 0x7ff8000000000042)
            var values = [Double?](repeating: nil, count: count)
            values[position] = nan
            var mixed = CSVCompactDoubleBuilder(capacity: count)
            for value in values { mixed.append(value) }
            let compact = mixed.column(named: "x")
            let optional = TypedColumn(name: "x", values: values)
            let expected = [position] + values.indices.filter { $0 != position }
            for ascending in [false, true] {
                #expect(compact.sortedIndices(ascending: ascending) == expected)
                #expect(optional.sortedIndices(ascending: ascending) == expected)
            }
            #expect(compact.nullCount == count - 1)
            #expect(compact[position]?.bitPattern == nan.bitPattern)
        }
    }
}
