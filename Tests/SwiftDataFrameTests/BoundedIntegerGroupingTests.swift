import Testing
import SwiftDataFrame

@Suite("Bounded integer group lookup")
struct BoundedIntegerGroupingTests {
    private func check<T: SupportedType>(_ keys: [T?], order: [T?]) throws {
        let expected = order.map { key in Int64(keys.filter { $0 == key }.count) }
        let frame = try DataFrame(columns: [
            TypedColumn<T>(name: "key", values: keys),
            TypedColumn<Double>(name: "value", values: Array(repeating: 1, count: keys.count))])
        let result = frame.groupBy("key").count()
        #expect(result[column: "key", as: String.self]?.values == order.map { $0.map { String(describing: $0) } })
        #expect(result[column: "value", as: Int64.self]?.values == expected.map { Optional($0) })
        #expect(frame.groupBy("key").sum()[column: "value", as: Double.self]?.values == expected.map { Double($0) })
    }

    @Test(arguments: [Int64.min, -3, Int64.max - 2])
    func narrowRangesAtIntegerLimits(base: Int64) throws {
        try check([nil, base + 2, base, base + 1, base + 2], order: [nil, base + 2, base, base + 1])
    }

    @Test(arguments: [65_535, 65_536])
    func lookupBudgetBoundary(width: Int) throws {
        var keys = [Int64?](repeating: 0, count: 65_538)
        keys[1] = Int64(width)
        keys[2] = nil
        try check(keys, order: [0, Int64(width), nil])
    }

    @Test func sparseOverflowAndNullOnlyInputs() throws {
        try check([Int64.min, .max, nil, 1, -1, .max], order: [.min, .max, nil, 1, -1])
        try check([Int64(0), 1_000_003, nil, 0], order: [0, 1_000_003, nil])
        try check([Int64?](repeating: nil, count: 17), order: [nil])
    }

    @Test func smallerAndNativeIntegerWidths() throws {
        try check([Int32.max, .max - 2, nil, .max - 1, .max], order: [.max, .max - 2, nil, .max - 1])
        try check([Int.min, .min + 2, nil, .min + 1, .min], order: [.min, .min + 2, nil, .min + 1])
    }
}
