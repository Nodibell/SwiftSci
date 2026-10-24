import Testing
import SwiftDataFrame

@Suite("Radix buffer permutations")
struct RadixBufferSortingTests {
    @Test(arguments: 1...6, [false, true])
    func varyingDigitCounts(bytes: Int, descending: Bool) {
        var state: UInt64 = 0x91234abcde
        let mask = (UInt64(1) << (bytes * 8)) - 1
        let values: [Double?] = (0..<4_097).map { row in
            state = state &* 6364136223846793005 &+ 1
            return row % 19 == 0 ? nil : Double(bitPattern: 0x3ff0000000000000 | (state & mask))
        }
        let expected = values.indices.sorted { left, right in
            switch (values[left], values[right]) {
            case (nil, nil): return left < right
            case (nil, _): return false
            case (_, nil): return true
            case let (a?, b?): return a == b ? left < right : descending ? a > b : a < b
            }
        }
        let column = TypedColumn<Double>(name: "key", values: values)
        #expect(column.sortedIndices(ascending: !descending) == expected)
        #expect(column.values.map { $0?.bitPattern } == values.map { $0?.bitPattern })
    }
}
