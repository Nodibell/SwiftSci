import Testing
@testable import SwiftDataFrame

@Suite("Radix final buffer ownership")
struct RadixFinalBufferTests {
    @Test("All-present columns retain stable rows after an odd number of radix passes",
          arguments: [false, true])
    func oddPasses(compact: Bool) {
        for bits in [8, 24, 48] {
            var state: UInt64 = 0x91234abcde
            let mask: UInt64 = (UInt64(1) << bits) - 1
            var values: [Double] = []
            for _ in 0..<4097 {
                state = state &* 6364136223846793005 &+ 1
                values.append(Double(bitPattern: 0x3ff0000000000000 | (state & mask)))
            }
            let column: TypedColumn<Double>
            if compact {
                var builder = CSVCompactDoubleBuilder(capacity: values.count)
                for value in values { builder.append(value) }
                column = builder.column(named: "x")
            } else {
                column = TypedColumn<Double>(name: "x", values: values)
            }
            for ascending in [false, true] {
                let expected: [Int] = values.indices.sorted { (left: Int, right: Int) -> Bool in
                    if values[left] == values[right] { return left < right }
                    return ascending ? values[left] < values[right] : values[left] > values[right]
                }
                #expect(column.sortedIndices(ascending: ascending) == expected)
            }
            #expect(column.nullCount == 0)
            let actualBits: [UInt64?] = column.values.map { $0?.bitPattern }
            let expectedBits: [UInt64?] = values.map { Optional($0.bitPattern) }
            #expect(actualBits == expectedBits)
        }
    }
}
