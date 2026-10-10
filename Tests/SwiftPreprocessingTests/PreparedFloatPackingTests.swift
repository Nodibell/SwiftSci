import Testing
import SwiftDataFrame
import SwiftPreprocessing

@Suite("Prepared Float32 packing")
struct PreparedFloatPackingTests {
    @Test func packingMatchesScalarConversionAcrossTileAndPageEdges() throws {
        let shapes = [(0, 3), (1, 1), (31, 33), (32, 32), (33, 31), (63, 64),
                      (64, 64), (65, 64), (127, 129), (1025, 65), (65, 17), (65, 63), (65, 127)]
        let extremes: [Double] = [0, -0.0, .leastNonzeroMagnitude,
            Double(Float.greatestFiniteMagnitude), .greatestFiniteMagnitude]
        for (rows, columns) in shapes {
            let source: [[Double]] = (0..<columns).map { column -> [Double] in
                (0..<rows).map { row -> Double in
                    let index: Int = row * columns + column
                    if index < extremes.count { return extremes[index] }
                    let numerator: Int = index % 997 - 498
                    return Double(numerator) / 127.0
                }
            }
            let batch = try PreparedNumericBatch(columnNames: (0..<columns).map { "x\($0)" }, columns: source)
            let count = rows * columns
            do {
                let allocation = UnsafeMutablePointer<Float>.allocate(capacity: count + 2)
                allocation.initialize(repeating: -99, count: count + 2)
                defer { allocation.deinitialize(count: count + 2); allocation.deallocate() }
                let destination = UnsafeMutableBufferPointer(start: allocation + 1, count: count)
                PreparedFloatPacking.initialize(batch, into: destination)
                #expect(allocation[0] == -99 && allocation[count + 1] == -99)
                for row in 0..<rows {
                    for column in 0..<columns {
                        #expect(destination[row * columns + column].bitPattern == Float(source[column][row]).bitPattern)
                    }
                }
            }
        }
    }
}
