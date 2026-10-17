import Testing
import SwiftDataFrame
@testable import SwiftPreprocessing

@Suite("Typed preprocessing row writer", .serialized)
struct StandardPreprocessingTypedRowWriterTests {
    @Test(arguments: [7, 31, 127, 128, 129, 512, 513])
    func conversionBitsAndPadding(width: Int) throws {
        try checkRows(Double.self, width: width) { $0.bitPattern }
        try checkRows(Float.self, width: width) { UInt64($0.bitPattern) }
        try checkRows(Float16.self, width: width) { UInt64($0.bitPattern) }
    }

    private func checkRows<T: BinaryFloatingPoint>(_ type: T.Type, width: Int,
                                                   bits: (T) -> UInt64) throws {
        let names: [String] = (0..<width).map { (c: Int) -> String in "x\(c)" }
        let columns: [[Double]] = (0..<width).map { (c: Int) -> [Double] in
            (0..<65).map { (r: Int) -> Double in
                let missing: Bool = (r + c) % 13 == 0
                let difference: Double = Double(r - c)
                let value: Double = difference / 19
                return missing ? .nan : value
            }
        }
        let input = try PreparedNumericBatch(columnNames: names, columns: columns)
        let plan = try StandardPreprocessingPlan(training: input)
        let expected: [Double] = try plan.scaler.transform(plan.imputer.transform(input)).matrix().values
        var writer = try plan.makeRowWriter(input, maximumRows: 33)
        let stride = width + 5
        for range in [0..<33, 17..<50, 64..<65, 32..<32] {
            var buffer = [T](repeating: T(-999), count: 35 * stride)
            try buffer.withUnsafeMutableBufferPointer {
                try writer.fill(rows: range, into: $0, stride: stride)
            }
            for r in 0..<35 {
                for c in 0..<stride {
                    let want: T = r < range.count && c < width
                        ? T(expected[(range.lowerBound + r) * width + c]) : T(-999)
                    #expect(bits(buffer[r * stride + c]) == bits(want))
                }
            }
        }
    }
}
