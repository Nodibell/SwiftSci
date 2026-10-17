import Testing
import SwiftDataFrame
import SwiftPreprocessing

private func conversionBoundaryValues() -> [Double] {
    let half: Double = 0x1.002p0
    let halfTiny: Double = 0x1p-25
    let halfOverflow: Double = 65520
    let single: Double = 0x1.000001p0
    let singleTiny: Double = 0x1p-150
    let singleMaximum: Double = Double(Float.greatestFiniteMagnitude)
    let singleHalfULP: Double = Double(Float.greatestFiniteMagnitude.ulp) / 2
    let singleOverflow: Double = singleMaximum + singleHalfULP
    let centers: [Double] = [half, halfTiny, halfOverflow, single, singleTiny, singleOverflow]
    let edges: [Double] = centers.flatMap { (value: Double) -> [Double] in
        [value.nextDown, value, value.nextUp]
    }
    let negatives: [Double] = edges.map { (value: Double) -> Double in -value }
    let exceptional: [Double] = [-0.0, 0, .infinity, -.infinity, .nan]
    return edges + negatives + exceptional
}

@Suite("Fitted preprocessing conversion boundaries", .serialized)
struct PreprocessingConversionBoundaryTests {
    @Test(arguments: [1, 7, 31, 127, 128, 129, 512, 513])
    func preservesSingleRounding(width: Int) throws {
        let names: [String] = (0..<width).map { (c: Int) -> String in "x\(c)" }
        let trainingColumns: [[Double]] = Array(repeating: [-1, 1], count: width)
        let training = try PreparedNumericBatch(columnNames: names, columns: trainingColumns)
        let plan = try StandardPreprocessingPlan(training: training)
        let edges: [Double] = conversionBoundaryValues()
        let columns: [[Double]] = (0..<width).map { (c: Int) -> [Double] in
            (0..<65).map { (r: Int) -> Double in
                let position: Int = (r + c) % edges.count
                return edges[position]
            }
        }
        let input = try PreparedNumericBatch(columnNames: names, columns: columns)
        let expected: [Double] = try plan.scaler.transform(plan.imputer.transform(input)).matrix().values
        let expectedSingle: [UInt32] = expected.map { (value: Double) -> UInt32 in Float(value).bitPattern }
        let expectedHalf: [UInt16] = expected.map { (value: Double) -> UInt16 in Float16(value).bitPattern }
        var single = [Float](repeating: 0, count: expected.count)
        var half = [Float16](repeating: 0, count: expected.count)
        try single.withUnsafeMutableBufferPointer { try plan.fillFloat32(input, into: $0) }
        try half.withUnsafeMutableBufferPointer { try plan.fillFloat16(input, into: $0) }
        #expect(single.map(\.bitPattern) == expectedSingle)
        #expect(half.map(\.bitPattern) == expectedHalf)
        #expect(half[0].bitPattern == 0x3c00)
        #expect(half[width].bitPattern == 0x3c00)
        #expect(half[2 * width].bitPattern == 0x3c01)
        #expect(single[9 * width].bitPattern == 0x3f800000)
        #expect(single[10 * width].bitPattern == 0x3f800000)
        #expect(single[11 * width].bitPattern == 0x3f800001)
    }
}
