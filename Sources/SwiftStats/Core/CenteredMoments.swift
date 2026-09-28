/// Two-pass centered moments without an input-sized temporary buffer.
enum CenteredMoments {
    private struct Sum {
        var high = 0.0
        var low = 0.0

        mutating func add(_ value: Double) {
            let next = high + value
            low += abs(high) >= abs(value) ? (high - next) + value : (value - next) + high
            high = next
        }

        var value: Double { high + low }
    }

    private struct VectorSum {
        var high = SIMD4<Double>(repeating: 0)
        var low = SIMD4<Double>(repeating: 0)

        mutating func add(_ value: SIMD4<Double>) {
            let next = high + value
            let recovered = next - high
            low += (high - (next - recovered)) + (value - recovered)
            high = next
        }

        func merge(into sum: inout Sum) {
            for lane in 0..<4 {
                sum.add(high[lane])
                sum.add(low[lane])
            }
        }
    }

    private static func load(_ values: UnsafeBufferPointer<Double>, _ index: Int) -> SIMD4<Double> {
        SIMD4(values[index], values[index + 1], values[index + 2], values[index + 3])
    }

    private static func relativeMean(_ values: UnsafeBufferPointer<Double>, origin: Double) -> Double {
        let shift = SIMD4<Double>(repeating: origin)
        var first = VectorSum(), second = VectorSum(), third = VectorSum(), fourth = VectorSum()
        var index = 0
        while index + 16 <= values.count {
            first.add(load(values, index) - shift)
            second.add(load(values, index + 4) - shift)
            third.add(load(values, index + 8) - shift)
            fourth.add(load(values, index + 12) - shift)
            index += 16
        }
        var sum = Sum()
        first.merge(into: &sum)
        second.merge(into: &sum)
        third.merge(into: &sum)
        fourth.merge(into: &sum)
        while index < values.count {
            sum.add(values[index] - origin)
            index += 1
        }
        return sum.value / Double(values.count)
    }

    /// Returns nil when the centered representation overflows or encounters nonfinite values.
    /// The caller retains its established handling for those inputs.
    static func sumOfSquares(_ values: [Double]) -> Double? {
        values.withUnsafeBufferPointer { buffer in
            guard let origin = buffer.first, origin.isFinite else { return nil }
            let mean = relativeMean(buffer, origin: origin)
            guard mean.isFinite else { return nil }
            let shift = SIMD4<Double>(repeating: origin)
            let center = SIMD4<Double>(repeating: mean)
            var squares = Sum(), deviations = Sum()
            var index = 0
            while index + 16 <= buffer.count {
                // Short blocks bound each lane's accumulation depth before compensation.
                let end = min(index + 256, buffer.count - buffer.count % 16)
                var s0 = SIMD4<Double>(repeating: 0), s1 = s0, s2 = s0, s3 = s0
                var d0 = s0, d1 = s0, d2 = s0, d3 = s0
                while index < end {
                    let x0 = (load(buffer, index) - shift) - center
                    let x1 = (load(buffer, index + 4) - shift) - center
                    let x2 = (load(buffer, index + 8) - shift) - center
                    let x3 = (load(buffer, index + 12) - shift) - center
                    s0 = s0.addingProduct(x0, x0)
                    s1 = s1.addingProduct(x1, x1)
                    s2 = s2.addingProduct(x2, x2)
                    s3 = s3.addingProduct(x3, x3)
                    d0 += x0; d1 += x1; d2 += x2; d3 += x3
                    index += 16
                }
                for lane in 0..<4 {
                    squares.add(s0[lane]); squares.add(s1[lane])
                    squares.add(s2[lane]); squares.add(s3[lane])
                    deviations.add(d0[lane]); deviations.add(d1[lane])
                    deviations.add(d2[lane]); deviations.add(d3[lane])
                }
            }
            while index < buffer.count {
                let difference = (buffer[index] - origin) - mean
                squares.add(difference * difference)
                deviations.add(difference)
                index += 1
            }
            // Correct for rounding of the relative mean without reconstructing an absolute mean.
            let result = squares.value.addingProduct(-deviations.value / Double(buffer.count), deviations.value)
            guard result.isFinite, result >= 0 else { return nil }
            return result
        }
    }
}
