import Testing
import SwiftStats
import SwiftDataFrame

@Suite("Dot-product accuracy policy")
struct DotProductAccuracyTests {
    @Test("Existing calls and function references retain the performance default",
          arguments: [1, 31, 384, 1536, 4095, 4096, 4097, 65536])
    func performanceDefault(count: Int) throws {
        let a = (0..<count).map { Double($0 % 17 - 8) / 16 }
        let b = Array(repeating: 2.0, count: count)
        let original: ([Double], [Double]) throws -> Double = Stats.dotProduct
        let expected = a.reduce(0, +) * 2
        #expect(try original(a, b) == expected)
        #expect(try Stats.dotProduct(a, b, accuracy: .performance) == expected)
        #expect(try Stats.dotProduct(a, b, accuracy: .compensated) == expected)
    }

    @Test("Retains product rounding residuals across vector and tail boundaries",
          arguments: [1, 2, 7, 8, 9, 17, 257, 4097])
    func productResiduals(repetitions: Int) throws {
        let delta = 0x1p-27
        let a = Array(repeating: [1 + delta, -1.0], count: repetitions).flatMap { $0 }
        let b = Array(repeating: [1 - delta, 1.0], count: repetitions).flatMap { $0 }
        #expect(try Stats.dotProduct(a, b, accuracy: .compensated) == -Double(repetitions) * 0x1p-54)
    }

    @Test("Preserves contributions through severe cancellation",
          arguments: [1, 4, 16, 65])
    func cancellation(repetitions: Int) throws {
        let pattern: [Double] = [0x1p100, 1, 0x1p-100, -0x1p100, -1]
        let values = Array(repeating: pattern, count: repetitions).flatMap { $0 }
        let ones = Array(repeating: 1.0, count: values.count)
        let expected = Double(repetitions) * 0x1p-100
        #expect(try Stats.dotProduct(values, ones, accuracy: .compensated) == expected)
        #expect(try Stats.dotProduct(Array(values.reversed()), ones, accuracy: .compensated) == expected)
        #expect(try Stats.dotProduct([1e16, 1, -1e16], [1, 1, 1], accuracy: .compensated) == 1)
    }

    @Test("Recovers a representable sum of individually underflowing products",
          arguments: [2, 3, 16, 257, 4096])
    func underflow(count: Int) throws {
        let a = Array(repeating: 0x1p-537, count: count)
        let b = Array(repeating: 0x1p-538, count: count)
        let units = (Double(count) / 2).rounded(.toNearestOrEven)
        let expected = units * Double.leastNonzeroMagnitude
        #expect(try Stats.dotProduct(a, b, accuracy: .compensated) == expected)
        #expect(try Stats.dotProduct(a.map { -$0 }, b, accuracy: .compensated) == -expected)
    }

    @Test("Finite cancellation survives overflowing products and partial sums")
    func overflow() throws {
        let largest = Double.greatestFiniteMagnitude
        #expect(try Stats.dotProduct([0x1p1023, -0x1p1023, 1], [2, 2, 1], accuracy: .compensated) == 1)
        #expect(try Stats.dotProduct([largest, largest, -largest], [1, 1, 1], accuracy: .compensated) == largest)
        #expect(try Stats.dotProduct([largest, largest], [2, 2], accuracy: .compensated) == .infinity)
    }

    @Test("Subnormal rounding uses ties to even and preserves negative underflow sign")
    func rounding() throws {
        let tiny = Double.leastNonzeroMagnitude
        #expect(try Stats.dotProduct([tiny], [1], accuracy: .compensated) == tiny)
        #expect(try Stats.dotProduct([tiny], [0.5], accuracy: .compensated) == 0)
        #expect(try Stats.dotProduct([3 * tiny], [0.5], accuracy: .compensated) == 2 * tiny)
        let negativeZero = try Stats.dotProduct([-tiny], [0.5], accuracy: .compensated)
        #expect(negativeZero == 0 && negativeZero.sign == .minus)
        #expect(try Stats.dotProduct([tiny, -tiny], [1, 1], accuracy: .compensated).bitPattern == 0)
    }

    @Test("Cancellation beyond the correction components uses range-safe accumulation")
    func separatedMagnitudes() throws {
        let a: [Double] = [0x1p500, 0x1p250, 1, 0x1p-250, -0x1p500, -0x1p250, -1]
        #expect(try Stats.dotProduct(a, Array(repeating: 1, count: a.count), accuracy: .compensated) == 0x1p-250)
    }

    @Test("Exact fallback rounds ties, carry into the normal range, and overflow midpoint")
    func roundingBoundaries() throws {
        let tiny = Double.leastNonzeroMagnitude
        let normal = Double.leastNormalMagnitude
        let largest = Double.greatestFiniteMagnitude
        #expect(try Stats.dotProduct([1, 0x1p-53, 0x1p-537], [1, 1, 0x1p-538], accuracy: .compensated) == 1.0.nextUp)
        #expect(try Stats.dotProduct([normal, -tiny, tiny], [1, 1, 0.5], accuracy: .compensated) == normal)
        #expect(try Stats.dotProduct([normal, -2 * tiny, tiny], [1, 1, 0.5], accuracy: .compensated) == normal - 2 * tiny)
        #expect(try Stats.dotProduct([largest, 0x1p970, -tiny], [1, 1, 1], accuracy: .compensated) == largest)
        #expect(try Stats.dotProduct([largest, 0x1p970], [1, 1], accuracy: .compensated) == .infinity)
    }

    @Test("Range qualification includes the product residual", arguments: [-968, -969, -970])
    func productRange(exponent: Int) throws {
        let delta = 0x1p-27
        let scale = Double(sign: .plus, exponent: exponent, significand: 1)
        #expect(try Stats.dotProduct([(1 + delta) * scale, -scale], [1 - delta, 1], accuracy: .compensated)
            == -Double(sign: .plus, exponent: exponent - 54, significand: 1))
    }

    @Test("SIMD compensation retains product residues without severe cancellation",
          arguments: [3, 16, 17, 32, 33])
    func ordinaryProductResiduals(count: Int) throws {
        let delta = 0x1p-27
        var a = Array(repeating: 0.0, count: count)
        var b = Array(repeating: 1.0, count: count)
        a[0] = 1 + delta; a[1] = 1 + delta; a[count - 1] = -1
        b[0] = 1 - delta; b[1] = 1 - delta
        #expect(try Stats.dotProduct(a, b, accuracy: .compensated) == 1.0.nextDown)
    }

    @Test("Exact carry and borrow chains preserve the smallest contributions")
    func carryAndStickyBits() throws {
        let largest = Double.greatestFiniteMagnitude
        let tiny = Double.leastNonzeroMagnitude
        let a = Array(repeating: largest, count: 64) + Array(repeating: -largest, count: 64) + [tiny]
        #expect(try Stats.dotProduct(a, Array(repeating: 1, count: a.count), accuracy: .compensated) == tiny)
        #expect(try Stats.dotProduct([1, 0x1p-53, tiny], [1, 1, tiny], accuracy: .compensated) == 1.0.nextUp)
        #expect(try Stats.dotProduct([largest, 0x1p970, -tiny], [1, 1, tiny], accuracy: .compensated) == largest)
    }

    @Test("Performance dispatch preserves exceptional propagation on large vectors",
          arguments: [4096, 4097])
    func largeExceptionalValues(count: Int) throws {
        var a = Array(repeating: 0.0, count: count)
        let b = Array(repeating: 1.0, count: count)
        a[0] = .infinity
        #expect(try Stats.dotProduct(a, b) == .infinity)
        a[count - 1] = -.infinity
        #expect(try Stats.dotProduct(a, b).isNaN)
        a[0] = .nan
        #expect(try Stats.dotProduct(a, b).isNaN)
    }

    @Test("Validation and nonfinite propagation are defined for both policies",
          arguments: [DotProductAccuracy.performance, .compensated])
    func exceptionalValues(accuracy: DotProductAccuracy) throws {
        #expect(throws: StatsError.emptyInput) { try Stats.dotProduct([], [], accuracy: accuracy) }
        #expect(throws: StatsError.self) { try Stats.dotProduct([1], [1, 2], accuracy: accuracy) }
        #expect(try Stats.dotProduct([.nan, 1], [1, 2], accuracy: accuracy).isNaN)
        #expect(try Stats.dotProduct([.infinity, 1], [1, 2], accuracy: accuracy) == .infinity)
        #expect(try Stats.dotProduct([.infinity, -.infinity], [1, 1], accuracy: accuracy).isNaN)
        #expect(try Stats.dotProduct([0], [.infinity], accuracy: accuracy).isNaN)
    }
}
