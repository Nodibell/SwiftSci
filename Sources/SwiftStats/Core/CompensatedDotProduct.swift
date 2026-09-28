import Accelerate

/// Retains product and addition residuals in three components.
/// Range exceptions restart from the original operands with exact integer products.
enum CompensatedDotProduct {
    private struct ScalarSum {
        var high = 0.0
        var middle = 0.0
        var low = 0.0

        mutating func add(_ value: Double) {
            let next = high + value
            let recovered = next - high
            let error = (high - (next - recovered)) + (value - recovered)
            high = next
            let correction = middle + error
            let recoveredCorrection = correction - middle
            low += (middle - (correction - recoveredCorrection)) + (error - recoveredCorrection)
            middle = correction
        }

        var isFinite: Bool { high.isFinite && middle.isFinite && low.isFinite }

        var value: Double {
            let next = high + middle
            let recovered = next - high
            let error = (high - (next - recovered)) + (middle - recovered)
            return next + (error + low)
        }
    }

    private struct VectorSum {
        var high = SIMD4<Double>(repeating: 0)
        var middle = SIMD4<Double>(repeating: 0)
        var low = SIMD4<Double>(repeating: 0)
        var largestProduct = 0.0

        mutating func add(_ value: SIMD4<Double>) {
            let next = high + value
            let recovered = next - high
            let error = (high - (next - recovered)) + (value - recovered)
            high = next
            let correction = middle + error
            let recoveredCorrection = correction - middle
            low += (middle - (correction - recoveredCorrection)) + (error - recoveredCorrection)
            middle = correction
        }

        mutating func addProduct(_ x: SIMD4<Double>, _ y: SIMD4<Double>) -> Bool {
            let product = x * y
            for lane in 0..<4 {
                if !safeProduct(product[lane], x: x[lane], y: y[lane]) { return false }
                largestProduct = max(largestProduct, abs(product[lane]))
            }
            add(product)
            add((-product).addingProduct(x, y))
            return true
        }

        func merge(into sum: inout ScalarSum) -> Bool {
            for lane in 0..<4 {
                guard high[lane].isFinite && middle[lane].isFinite && low[lane].isFinite else { return false }
                sum.add(high[lane])
                sum.add(middle[lane])
                sum.add(low[lane])
            }
            return true
        }
    }

    /// A product has at most 106 significant bits. Above 2^-969 its lowest
    /// possible exact bit is representable, so FMA can retain the product error.
    private static func safeProduct(_ product: Double, x: Double, y: Double) -> Bool {
        product.isFinite && (abs(product) > 0x1p-969 || x == 0 || y == 0)
    }

    private static func load(_ values: UnsafeBufferPointer<Double>, _ index: Int) -> SIMD4<Double> {
        SIMD4(values[index], values[index + 1], values[index + 2], values[index + 3])
    }

    static func evaluate(_ a: [Double], _ b: [Double]) -> Double {
        let result: Double? = a.withUnsafeBufferPointer { left in
            b.withUnsafeBufferPointer { right in
                var first = VectorSum(), second = VectorSum(), third = VectorSum(), fourth = VectorSum()
                var index = 0
                let vectorEnd = left.count - left.count % 16
                while index < vectorEnd {
                    guard first.addProduct(load(left, index), load(right, index)),
                          second.addProduct(load(left, index + 4), load(right, index + 4)),
                          third.addProduct(load(left, index + 8), load(right, index + 8)),
                          fourth.addProduct(load(left, index + 12), load(right, index + 12))
                    else { return nil }
                    index += 16
                }
                var largestProduct = max(first.largestProduct, second.largestProduct, third.largestProduct, fourth.largestProduct)
                var total = ScalarSum()
                guard first.merge(into: &total), second.merge(into: &total),
                      third.merge(into: &total), fourth.merge(into: &total)
                else { return nil }
                while index < left.count {
                    let x = left[index], y = right[index], product = x * y
                    guard safeProduct(product, x: x, y: y) else { return nil }
                    largestProduct = max(largestProduct, abs(product))
                    total.add(product)
                    total.add((-product).addingProduct(x, y))
                    index += 1
                }
                guard total.isFinite, total.value.isFinite else { return nil }
                // Corrections have finite depth. Resolve severely cancelled sums exactly.
                // This conservative trigger is not a public error-bound estimate.
                if largestProduct > 0 && abs(total.value) / largestProduct <= Double(left.count) * Double.ulpOfOne {
                    return nil
                }
                return total.value
            }
        }
        return result ?? ExactDotProduct.evaluate(a, b)
    }
}
