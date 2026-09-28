/// Exact finite products in units of 2^-2148, rounded once to binary64.
/// 67 limbs cover a 106-bit product at every Double exponent plus Int.max terms.
enum ExactDotProduct {
    private static let limbCount = 67

    static func evaluate(_ a: [Double], _ b: [Double]) -> Double {
        var positive = [UInt64](repeating: 0, count: limbCount)
        var negative = positive
        var positiveInfinity = false
        var negativeInfinity = false
        for index in a.indices {
            let x = a[index], y = b[index]
            if x.isNaN || y.isNaN { return .nan }
            let minus = x.sign != y.sign
            if x.isInfinite || y.isInfinite {
                if x == 0 || y == 0 { return .nan }
                if minus { negativeInfinity = true } else { positiveInfinity = true }
                continue
            }
            let left = parts(x), right = parts(y)
            if left.significand == 0 || right.significand == 0 { continue }
            let product = left.significand.multipliedFullWidth(by: right.significand)
            let shift = left.exponent + right.exponent + 2148
            if minus {
                add(product, shift: shift, to: &negative)
            } else {
                add(product, shift: shift, to: &positive)
            }
        }
        if positiveInfinity && negativeInfinity { return .nan }
        if positiveInfinity { return .infinity }
        if negativeInfinity { return -.infinity }
        for index in stride(from: limbCount - 1, through: 0, by: -1) {
            if positive[index] == negative[index] { continue }
            if positive[index] > negative[index] {
                subtract(negative, from: &positive)
                return rounded(positive, sign: .plus)
            }
            subtract(positive, from: &negative)
            return rounded(negative, sign: .minus)
        }
        return 0
    }

    private static func parts(_ value: Double) -> (significand: UInt64, exponent: Int) {
        let bits = value.bitPattern
        let exponent = Int((bits >> 52) & 0x7ff)
        let fraction = bits & 0x000f_ffff_ffff_ffff
        return exponent == 0 ? (fraction, -1074) : (fraction | (1 << 52), exponent - 1075)
    }

    private static func add(
        _ product: (high: UInt64, low: UInt64), shift: Int, to limbs: inout [UInt64]
    ) {
        let index = shift / 64, offset = shift % 64
        if offset == 0 {
            addWord(product.low, at: index, to: &limbs)
            addWord(product.high, at: index + 1, to: &limbs)
        } else {
            addWord(product.low << offset, at: index, to: &limbs)
            addWord((product.high << offset) | (product.low >> (64 - offset)), at: index + 1, to: &limbs)
            addWord(product.high >> (64 - offset), at: index + 2, to: &limbs)
        }
    }

    private static func addWord(_ word: UInt64, at index: Int, to limbs: inout [UInt64]) {
        var index = index, carry = word
        while carry != 0 {
            let addition = limbs[index].addingReportingOverflow(carry)
            limbs[index] = addition.partialValue
            carry = addition.overflow ? 1 : 0
            index += 1
        }
    }

    private static func subtract(_ smaller: [UInt64], from larger: inout [UInt64]) {
        var borrow: UInt64 = 0
        for index in larger.indices {
            let first = larger[index].subtractingReportingOverflow(smaller[index])
            let second = first.partialValue.subtractingReportingOverflow(borrow)
            larger[index] = second.partialValue
            borrow = first.overflow || second.overflow ? 1 : 0
        }
    }

    private static func rounded(_ magnitude: [UInt64], sign: FloatingPointSign) -> Double {
        let top = magnitude.lastIndex(where: { $0 != 0 })!
        let highestBit = top * 64 + 63 - magnitude[top].leadingZeroBitCount
        let discard = max(1074, highestBit - 52)
        let index = discard / 64, offset = discard % 64
        var retained = magnitude[index] >> offset
        if offset != 0 && index + 1 < magnitude.count {
            retained |= magnitude[index + 1] << (64 - offset)
        }
        let guardPosition = discard - 1
        let guardIndex = guardPosition / 64, guardOffset = guardPosition % 64
        let guardBit = (magnitude[guardIndex] >> guardOffset) & 1
        let lowerMask = (UInt64(1) << guardOffset) - 1
        let sticky = magnitude[..<guardIndex].contains(where: { $0 != 0 })
            || magnitude[guardIndex] & lowerMask != 0
        if guardBit != 0 && (sticky || retained & 1 != 0) { retained += 1 }
        return Double(sign: sign, exponent: discard - 2148, significand: Double(retained))
    }
}
