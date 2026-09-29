import SwiftDataFrame

extension Stats {
    /// One-way ANOVA that preserves decimal differences before binary64 conversion.
    ///
    /// Pass the original decimal text, not strings reconstructed from `Double`.
    /// The first observation is subtracted from every observation in decimal
    /// arithmetic. The existing binary64 ANOVA then operates on those residuals.
    /// See <doc:DecimalANOVA> for input limits and numerical guarantees.
    public static func oneWayANOVA(decimalGroups: [[String]]) throws -> ANOVAResult {
        guard decimalGroups.count >= 2 else {
            throw StatsError.invalidGroupCount(minimum: 2, got: decimalGroups.count)
        }
        for group in decimalGroups {
            guard !group.isEmpty else { throw StatsError.emptyInput }
        }
        let origin = try ANOVADecimal.parse(decimalGroups[0][0])
        let residuals = try decimalGroups.map { group in
            try group.map { text in
                let value = try ANOVADecimal.parse(text)
                let residual = try value.subtracting(origin)
                guard let converted = Double(residual.text), converted.isFinite,
                      converted != 0 || residual.digits == "0" else {
                    throw StatsError.invalidInput("oneWayANOVA: decimal residual exceeds binary64 range")
                }
                return converted
            }
        }
        return try oneWayANOVA(groups: residuals)
    }
}

/// Bounded decimal coefficients keep parsing and subtraction exact before conversion.
private struct ANOVADecimal: Equatable {
    let negative: Bool
    let digits: String
    let exponent: Int

    var text: String { (negative ? "-" : "") + digits + "e" + String(exponent) }

    static func parse(_ text: String) throws -> ANOVADecimal {
        let value = try normalize(text)
        guard value.digits.count <= 38, (-128...127).contains(value.exponent) else {
            throw StatsError.invalidInput("oneWayANOVA: decimal input is outside the supported exact range")
        }
        return value
    }

    func subtracting(_ other: ANOVADecimal) throws -> ANOVADecimal {
        let commonExponent = min(exponent, other.exponent)
        // Input bounds limit the aligned coefficients to at most 293 digits.
        func aligned(_ value: ANOVADecimal) -> [UInt8] {
            if value.digits == "0" { return [0] }
            return Array(repeating: 0, count: value.exponent - commonExponent)
                + value.digits.utf8.reversed().map { $0 - 48 }
        }
        let left = aligned(self), right = aligned(other)
        var result: [UInt8] = []
        let sign: Bool
        if negative != other.negative {
            sign = negative
            var carry: UInt8 = 0
            for i in 0..<max(left.count, right.count) {
                let sum = (i < left.count ? left[i] : 0) + (i < right.count ? right[i] : 0) + carry
                result.append(sum % 10)
                carry = sum / 10
            }
            if carry != 0 { result.append(carry) }
        } else {
            let leftIsSmaller = left.count != right.count
                ? left.count < right.count
                : left.reversed().lexicographicallyPrecedes(right.reversed())
            let larger = leftIsSmaller ? right : left
            let smaller = leftIsSmaller ? left : right
            sign = leftIsSmaller ? !negative : negative
            var borrow = 0
            for i in larger.indices {
                var difference = Int(larger[i]) - (i < smaller.count ? Int(smaller[i]) : 0) - borrow
                borrow = difference < 0 ? 1 : 0
                if difference < 0 { difference += 10 }
                result.append(UInt8(difference))
            }
        }
        guard let first = result.firstIndex(where: { $0 != 0 }),
              let last = result.lastIndex(where: { $0 != 0 }) else {
            return ANOVADecimal(negative: false, digits: "0", exponent: 0)
        }
        guard last - first + 1 <= 38 else {
            throw StatsError.invalidInput("oneWayANOVA: decimal subtraction exceeds 38 significant digits")
        }
        return ANOVADecimal(negative: sign,
                           digits: String(decoding: result[first...last].reversed().map { $0 + 48 }, as: UTF8.self),
                           exponent: commonExponent + first)
    }

    private static func normalize(_ text: String) throws -> ANOVADecimal {
        let bytes = Array(text.utf8)
        var i = 0
        func digit(_ b: UInt8) -> Bool { b >= 48 && b <= 57 }
        let negative = bytes.first == 45
        if bytes.first == 45 || bytes.first == 43 { i += 1 }
        var digits: [UInt8] = []
        while i < bytes.count && digit(bytes[i]) { digits.append(bytes[i]); i += 1 }
        var fractionCount = 0
        if i < bytes.count && bytes[i] == 46 {
            i += 1
            while i < bytes.count && digit(bytes[i]) {
                digits.append(bytes[i]); fractionCount += 1; i += 1
            }
        }
        guard !digits.isEmpty else {
            throw StatsError.invalidInput("oneWayANOVA: expected an ASCII decimal number")
        }
        var exponent = 0
        if i < bytes.count && (bytes[i] == 69 || bytes[i] == 101) {
            i += 1
            let start = i
            if i < bytes.count && (bytes[i] == 43 || bytes[i] == 45) { i += 1 }
            let firstDigit = i
            while i < bytes.count && digit(bytes[i]) { i += 1 }
            guard i > firstDigit,
                  let parsed = Int(String(decoding: bytes[start..<i], as: UTF8.self)) else {
                throw StatsError.invalidInput("oneWayANOVA: invalid decimal exponent")
            }
            exponent = parsed
        }
        guard i == bytes.count else {
            throw StatsError.invalidInput("oneWayANOVA: expected an ASCII decimal number")
        }
        guard let first = digits.firstIndex(where: { $0 != 48 }) else {
            return ANOVADecimal(negative: false, digits: "0", exponent: 0)
        }
        let last = digits.lastIndex(where: { $0 != 48 })!
        let (fractionAdjusted, overflow) = exponent.subtractingReportingOverflow(fractionCount)
        let (normalizedExponent, trailingOverflow) = fractionAdjusted.addingReportingOverflow(digits.count - last - 1)
        guard !overflow && !trailingOverflow else {
            throw StatsError.invalidInput("oneWayANOVA: decimal exponent exceeds supported range")
        }
        return ANOVADecimal(negative: negative,
                         digits: String(decoding: digits[first...last], as: UTF8.self),
                         exponent: normalizedExponent)
    }
}
