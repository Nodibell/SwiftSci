import Foundation

/// Internal numerical safety and polynomial validation helpers for ARIMA/SARIMA models.
internal enum ForecastValidation {
    /// Verifies that every element in `values` is finite.
    ///
    /// Checks for `NaN` precedence matching existing `SwiftForecast` guards:
    /// if any element is `NaN`, throws `ForecastError.containsNaN`; otherwise
    /// if any element is infinite, throws `ForecastError.containsInfinity`.
    static func ensureFinite(_ values: [Double]) throws {
        var hasInfinity = false
        for value in values {
            if value.isNaN {
                throw ForecastError.containsNaN
            }
            if value.isInfinite {
                hasInfinity = true
            }
        }
        if hasInfinity {
            throw ForecastError.containsInfinity
        }
    }

    /// Verifies that a scalar floating-point value is finite.
    static func ensureFinite(_ value: Double) throws {
        if value.isNaN {
            throw ForecastError.containsNaN
        }
        if value.isInfinite {
            throw ForecastError.containsInfinity
        }
    }

    /// Verifies that moving-average polynomials are invertible under the `1 + sum(theta_k B^k)` sign convention.
    static func ensureMAInvertible(ma: [Double], seasonalMA: [Double] = []) throws {
        guard AutoARIMA.isMAPolynomialInvertible(coefficients: ma) else {
            throw ForecastError.trainingFailed(
                "Estimated moving-average polynomial is non-invertible (roots on or inside the unit circle)."
            )
        }
        guard AutoARIMA.isMAPolynomialInvertible(coefficients: seasonalMA) else {
            throw ForecastError.trainingFailed(
                "Estimated seasonal moving-average polynomial is non-invertible (roots on or inside the unit circle)."
            )
        }
    }
}
