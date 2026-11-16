import Testing
import Foundation
@testable import SwiftForecast

@Suite("G-037 ARIMA and SARIMA Fitted Values Verification")
struct ARIMASARIMAFittedValuesTests {

    private static let airPassengers144: [Double] = [
        112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
        115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
        145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
        171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
        196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
        204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
        242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
        284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
        315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
        340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
        360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
        417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432
    ]

    private func causalARIMAPrediction(
        series: [Double],
        t: Int,
        d: Int,
        intercept: Double,
        ar: [Double],
        ma: [Double],
        residuals: [Double]
    ) -> Double {
        let p = ar.count
        let q = ma.count
        let startIdx = Swift.max(p, q)
        let warmUp = d
        let diffIdx = t - warmUp
        guard diffIdx >= startIdx else {
            return series[t]
        }

        func diffValue(at k: Int, order dOrder: Int) -> Double {
            switch dOrder {
            case 0:
                return series[k]
            case 1:
                return series[k] - series[k - 1]
            case 2:
                return (series[k] - series[k - 1]) - (series[k - 1] - series[k - 2])
            default:
                fatalError("Unsupported test order \(dOrder)")
            }
        }

        // Strict causal prediction of differenced w_t using only indices < t.
        var wHat = intercept
        for j in 0..<p {
            wHat += ar[j] * diffValue(at: t - 1 - j, order: d)
        }
        for j in 0..<q {
            wHat += ma[j] * residuals[diffIdx - 1 - j]
        }

        switch d {
        case 0:
            return wHat
        case 1:
            return series[t - 1] + wHat
        case 2:
            return 2.0 * series[t - 1] - series[t - 2] + wHat
        default:
            fatalError("Unsupported test order \(d)")
        }
    }

    private func causalSARIMAPrediction(
        series: [Double],
        t: Int,
        d: Int,
        D: Int,
        s: Int,
        intercept: Double,
        ar: [Double],
        sar: [Double],
        ma: [Double],
        sma: [Double],
        residuals: [Double]
    ) -> Double {
        let p = ar.count
        let P = sar.count
        let q = ma.count
        let Q = sma.count
        let maxLag = Swift.max(p + P * s, q + Q * s)
        let startIdx = Swift.max(maxLag, 1)
        let warmUp = d + D * s
        let diffIdx = t - warmUp
        guard diffIdx >= startIdx else {
            return series[t]
        }

        func w(at idx: Int) -> Double {
            if D == 0 && d == 0 {
                return series[idx]
            } else if D == 0 && d == 1 {
                return series[idx] - series[idx - 1]
            } else if D == 1 && d == 0 {
                return series[idx] - series[idx - s]
            } else if D == 1 && d == 1 {
                return (series[idx] - series[idx - s]) - (series[idx - 1] - series[idx - 1 - s])
            } else if D == 2 && d == 0 {
                return (series[idx] - series[idx - s]) - (series[idx - s] - series[idx - 2 * s])
            } else if D == 2 && d == 1 {
                let z0 = (series[idx] - series[idx - s]) - (series[idx - s] - series[idx - 2 * s])
                let z1 = (series[idx - 1] - series[idx - 1 - s]) - (series[idx - 1 - s] - series[idx - 1 - 2 * s])
                return z0 - z1
            }
            fatalError("Unsupported (d, D)= (\(d), \(D)) in test helper")
        }

        var wHat = intercept
        for i in 0..<p {
            wHat += ar[i] * w(at: t - 1 - i)
        }
        for I in 0..<P {
            wHat += sar[I] * w(at: t - (I + 1) * s)
        }
        for i in 0..<p {
            for I in 0..<P {
                wHat -= ar[i] * sar[I] * w(at: t - 1 - i - (I + 1) * s)
            }
        }
        for j in 0..<q {
            wHat += ma[j] * residuals[diffIdx - 1 - j]
        }
        for J in 0..<Q {
            wHat += sma[J] * residuals[diffIdx - (J + 1) * s]
        }
        for j in 0..<q {
            for J in 0..<Q {
                wHat += ma[j] * sma[J] * residuals[diffIdx - 1 - j - (J + 1) * s]
            }
        }

        if D == 0 && d == 0 {
            return wHat
        } else if D == 0 && d == 1 {
            return series[t - 1] + wHat
        } else if D == 1 && d == 0 {
            return series[t - s] + wHat
        } else if D == 1 && d == 1 {
            return series[t - 1] + series[t - s] - series[t - 1 - s] + wHat
        } else if D == 2 && d == 0 {
            return 2.0 * series[t - s] - series[t - 2 * s] + wHat
        } else if D == 2 && d == 1 {
            let zPast = (series[t - 1] - series[t - 1 - s]) - (series[t - 1 - s] - series[t - 1 - 2 * s])
            return 2.0 * series[t - s] - series[t - 2 * s] + zPast + wHat
        }
        fatalError("Unsupported (d, D) = (\(d), \(D)) in test helper")
    }

    @Test("AirPassengers SARIMA(1,1,0)(0,1,0)[12] fittedValues match closed-loop one-step predictions and residuals")
    func testAirPassengersSARIMAFittedValuesAlignment() async throws {
        let series = Self.airPassengers144
        let model = try SARIMAModel(p: 1, d: 1, q: 0, P: 0, D: 1, Q: 0, s: 12)
        try await model.fit(series: series)

        let fitted = await model.fittedValues
        let residuals = await model.residuals
        let warmUp = series.count - residuals.count

        #expect(fitted.count == series.count)
        #expect(warmUp == 13)
        #expect(residuals.count == series.count - 13)

        for t in 0..<warmUp {
            #expect(abs(fitted[t] - series[t]) < 1e-12)
        }

        let ar = await model.arCoefficients
        let sar = await model.seasonalArCoefficients
        let ma = await model.maCoefficients
        let sma = await model.seasonalMaCoefficients
        let intercept = await model.intercept

        var maxIdentityDiff = 0.0
        var maxCausalDiff = 0.0
        for t in warmUp..<series.count {
            let identityErr = abs((series[t] - residuals[t - warmUp]) - fitted[t])
            maxIdentityDiff = Swift.max(maxIdentityDiff, identityErr)

            let causal = causalSARIMAPrediction(
                series: series,
                t: t,
                d: 1,
                D: 1,
                s: 12,
                intercept: intercept,
                ar: ar,
                sar: sar,
                ma: ma,
                sma: sma,
                residuals: residuals
            )
            maxCausalDiff = Swift.max(maxCausalDiff, abs(fitted[t] - causal))
        }

        #expect(maxIdentityDiff < 1e-10)
        #expect(maxCausalDiff < 1e-10)
        // Late-1960 values (index 139..143: 606, 508, 461, 390, 432) must remain near observed scale, not drifted near ~250
        for t in 139..<144 {
            #expect(abs(fitted[t] - series[t]) < 50.0)
        }
    }

    @Test("AirPassengers canonical SARIMA(0,1,1)(0,1,1)[12] and ARIMA(1,1,1) satisfy exact one-step identity without holdout regression")
    func testAirPassengersOtherOrdersAndHoldoutConsistency() async throws {
        let series = Self.airPassengers144
        let train = Array(series.prefix(120))

        // Canonical airline SARIMA(0,1,1)(0,1,1)[12]
        let airline = try SARIMAModel(p: 0, d: 1, q: 1, P: 0, D: 1, Q: 1, s: 12)
        try await airline.fit(series: train)
        let airFitted = await airline.fittedValues
        let airRes = await airline.residuals
        let airWarmUp = train.count - airRes.count
        #expect(airFitted.count == train.count)
        #expect(airWarmUp == 13)
        for t in airWarmUp..<train.count {
            #expect(abs((train[t] - airRes[t - airWarmUp]) - airFitted[t]) < 1e-10)
            let causal = causalSARIMAPrediction(
                series: train,
                t: t,
                d: 1,
                D: 1,
                s: 12,
                intercept: await airline.intercept,
                ar: await airline.arCoefficients,
                sar: await airline.seasonalArCoefficients,
                ma: await airline.maCoefficients,
                sma: await airline.seasonalMaCoefficients,
                residuals: airRes
            )
            #expect(abs(airFitted[t] - causal) < 1e-10)
        }
        // Holdout forecasts must remain unchanged vs v3.13.2 baseline
        let airFc = try await airline.forecast(steps: 24)
        #expect(abs(airFc[0] - 342.7946704250881) < 1e-8)
        #expect(abs(airFc[1] - 321.4323556998727) < 1e-8)
        #expect(abs(airFc[2] - 366.15239606803044) < 1e-8)

        // Non-seasonal ARIMA(1,1,1) on full AirPassengers
        let arima = try ARIMAModel(p: 1, d: 1, q: 1)
        try await arima.fit(series: series)
        let arimaRes = try await arima.forecast(horizon: 6)
        let aFitted = arimaRes.forecast.fittedValues
        let aResid = arimaRes.forecast.residuals
        let aWarmUp = series.count - aResid.count
        #expect(aFitted.count == series.count)
        #expect(aWarmUp == 1)
        for t in aWarmUp..<series.count {
            #expect(abs((series[t] - aResid[t - aWarmUp]) - aFitted[t]) < 1e-10)
            let causal = causalARIMAPrediction(
                series: series,
                t: t,
                d: 1,
                intercept: arimaRes.intercept,
                ar: arimaRes.arCoefficients,
                ma: arimaRes.maCoefficients,
                residuals: aResid
            )
            #expect(abs(aFitted[t] - causal) < 1e-10)
        }
    }

    @Test("Synthetic ARIMA (d=0,1,2 and ARIMAX) and SARIMA (seasonal differencing) satisfy causal one-step identity")
    func testSyntheticDifferencingOrdersAndExog() async throws {
        var rng = TestLCG(seed: 77)
        let n = 120
        let seriesD2 = (0..<n).map { t in
            20.0 + 0.5 * Double(t) + 0.02 * Double(t * t) + 2.0 * sin(Double(t) * 0.4) + rng.nextDouble(in: -0.5...0.5)
        }

        // ARIMA(1, 2, 1) (d = 2)
        let arimaD2 = try ARIMAModel(p: 1, d: 2, q: 1)
        try await arimaD2.fit(series: seriesD2)
        let fD2 = await arimaD2.fittedValues
        let rD2 = await arimaD2.residuals
        #expect(fD2.count == n)
        #expect(rD2.count == n - 2)
        for t in 2..<n {
            #expect(abs((seriesD2[t] - rD2[t - 2]) - fD2[t]) < 1e-9)
            let causal = causalARIMAPrediction(
                series: seriesD2,
                t: t,
                d: 2,
                intercept: await arimaD2.intercept,
                ar: await arimaD2.arCoefficients,
                ma: await arimaD2.maCoefficients,
                residuals: rD2
            )
            #expect(abs(fD2[t] - causal) < 1e-9)
        }

        // ARIMA(0, 1, 0) (trivial random-walk + drift path)
        let arima010 = try ARIMAModel(p: 0, d: 1, q: 0)
        try await arima010.fit(series: seriesD2)
        let f010 = await arima010.fittedValues
        let r010 = await arima010.residuals
        #expect(f010.count == n)
        #expect(r010.count == n - 1)
        for t in 1..<n {
            #expect(abs((seriesD2[t] - r010[t - 1]) - f010[t]) < 1e-10)
        }

        // ARIMAX(1, 1, 0) and ARIMAX(0, 1, 0) return fittedValues on original scale with exact residual identity
        let exog = (0..<n).map { [Double($0) * 0.25, sin(Double($0) * 0.3)] }
        let yExog = (0..<n).map { t in
            12.0 + 3.0 * exog[t][0] - 4.0 * exog[t][1] + seriesD2[t] * 0.1
        }
        for (p, q) in [(1, 0), (0, 0)] {
            let arimax = try ARIMAModel(p: p, d: 1, q: q)
            try await arimax.fit(series: yExog, exog: exog)
            let fx = await arimax.fittedValues
            let rx = await arimax.residuals
            #expect(fx.count == n)
            #expect(rx.count == n - 1)
            for t in 0..<n {
                #expect(abs(fx[t] - yExog[t]) < 10.0)
            }
            for t in 1..<n {
                #expect(abs((yExog[t] - rx[t - 1]) - fx[t]) < 1e-9)
            }
        }

        // SARIMA(1, 0, 0)(0, 1, 0)[4] (D = 1, d = 0) and SARIMA(1, 0, 0)(0, 2, 0)[4] (D = 2, d = 0)
        let s = 4
        let seasonalY = (0..<n).map { t in
            50.0 + 0.8 * Double(t) + [4.0, -3.0, 7.0, -2.0][t % s] + rng.nextDouble(in: -0.6...0.6)
        }
        for (D, Q) in [(1, 0), (2, 0)] {
            let sarima = try SARIMAModel(p: 1, d: 0, q: 0, P: 0, D: D, Q: Q, s: s)
            try await sarima.fit(series: seasonalY)
            let sf = await sarima.fittedValues
            let sr = await sarima.residuals
            let warmUp = D * s
            #expect(sf.count == n)
            #expect(sr.count == n - warmUp)
            for t in 0..<warmUp {
                #expect(abs(sf[t] - seasonalY[t]) < 1e-10)
            }
            for t in warmUp..<n {
                #expect(abs((seasonalY[t] - sr[t - warmUp]) - sf[t]) < 1e-9)
                let causal = causalSARIMAPrediction(
                    series: seasonalY,
                    t: t,
                    d: 0,
                    D: D,
                    s: s,
                    intercept: await sarima.intercept,
                    ar: await sarima.arCoefficients,
                    sar: await sarima.seasonalArCoefficients,
                    ma: await sarima.maCoefficients,
                    sma: await sarima.seasonalMaCoefficients,
                    residuals: sr
                )
                #expect(abs(sf[t] - causal) < 1e-9)
            }
        }
    }

    @Test("AutoARIMA fittedValues match closed-loop one-step predictions and non-leakage under future perturbation")
    func testAutoARIMAFittedValuesAndNoFutureLeakage() async throws {
        let series = Self.airPassengers144
        let auto = try AutoARIMA(
            maxP: 1, maxD: 1, maxQ: 1,
            seasonal: true, seasonalPeriod: 12,
            maxSeasonalP: 1, maxSeasonalD: 1, maxSeasonalQ: 1
        )
        let res = try await auto.fit(series: Array(series.prefix(120)))
        let train = Array(series.prefix(120))
        let fitted = res.forecast.fittedValues
        let residuals = res.forecast.residuals
        let warmUp = train.count - residuals.count

        #expect(fitted.count == train.count)
        for t in warmUp..<train.count {
            #expect(abs((train[t] - residuals[t - warmUp]) - fitted[t]) < 1e-9)
        }

        // Fixed-parameter one-step recursion at time t uses no series values at indices >= t.
        // Fit SARIMA(1,1,0)(0,1,0)[12] on perturbed final 12 points; index t=60 stays anchored
        // within 1-step horizon rather than accumulating future shifts into past fitted values.
        var perturbed = series
        perturbed[143] += 5000.0
        let sClean = try SARIMAModel(p: 1, d: 1, q: 0, P: 0, D: 1, Q: 0, s: 12)
        try await sClean.fit(series: series)
        let sPert = try SARIMAModel(p: 1, d: 1, q: 0, P: 0, D: 1, Q: 0, s: 12)
        try await sPert.fit(series: perturbed)
        let cleanFittedAt142 = (await sClean.fittedValues)[142]
        let cleanResAt142 = (await sClean.residuals)[142 - 13]
        // Exact closed-loop check right before perturbed point t=143:
        #expect(abs((series[142] - cleanResAt142) - cleanFittedAt142) < 1e-10)
        // Even after huge perturbation at t=143, fittedValues[143] still satisfies causal formula using only t<=142!
        let pertFitted = await sPert.fittedValues
        let pertRes = await sPert.residuals
        let pertCausal143 = causalSARIMAPrediction(
            series: perturbed,
            t: 143,
            d: 1,
            D: 1,
            s: 12,
            intercept: await sPert.intercept,
            ar: await sPert.arCoefficients,
            sar: await sPert.seasonalArCoefficients,
            ma: await sPert.maCoefficients,
            sma: await sPert.seasonalMaCoefficients,
            residuals: pertRes
        )
        #expect(abs(pertFitted[143] - pertCausal143) < 1e-9)
        // fittedValues[143] equals actual[143] - residual[143 - 13] identically even with +5000 jump at 143
        #expect(abs((perturbed[143] - pertRes[143 - 13]) - pertFitted[143]) < 1e-9)
    }
}

private struct TestLCG {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed }
    mutating func nextUniform() -> Double {
        state = 6364136223846793005 &* state &+ 1442695040888963407
        return Double(state >> 11) / Double(1 << 53)
    }
    mutating func nextDouble(in range: ClosedRange<Double>) -> Double {
        range.lowerBound + (range.upperBound - range.lowerBound) * nextUniform()
    }
}
