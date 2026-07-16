// ForecastBenchmarks.swift
// Benchmarks for SwiftForecast vs Statsmodels:
//   • Holt-Winters triple exponential smoothing (1k points, seasonal period = 12)
//   • ARIMA(1,1,1) parameter estimation (500 points)
//   • Kalman Filter 1D (10k observations)

import Foundation
import SwiftForecast

struct ForecastBenchmarks: BenchmarkSuite {
    let module = "SwiftForecast"

    // MARK: – Synthetic time-series generators

    /// Seasonal series: trend + sine seasonality + small noise.
    private static func makeSeasonal(n: Int, period: Int = 12, seed: UInt64 = 42) -> [Double] {
        var rng = LCGFC(seed: seed)
        return (0..<n).map { t in
            let trend    = Double(t) * 0.3
            let seasonal = 5.0 * sin(Double(t) * 2.0 * .pi / Double(period))
            let noise    = Double(rng.next() % 1000) / 1000.0 - 0.5
            return 20.0 + trend + seasonal + noise
        }
    }

    /// Random walk for ARIMA.
    private static func makeRandomWalk(n: Int, seed: UInt64 = 42) -> [Double] {
        var rng = LCGFC(seed: seed)
        var series = [0.0]
        for _ in 1..<n {
            let step = Double(rng.next() % 200) / 100.0 - 1.0
            series.append(series.last! + step)
        }
        return series
    }

    // MARK: – Run

    func run() async -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []

        // ── 1. Holt-Winters (1000 points, period = 12) ───────────────────
        let hwSeries = ForecastBenchmarks.makeSeasonal(n: 1_000, period: 12)
        let hwResult = await BenchmarkRunner.run(
            name: "Holt-Winters fit (1k pts, period=12)",
            module: module,
            warmup: 1,
            iterations: 5
        ) {
            let hw = ExponentialSmoothing(
                method: .holtWinters(beta: 0.1, gamma: 0.1, period: 12, seasonal: .additive),
                alpha: nil   // auto-optimize
            )
            try await hw.fit(series: hwSeries)
        }
        results.append(hwResult)

        // ── 2. ARIMA(1,1,1) — 500 points ─────────────────────────────────
        let arimaSeries = ForecastBenchmarks.makeRandomWalk(n: 500)
        let arimaResult = await BenchmarkRunner.run(
            name: "ARIMA(1,1,1) fit (500 pts)",
            module: module,
            warmup: 1,
            iterations: 5
        ) {
            let arima = try ARIMAModel(p: 1, d: 1, q: 1)
            try await arima.fit(series: arimaSeries)
        }
        results.append(arimaResult)

        // ── 3. ARIMA forecast (horizon = 24) ──────────────────────────────
        let arimaForecastResult = await BenchmarkRunner.run(
            name: "ARIMA(1,1,1) forecast horizon=24",
            module: module,
            warmup: 1,
            iterations: 5
        ) {
            let arima = try ARIMAModel(p: 1, d: 1, q: 1)
            try await arima.fit(series: arimaSeries)
            _ = try await arima.forecast(horizon: 24)
        }
        results.append(arimaForecastResult)

        // ── 4. Kalman Filter 1D (10k observations) ────────────────────────
        let noisyObs: [[Double]] = (0..<10_000).map { _ in [Double.random(in: 24.0...26.0)] }
        let kalmanResult = await BenchmarkRunner.run(
            name: "Kalman Filter 1D (10k obs)",
            module: module,
            warmup: 1,
            iterations: 5
        ) {
            let kf = try await KalmanFilter.oneDimensional(processNoise: 0.05, measurementNoise: 1.0)
            try await kf.setInitialState(mean: [25.0, 0.0], covariance: [[1.0, 0.0], [0.0, 1.0]])
            _ = try await kf.filter(observations: noisyObs)
        }
        results.append(kalmanResult)

        // ── 5. Time Series Decomposition (1000 points) ────────────────────
        let decompResult = await BenchmarkRunner.run(
            name: "TS Decomposition additive (1k pts)",
            module: module,
            warmup: 1,
            iterations: 7
        ) {
            _ = try TimeSeriesDecomposition.decompose(
                series: hwSeries,
                period: 12,
                model: .additive
            )
        }
        results.append(decompResult)

        return results
    }
}

private struct LCGFC {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return state
    }
}
