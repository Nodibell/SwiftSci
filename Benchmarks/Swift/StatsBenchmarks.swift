// StatsBenchmarks.swift
// Benchmarks for SwiftStats vs NumPy vector operations.
// All computations run on 1,000,000-element Double arrays backed by vDSP (Accelerate).

import Foundation
import SwiftStats

struct StatsBenchmarks: BenchmarkSuite {
    let module = "SwiftStats"

    /// Deterministic data via LCG (same seed as Python counterpart uses `numpy.random.seed(42)`).
    private static func makeVector(_ n: Int, seed: UInt64 = 42) -> [Double] {
        var rng = LCGStats(seed: seed)
        return (0..<n).map { _ in Double(rng.next() % 100_000) / 1000.0 - 50.0 }
    }

    func run() async -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        let n = 1_000_000
        let data = StatsBenchmarks.makeVector(n)

        // ── 1. Mean (vDSP.mean) ───────────────────────────────────────────
        let meanResult = await BenchmarkRunner.run(
            name: "Mean (vDSP, 1M elements)",
            module: module
        ) {
            _ = try Stats.mean(data)
        }
        results.append(meanResult)

        // ── 2. Standard Deviation ─────────────────────────────────────────
        let stdResult = await BenchmarkRunner.run(
            name: "StdDev (vDSP, 1M elements)",
            module: module
        ) {
            _ = try Stats.standardDeviation(data)
        }
        results.append(stdResult)

        // ── 3. Variance ───────────────────────────────────────────────────
        let varResult = await BenchmarkRunner.run(
            name: "Variance (vDSP, 1M elements)",
            module: module
        ) {
            _ = try Stats.variance(data)
        }
        results.append(varResult)

        // ── 4. Pearson Correlation (500k pairs) ───────────────────────────
        let dataB = StatsBenchmarks.makeVector(500_000, seed: 99)
        let dataA = StatsBenchmarks.makeVector(500_000, seed: 42)
        let corrResult = await BenchmarkRunner.run(
            name: "Pearson Correlation (500k)",
            module: module
        ) {
            _ = try Stats.pearsonCorrelation(dataA, dataB)
        }
        results.append(corrResult)

        return results
    }
}

// Separate struct name to avoid conflict with DataFrameBenchmarks.LCG
private struct LCGStats {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return state
    }
}
