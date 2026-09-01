// StatsBenchmarks.swift
// Benchmarks for SwiftStats vs NumPy vector operations.
// All computations run on 1,000,000-element Double arrays backed by vDSP (Accelerate).

import Foundation
import SwiftStats

struct StatsBenchmarks: BenchmarkSuite {
    let module = "SwiftStats"

    /// Deterministic data via LCG (same seed as Python counterpart uses `numpy.random.seed(42)`).
    private static func makeVector(_ n: Int, seed: UInt64 = 42) -> [Double] {
        var rng = BenchmarkLCG(seed: seed)
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
            _ = try Stats.mean(data, checkNaN: false)
        }
        results.append(meanResult)

        // ── 2. Standard Deviation ─────────────────────────────────────────
        let stdResult = await BenchmarkRunner.run(
            name: "StdDev (vDSP, 1M elements)",
            module: module
        ) {
            _ = try Stats.standardDeviation(data, checkNaN: false)
        }
        results.append(stdResult)

        // ── 3. Variance ───────────────────────────────────────────────────
        let varResult = await BenchmarkRunner.run(
            name: "Variance (vDSP, 1M elements)",
            module: module
        ) {
            _ = try Stats.variance(data, checkNaN: false)
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

        // ── 5. Two-Sample T-Test (100k samples) ───────────────────────────
        let sample1 = StatsBenchmarks.makeVector(100_000, seed: 101)
        let sample2 = StatsBenchmarks.makeVector(100_000, seed: 202)
        let ttestResult = await BenchmarkRunner.run(
            name: "Two-Sample T-Test (100k)",
            module: module
        ) {
            _ = try Stats.tTest(sample1: sample1, sample2: sample2)
        }
        results.append(ttestResult)

        // ── 6. Spearman Rank Correlation (100k pairs) ─────────────────────
        let sDataA = StatsBenchmarks.makeVector(100_000, seed: 303)
        let sDataB = StatsBenchmarks.makeVector(100_000, seed: 404)
        let spearmanResult = await BenchmarkRunner.run(
            name: "Spearman Correlation (100k)",
            module: module
        ) {
            _ = try Stats.spearmanCorrelation(sDataA, sDataB)
        }
        results.append(spearmanResult)

        return results
    }
}
