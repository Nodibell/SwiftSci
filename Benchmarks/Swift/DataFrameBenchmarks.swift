// DataFrameBenchmarks.swift
// Performance benchmarks for SwiftDataFrame operations:
//   • CSV read (100 k rows, 5 columns)
//   • Row filter (predicate on Double column)
//   • groupBy + aggregation (sum, mean, count)
// Compare with Python baseline in Benchmarks/Python/benchmarks.py [section: dataframe]

import Foundation
import SwiftDataFrame

struct DataFrameBenchmarks: BenchmarkSuite {
    let module = "SwiftDataFrame"

    // MARK: – Synthetic data

    /// Generates a 100 000-row CSV string at runtime.
    private static func makeCSVData(rows: Int = 100_000) -> String {
        var lines: [String] = ["id,category,value_a,value_b,flag"]
        let categories = ["alpha", "beta", "gamma", "delta"]
        var rng = LCG(seed: 42)
        for i in 0..<rows {
            let cat   = categories[Int(rng.next() % 4)]
            let va    = Double(rng.next() % 10_000) / 100.0
            let vb    = Double(rng.next() % 5_000)  / 100.0
            let flag  = i % 2 == 0 ? "true" : "false"
            lines.append("\(i),\(cat),\(va),\(vb),\(flag)")
        }
        return lines.joined(separator: "\n")
    }

    // MARK: – Benchmark scenarios

    /// Writes data to a temp file once; returns its URL for all CSV-read iterations.
    private func tempCSVURL() throws -> URL {
        let url = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("swiftanalytics_bench_\(UUID().uuidString).csv")
        let data = DataFrameBenchmarks.makeCSVData()
        try data.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    func run() async -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []

        // ── 1. CSV Read ────────────────────────────────────────────────────
        let csvURL: URL
        do {
            csvURL = try tempCSVURL()
        } catch {
            print("  [DataFrameBenchmarks] Could not write temp CSV: \(error)")
            return []
        }

        let csvRead = await BenchmarkRunner.run(
            name: "CSV Read (100k rows, 5 cols)",
            module: module,
            warmup: 1,
            iterations: 5
        ) {
            _ = try await DataFrame(csv: csvURL)
        }
        results.append(csvRead)

        // Pre-build an in-memory DataFrame for subsequent benchmarks
        guard let df = try? await DataFrame(csv: csvURL) else {
            print("  [DataFrameBenchmarks] Could not load DataFrame — skipping remaining benchmarks")
            try? FileManager.default.removeItem(at: csvURL)
            return results
        }

        // ── 2. Row Filter ──────────────────────────────────────────────────
        // Mirrors pandas `df[df["value_a"] > 50]` via the typed column path.
        let filterResult = await BenchmarkRunner.run(
            name: "Filter rows (predicate, 100k rows)",
            module: module
        ) {
            _ = try? df.filter(column: "value_a", where: .greaterThan(50.0))
        }
        results.append(filterResult)

        // ── 3. groupBy + sum/mean ──────────────────────────────────────────
        let groupResult = await BenchmarkRunner.run(
            name: "GroupBy + sum/mean (4 groups)",
            module: module
        ) {
            _ = df.groupBy("category").agg(["value_a": .sum, "value_b": .mean])
        }
        results.append(groupResult)

        // ── 4. sortBy ──────────────────────────────────────────────────────
        let sortResult = await BenchmarkRunner.run(
            name: "SortBy double column (100k rows)",
            module: module
        ) {
            _ = try? df.sortBy("value_a", ascending: true)
        }
        results.append(sortResult)

        try? FileManager.default.removeItem(at: csvURL)
        return results
    }
}

// MARK: – Minimal LCG for reproducible data generation (no Foundation.arc4random needed)

private struct LCG {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return state
    }
}
