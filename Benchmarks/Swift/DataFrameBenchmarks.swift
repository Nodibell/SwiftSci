// DataFrameBenchmarks.swift
// Performance benchmarks for SwiftDataFrame operations:
//   • CSV read
//   • CSV streaming read
//   • Streaming filter
//   • Streaming groupBy
//   • In-memory filter
//   • In-memory groupBy
//   • Sort

import Foundation
import SwiftDataFrame

struct DataFrameBenchmarks: BenchmarkSuite {
    let module = "SwiftDataFrame"

    // MARK: - Synthetic data

    private static func makeCSVData(rows: Int = 100_000) -> String {
        var lines: [String] = ["id,category,value_a,value_b,flag"]

        let categories = ["alpha", "beta", "gamma", "delta"]
        var rng = LCG(seed: 42)

        for i in 0..<rows {
            let cat = categories[Int(rng.next() % 4)]
            let va = Double(rng.next() % 10_000) / 100.0
            let vb = Double(rng.next() % 5_000) / 100.0
            let flag = i % 2 == 0 ? "true" : "false"

            lines.append("\(i),\(cat),\(va),\(vb),\(flag)")
        }

        return lines.joined(separator: "\n")
    }

    // MARK: - Temp CSV

    private func tempCSVURL() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("swiftanalytics_bench_\(UUID().uuidString).csv")

        try DataFrameBenchmarks.makeCSVData()
            .write(to: url, atomically: true, encoding: .utf8)

        return url
    }

    // MARK: - Benchmarks

    func run() async -> [BenchmarkResult] {

        var results: [BenchmarkResult] = []

        let csvURL: URL

        do {
            csvURL = try tempCSVURL()
        } catch {
            print("Could not create temporary CSV: \(error)")
            return []
        }

        // -----------------------------------------------------------------
        // 1. CSV Read
        // -----------------------------------------------------------------

        let csvRead = await BenchmarkRunner.run(
            name: "CSV Read (100k rows)",
            module: module,
            warmup: 1,
            iterations: 5
        ) {
            _ = try await DataFrame(csv: csvURL)
        }

        results.append(csvRead)

        // -----------------------------------------------------------------
        // 2. CSV Streaming Read
        // -----------------------------------------------------------------

        let streamRead = await BenchmarkRunner.run(
            name: "CSV Stream Read (chunk=10k)",
            module: module,
            warmup: 1,
            iterations: 5
        ) {

            var rows = 0

            for try await chunk in DataFrame.readCSVStream(
                contentsOf: csvURL,
                chunkSize: 10_000
            ) {
                rows += chunk.shape.rows
            }

            precondition(rows == 100_000)
        }

        results.append(streamRead)

        // -----------------------------------------------------------------
        // 3. CSV Stream + Filter
        // -----------------------------------------------------------------

        let streamFilter = await BenchmarkRunner.run(
            name: "CSV Stream + Filter",
            module: module,
            warmup: 1,
            iterations: 5
        ) {

            var rows = 0

            for try await chunk in DataFrame.readCSVStream(
                contentsOf: csvURL,
                chunkSize: 10_000
            ) {

                let filtered = try chunk.filter(
                    column: "value_a",
                    where: .greaterThan(50.0)
                )

                rows += filtered.shape.rows
            }

            precondition(rows > 0)
        }

        results.append(streamFilter)

        // -----------------------------------------------------------------
        // 4. CSV Stream + GroupBy
        // -----------------------------------------------------------------

        let streamGroup = await BenchmarkRunner.run(
            name: "CSV Stream + GroupBy",
            module: module,
            warmup: 1,
            iterations: 5
        ) {

            for try await chunk in DataFrame.readCSVStream(
                contentsOf: csvURL,
                chunkSize: 10_000
            ) {

                _ = chunk
                    .groupBy("category")
                    .agg([
                        "value_a": .sum,
                        "value_b": .mean
                    ])
            }
        }

        results.append(streamGroup)

        // -----------------------------------------------------------------
        // Load DataFrame into memory
        // -----------------------------------------------------------------

        guard let df = try? await DataFrame(csv: csvURL) else {

            print("Failed to load DataFrame.")

            try? FileManager.default.removeItem(at: csvURL)

            return results
        }

        // -----------------------------------------------------------------
        // 5. Filter
        // -----------------------------------------------------------------

        let filterResult = await BenchmarkRunner.run(
            name: "Filter rows (100k)",
            module: module
        ) {

            _ = try? df.filter(
                column: "value_a",
                where: .greaterThan(50.0)
            )
        }

        results.append(filterResult)

        // -----------------------------------------------------------------
        // 6. GroupBy
        // -----------------------------------------------------------------

        let groupResult = await BenchmarkRunner.run(
            name: "GroupBy + Aggregation",
            module: module
        ) {

            _ = df.groupBy("category")
                .agg([
                    "value_a": .sum,
                    "value_b": .mean
                ])
        }

        results.append(groupResult)

        // -----------------------------------------------------------------
        // 7. Sort
        // -----------------------------------------------------------------

        let sortResult = await BenchmarkRunner.run(
            name: "SortBy double column",
            module: module
        ) {

            _ = try? df.sortBy(
                "value_a",
                ascending: true
            )
        }

        results.append(sortResult)

        try? FileManager.default.removeItem(at: csvURL)

        return results
    }
}

// MARK: - Deterministic RNG

private struct LCG {

    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {

        state = state
            &* 6_364_136_223_846_793_005
            &+ 1_442_695_040_888_963_407

        return state
    }
}
