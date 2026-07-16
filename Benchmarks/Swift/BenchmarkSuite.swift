// BenchmarkSuite.swift
// Lightweight measurement harness for SwiftAnalytics performance comparisons.
// Uses ContinuousClock (Swift 5.7+) — no external dependencies.

import Foundation

// MARK: – Result types

/// One benchmark scenario result.
public struct BenchmarkResult: Codable {
    public let name: String
    public let module: String
    public let iterations: Int
    public let warmup: Int
    public let meanMs: Double
    public let medianMs: Double
    public let minMs: Double
    public let maxMs: Double
    public let stdMs: Double
}

/// Full report written to JSON output.
public struct BenchmarkReport: Codable {
    public let platform: String
    public let swiftVersion: String
    public let timestamp: String
    public let results: [BenchmarkResult]
}

// MARK: – Protocol

public protocol BenchmarkSuite {
    /// Human-readable name of the benchmark group (matches a Python section).
    var module: String { get }
    /// Run all contained benchmarks and return results.
    func run() async -> [BenchmarkResult]
}

// MARK: – Runner

public enum BenchmarkRunner {

    /// Measure `block` with `warmup` warm-up iterations then `iterations` timed runs.
    /// Returns individual durations in milliseconds.
    public static func measure(
        warmup: Int = 2,
        iterations: Int = 7,
        block: () async throws -> Void
    ) async -> [Double] {
        // Warm-up (results discarded)
        for _ in 0..<warmup {
            try? await block()
        }

        var durationsMs: [Double] = []
        durationsMs.reserveCapacity(iterations)
        let clock = ContinuousClock()

        for _ in 0..<iterations {
            let elapsed = await clock.measure { try? await block() }
            let ms = Double(elapsed.components.seconds) * 1_000.0
                   + Double(elapsed.components.attoseconds) / 1_000_000_000_000_000.0
            durationsMs.append(ms)
        }
        return durationsMs
    }

    /// Convenience: measure and build a full `BenchmarkResult`.
    /// Prints a live progress line before the run and a ✓ summary when it finishes.
    public static func run(
        name: String,
        module: String,
        warmup: Int = 2,
        iterations: Int = 7,
        block: () async throws -> Void
    ) async -> BenchmarkResult {
        let nameField = name.padding(toLength: 52, withPad: " ", startingAt: 0)
        print("  … \(nameField)")
        fflush(stdout)

        let times = await measure(warmup: warmup, iterations: iterations, block: block)
        let result = statistics(name: name, module: module,
                                warmup: warmup, iterations: iterations, times: times)

        print("  ✓ \(nameField)  \(String(format: "%8.3f", result.medianMs)) ms (median)")
        return result
    }

    // MARK: – Private statistics

    private static func statistics(
        name: String,
        module: String,
        warmup: Int,
        iterations: Int,
        times: [Double]
    ) -> BenchmarkResult {
        guard !times.isEmpty else {
            return BenchmarkResult(name: name, module: module,
                                   iterations: iterations, warmup: warmup,
                                   meanMs: 0, medianMs: 0, minMs: 0, maxMs: 0, stdMs: 0)
        }
        let sorted = times.sorted()
        let mean   = times.reduce(0, +) / Double(times.count)
        let median = sorted.count % 2 == 0
            ? (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
            : sorted[sorted.count / 2]
        let variance = times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(times.count)
        let std = variance.squareRoot()
        return BenchmarkResult(
            name: name,
            module: module,
            iterations: iterations,
            warmup: warmup,
            meanMs:   mean,
            medianMs: median,
            minMs:    sorted.first!,
            maxMs:    sorted.last!,
            stdMs:    std
        )
    }
}

// MARK: – Pretty printer

public enum BenchmarkPrinter {

    public static func printTable(results: [BenchmarkResult]) {
        let nameW = max(35, (results.map(\.name).max(by: { $0.count < $1.count })?.count ?? 10) + 2)
        let header = String("Benchmark".padding(toLength: nameW, withPad: " ", startingAt: 0))
                   + "  Module              "
                   + "  Mean(ms)  Median(ms)  Min(ms)  Max(ms)  Std(ms)"
        print("\n" + String(repeating: "─", count: header.count))
        print(header)
        print(String(repeating: "─", count: header.count))

        for r in results {
            let nameCol   = r.name.padding(toLength: nameW, withPad: " ", startingAt: 0)
            let modCol    = r.module.padding(toLength: 20, withPad: " ", startingAt: 0)
            print(String(format: "%@  %@  %9.3f  %10.3f  %8.3f  %8.3f  %7.3f",
                         nameCol, modCol,
                         r.meanMs, r.medianMs, r.minMs, r.maxMs, r.stdMs))
        }
        print(String(repeating: "─", count: header.count))
    }
}
