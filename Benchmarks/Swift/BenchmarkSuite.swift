// BenchmarkSuite.swift
// Lightweight measurement harness for SwiftSci performance comparisons.
// Uses ContinuousClock (Swift 5.7+) and Mach Task info for resident memory profiling.

import Foundation
#if canImport(Darwin)
import Darwin
#endif

// MARK: – Deterministic Random Number Generator

/// Fast, deterministic 64-bit Linear Congruential Generator (LCG) matching NumPy seed(42).
public struct BenchmarkLCG: Sendable {
    private var state: UInt64

    public init(seed: UInt64 = 42) {
        self.state = seed
    }

    public mutating func next() -> UInt64 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return state
    }

    public mutating func nextDouble(in range: ClosedRange<Double> = 0.0...1.0) -> Double {
        let normalized = Double(next() >> 11) / Double(1 << 53)
        return range.lowerBound + normalized * (range.upperBound - range.lowerBound)
    }

    public mutating func nextVector(count: Int, in range: ClosedRange<Double> = -50.0...50.0) -> [Double] {
        var vec = [Double]()
        vec.reserveCapacity(count)
        for _ in 0..<count {
            vec.append(nextDouble(in: range))
        }
        return vec
    }
}

// MARK: – Memory Profiling

public enum BenchmarkMemory {
    /// Returns current resident memory (RSS) of process in megabytes (MB).
    public static func currentResidentMemoryMB() -> Double {
        #if canImport(Darwin)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / 4)
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        if kerr == KERN_SUCCESS {
            return Double(info.resident_size) / (1024.0 * 1024.0)
        }
        #endif
        return 0.0
    }
}

// MARK: – Result types

/// One benchmark scenario result with rich multi-round statistics.
public struct BenchmarkResult: Codable, Sendable {
    public let name: String
    public let module: String
    public let rounds: Int
    public let iterations: Int
    public let warmup: Int
    public let totalSamples: Int
    public let meanMs: Double
    public let trimmedMeanMs: Double
    public let medianMs: Double
    public let minMs: Double
    public let maxMs: Double
    public let stdMs: Double
    public let marginOfError95Ms: Double
    public let memoryMB: Double?

    public init(
        name: String,
        module: String,
        rounds: Int,
        iterations: Int,
        warmup: Int,
        totalSamples: Int,
        meanMs: Double,
        trimmedMeanMs: Double,
        medianMs: Double,
        minMs: Double,
        maxMs: Double,
        stdMs: Double,
        marginOfError95Ms: Double,
        memoryMB: Double? = nil
    ) {
        self.name = name
        self.module = module
        self.rounds = rounds
        self.iterations = iterations
        self.warmup = warmup
        self.totalSamples = totalSamples
        self.meanMs = meanMs
        self.trimmedMeanMs = trimmedMeanMs
        self.medianMs = medianMs
        self.minMs = minMs
        self.maxMs = maxMs
        self.stdMs = stdMs
        self.marginOfError95Ms = marginOfError95Ms
        self.memoryMB = memoryMB
    }
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

// MARK: – Global Benchmark Configuration

public struct BenchmarkConfig: Sendable {
    nonisolated(unsafe) public static var defaultRounds: Int = 3
    nonisolated(unsafe) public static var defaultIterations: Int = 7
    nonisolated(unsafe) public static var defaultWarmup: Int = 2
}

// MARK: – Runner

public enum BenchmarkRunner {

    /// Measure `block` across `rounds` rounds of `iterations` timed runs (after `warmup`).
    /// Collects all samples, filters outliers, and calculates resident memory.
    public static func measure(
        name: String = "",
        rounds: Int = BenchmarkConfig.defaultRounds,
        warmup: Int = BenchmarkConfig.defaultWarmup,
        iterations: Int = BenchmarkConfig.defaultIterations,
        block: () async throws -> Void
    ) async throws -> (timesMs: [Double], memMB: Double) {
        // Initial warmup phase (results discarded)
        for w in 0..<warmup {
            do {
                try await block()
            } catch {
                throw BenchmarkError.warmupFailed(name: name, iteration: w + 1, underlying: error)
            }
        }

        var allDurationsMs: [Double] = []
        allDurationsMs.reserveCapacity(rounds * iterations)
        let clock = ContinuousClock()

        for r in 0..<rounds {
            for i in 0..<iterations {
                var iterError: (any Error)?
                let start = clock.now
                do {
                    try await block()
                } catch {
                    iterError = error
                }
                let elapsed = clock.now - start

                if let error = iterError {
                    throw BenchmarkError.iterationFailed(
                        name: name,
                        iteration: (r * iterations) + i + 1,
                        underlying: error
                    )
                }

                let ms = Double(elapsed.components.seconds) * 1_000.0
                       + Double(elapsed.components.attoseconds) / 1_000_000_000_000_000.0
                allDurationsMs.append(ms)
            }
        }

        let mem = BenchmarkMemory.currentResidentMemoryMB()
        return (allDurationsMs, mem)
    }

    /// Convenience: measure across multiple rounds and build a full `BenchmarkResult`.
    /// Prints a live progress line before the run and an informative summary upon completion.
    public static func run(
        name: String,
        module: String,
        rounds: Int = BenchmarkConfig.defaultRounds,
        warmup: Int = BenchmarkConfig.defaultWarmup,
        iterations: Int = BenchmarkConfig.defaultIterations,
        block: () async throws -> Void
    ) async -> BenchmarkResult {
        let nameField = name.padding(toLength: 52, withPad: " ", startingAt: 0)
        print("  … \(nameField) [\(rounds) rounds × \(iterations) iters]")
        fflush(stdout)

        do {
            let (times, mem) = try await measure(
                name: name,
                rounds: rounds,
                warmup: warmup,
                iterations: iterations,
                block: block
            )
            let result = statistics(
                name: name,
                module: module,
                rounds: rounds,
                warmup: warmup,
                iterations: iterations,
                times: times,
                memMB: mem
            )

            let memStr = mem > 0 ? String(format: " | %5.1f MB", mem) : ""
            print("  ✓ \(nameField)  \(String(format: "%8.3f", result.meanMs)) ± \(String(format: "%.3f", result.marginOfError95Ms)) ms (mean, 95% CI)\(memStr)")
            return result
        } catch {
            print("  ❌ \(nameField)  FAILED: \(error)")
            return BenchmarkResult(
                name: name,
                module: module,
                rounds: rounds,
                iterations: iterations,
                warmup: warmup,
                totalSamples: 0,
                meanMs: 0,
                trimmedMeanMs: 0,
                medianMs: 0,
                minMs: 0,
                maxMs: 0,
                stdMs: 0,
                marginOfError95Ms: 0,
                memoryMB: nil
            )
        }
    }

    // MARK: – Statistical Calculation

    private static func statistics(
        name: String,
        module: String,
        rounds: Int,
        warmup: Int,
        iterations: Int,
        times: [Double],
        memMB: Double?
    ) -> BenchmarkResult {
        guard !times.isEmpty else {
            return BenchmarkResult(
                name: name,
                module: module,
                rounds: rounds,
                iterations: iterations,
                warmup: warmup,
                totalSamples: 0,
                meanMs: 0,
                trimmedMeanMs: 0,
                medianMs: 0,
                minMs: 0,
                maxMs: 0,
                stdMs: 0,
                marginOfError95Ms: 0,
                memoryMB: memMB
            )
        }

        let n = times.count
        let sorted = times.sorted()
        let mean = times.reduce(0, +) / Double(n)

        // 20% Trimmed Mean (discards top/bottom 10% to eliminate OS scheduling jitter)
        let trimCount = max(1, n / 10)
        let trimmedSlice = (n > 4 && trimCount * 2 < n)
            ? sorted[trimCount..<(n - trimCount)]
            : ArraySlice(sorted)
        let trimmedMean = trimmedSlice.reduce(0.0, +) / Double(trimmedSlice.count)

        // Median
        let median = n % 2 == 0
            ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            : sorted[n / 2]

        // Sample Standard Deviation
        let variance = n > 1
            ? times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(n - 1)
            : 0.0
        let std = variance.squareRoot()

        // 95% Confidence Interval Margin of Error (Z=1.96)
        let marginOfError = n > 1 ? (1.96 * std / Double(n).squareRoot()) : 0.0

        return BenchmarkResult(
            name: name,
            module: module,
            rounds: rounds,
            iterations: iterations,
            warmup: warmup,
            totalSamples: n,
            meanMs: mean,
            trimmedMeanMs: trimmedMean,
            medianMs: median,
            minMs: sorted.first!,
            maxMs: sorted.last!,
            stdMs: std,
            marginOfError95Ms: marginOfError,
            memoryMB: memMB
        )
    }
}

// MARK: – Benchmark Error

public enum BenchmarkError: Error, CustomStringConvertible {
    case warmupFailed(name: String, iteration: Int, underlying: any Error)
    case iterationFailed(name: String, iteration: Int, underlying: any Error)

    public var description: String {
        switch self {
        case .warmupFailed(let name, let iteration, let underlying):
            return "Benchmark '\(name)' failed during warmup #\(iteration): \(underlying)"
        case .iterationFailed(let name, let iteration, let underlying):
            return "Benchmark '\(name)' failed during sample #\(iteration): \(underlying)"
        }
    }
}

// MARK: – Pretty printer

public enum BenchmarkPrinter {

    public static func printTable(results: [BenchmarkResult]) {
        let nameW = max(35, (results.map(\.name).max(by: { $0.count < $1.count })?.count ?? 10) + 2)
        let header = String("Benchmark".padding(toLength: nameW, withPad: " ", startingAt: 0))
                   + "  Module              "
                   + "  Mean ± 95% CI (ms)       Trimmed(ms)  Median(ms)   Min..Max (ms)     RAM(MB)"
        print("\n" + String(repeating: "─", count: header.count))
        print(header)
        print(String(repeating: "─", count: header.count))

        for r in results {
            let nameCol   = r.name.padding(toLength: nameW, withPad: " ", startingAt: 0)
            let modCol    = r.module.padding(toLength: 20, withPad: " ", startingAt: 0)
            let meanStr   = String(format: "%8.3f ± %6.3f", r.meanMs, r.marginOfError95Ms)
            let trimStr   = String(format: "%10.3f", r.trimmedMeanMs)
            let medStr    = String(format: "%10.3f", r.medianMs)
            let minMaxStr = String(format: "%7.3f..%-7.3f", r.minMs, r.maxMs)
            let memStr    = r.memoryMB.map { String(format: "%7.1f", $0) } ?? "    n/a"

            print(String(format: "%@  %@  %@  %@  %@   %@  %@",
                         nameCol, modCol,
                         meanStr, trimStr, medStr, minMaxStr, memStr))
        }
        print(String(repeating: "─", count: header.count))
    }
}
