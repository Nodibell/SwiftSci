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

/// One benchmark scenario result.
public struct BenchmarkResult: Codable, Sendable {
    public let name: String
    public let module: String
    public let iterations: Int
    public let warmup: Int
    public let meanMs: Double
    public let medianMs: Double
    public let minMs: Double
    public let maxMs: Double
    public let stdMs: Double
    public let memoryMB: Double?

    public init(
        name: String,
        module: String,
        iterations: Int,
        warmup: Int,
        meanMs: Double,
        medianMs: Double,
        minMs: Double,
        maxMs: Double,
        stdMs: Double,
        memoryMB: Double? = nil
    ) {
        self.name = name
        self.module = module
        self.iterations = iterations
        self.warmup = warmup
        self.meanMs = meanMs
        self.medianMs = medianMs
        self.minMs = minMs
        self.maxMs = maxMs
        self.stdMs = stdMs
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

// MARK: – Runner

public enum BenchmarkRunner {

    /// Measure `block` with `warmup` warm-up iterations then `iterations` timed runs.
    /// Returns individual durations in milliseconds and peak memory observed.
    public static func measure(
        name: String = "",
        warmup: Int = 2,
        iterations: Int = 7,
        block: () async throws -> Void
    ) async throws -> (timesMs: [Double], memMB: Double) {
        // Warm-up
        for w in 0..<warmup {
            do {
                try await block()
            } catch {
                throw BenchmarkError.warmupFailed(name: name, iteration: w, underlying: error)
            }
        }

        var durationsMs: [Double] = []
        durationsMs.reserveCapacity(iterations)
        let clock = ContinuousClock()

        for i in 0..<iterations {
            var iterError: Error?
            let start = clock.now
            do {
                try await block()
            } catch {
                iterError = error
            }
            let elapsed = clock.now - start

            if let error = iterError {
                throw BenchmarkError.iterationFailed(name: name, iteration: i, underlying: error)
            }

            let ms = Double(elapsed.components.seconds) * 1_000.0
                   + Double(elapsed.components.attoseconds) / 1_000_000_000_000_000.0
            durationsMs.append(ms)
        }

        let mem = BenchmarkMemory.currentResidentMemoryMB()
        return (durationsMs, mem)
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

        do {
            let (times, mem) = try await measure(name: name, warmup: warmup, iterations: iterations, block: block)
            let result = statistics(name: name, module: module,
                                    warmup: warmup, iterations: iterations, times: times, memMB: mem)

            let memStr = mem > 0 ? String(format: " | %6.1f MB", mem) : ""
            print("  ✓ \(nameField)  \(String(format: "%8.3f", result.medianMs)) ms (median)\(memStr)")
            return result
        } catch {
            print("  ❌ \(nameField)  FAILED: \(error)")
            return BenchmarkResult(
                name: name,
                module: module,
                iterations: iterations,
                warmup: warmup,
                meanMs: 0,
                medianMs: 0,
                minMs: 0,
                maxMs: 0,
                stdMs: 0,
                memoryMB: nil
            )
        }
    }

    // MARK: – Private statistics

    private static func statistics(
        name: String,
        module: String,
        warmup: Int,
        iterations: Int,
        times: [Double],
        memMB: Double?
    ) -> BenchmarkResult {
        guard !times.isEmpty else {
            return BenchmarkResult(name: name, module: module,
                                   iterations: iterations, warmup: warmup,
                                   meanMs: 0, medianMs: 0, minMs: 0, maxMs: 0, stdMs: 0, memoryMB: memMB)
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
            stdMs:    std,
            memoryMB: memMB
        )
    }
}

// MARK: – Benchmark Error

public enum BenchmarkError: Error, CustomStringConvertible {
    case warmupFailed(name: String, iteration: Int, underlying: Error)
    case iterationFailed(name: String, iteration: Int, underlying: Error)

    public var description: String {
        switch self {
        case .warmupFailed(let name, let iteration, let underlying):
            return "Benchmark '\(name)' failed during warmup #\(iteration): \(underlying)"
        case .iterationFailed(let name, let iteration, let underlying):
            return "Benchmark '\(name)' failed during timed iteration #\(iteration): \(underlying)"
        }
    }
}

// MARK: – Pretty printer

public enum BenchmarkPrinter {

    public static func printTable(results: [BenchmarkResult]) {
        let nameW = max(35, (results.map(\.name).max(by: { $0.count < $1.count })?.count ?? 10) + 2)
        let header = String("Benchmark".padding(toLength: nameW, withPad: " ", startingAt: 0))
                   + "  Module              "
                   + "  Mean(ms)  Median(ms)  Min(ms)  Max(ms)  Std(ms)  RAM(MB)"
        print("\n" + String(repeating: "─", count: header.count))
        print(header)
        print(String(repeating: "─", count: header.count))

        for r in results {
            let nameCol   = r.name.padding(toLength: nameW, withPad: " ", startingAt: 0)
            let modCol    = r.module.padding(toLength: 20, withPad: " ", startingAt: 0)
            let memStr    = r.memoryMB.map { String(format: "%7.1f", $0) } ?? "    n/a"
            print(String(format: "%@  %@  %9.3f  %10.3f  %8.3f  %8.3f  %7.3f  %@",
                         nameCol, modCol,
                         r.meanMs, r.medianMs, r.minMs, r.maxMs, r.stdMs, memStr))
        }
        print(String(repeating: "─", count: header.count))
    }
}
