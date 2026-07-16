// main.swift
// Entry point for the SwiftAnalytics benchmark runner.
//
// Usage:
//   swift run -c release SwiftAnalyticsBenchmarks                  # console output only
//   swift run -c release SwiftAnalyticsBenchmarks --json results.json # + write JSON file
//
// The JSON file can be compared against Benchmarks/Python/python_results.json
// using the compare.py script.

import Foundation

// MARK: – CLI argument parsing

struct BenchmarkArgs {
    var jsonOutputPath: String? = nil
    var filter: String? = nil      // optional: run only benchmarks whose name contains this string

    static func parse() -> BenchmarkArgs {
        var args = BenchmarkArgs()
        let argv = CommandLine.arguments.dropFirst()
        var iter = argv.makeIterator()
        while let arg = iter.next() {
            switch arg {
            case "--json":
                args.jsonOutputPath = iter.next()
            case "--filter":
                args.filter = iter.next()
            default:
                break
            }
        }
        return args
    }
}

// MARK: – Main

@main
struct BenchmarkEntryPoint {
    static func main() async {
        let args = BenchmarkArgs.parse()

        print("╔════════════════════════════════════════════════════╗")
        print("║      SwiftAnalytics Benchmark Suite — v0.7         ║")
        print("╚════════════════════════════════════════════════════╝")
        print("Platform : \(platformString())")
        print("Swift    : \(swiftVersion())")
        print("Config   : Release")
        if let filter = args.filter {
            print("Filter   : \"\(filter)\"")
        }
        print("")

        // ── Collect all suites ─────────────────────────────────────────────
        let suites: [any BenchmarkSuite] = [
            StatsBenchmarks(),
            DataFrameBenchmarks(),
            MLBenchmarks(),
            ForecastBenchmarks(),
        ]

        var allResults: [BenchmarkResult] = []

        for suite in suites {
            print("▶ Running \(suite.module) benchmarks …")
            let results = await suite.run()

            // Apply optional name filter (live ✓ lines are printed by BenchmarkRunner)
            let filtered = args.filter.map { f in results.filter { $0.name.lowercased().contains(f.lowercased()) } } ?? results
            allResults.append(contentsOf: filtered)
            print("")
        }

        // ── Pretty table ───────────────────────────────────────────────────
        BenchmarkPrinter.printTable(results: allResults)

        // ── JSON export ────────────────────────────────────────────────────
        if let path = args.jsonOutputPath {
            let report = BenchmarkReport(
                platform: platformString(),
                swiftVersion: swiftVersion(),
                timestamp: ISO8601DateFormatter().string(from: Date()),
                results: allResults
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            do {
                let data = try encoder.encode(report)
                let url: URL
                if path.hasPrefix("/") {
                    url = URL(filePath: path)
                } else {
                    url = URL(filePath: FileManager.default.currentDirectoryPath)
                              .appending(path: path)
                }
                try data.write(to: url)
                print("\n✅ Results exported to: \(url.path)\n")
            } catch {
                print("\n⚠️  Could not write JSON: \(error)\n")
            }
        }
    }

    // MARK: – Environment helpers

    private static func platformString() -> String {
        #if arch(arm64)
        "Apple Silicon (arm64)"
        #elseif arch(x86_64)
        "Intel (x86_64)"
        #else
        "unknown"
        #endif
    }

    private static func swiftVersion() -> String {
        // Injected by compiler at build time
        #if swift(>=6.0)
        return "Swift 6.x"
        #elseif swift(>=5.10)
        return "Swift 5.10"
        #else
        return "Swift <5.10"
        #endif
    }
}
