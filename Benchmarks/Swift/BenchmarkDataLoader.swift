// BenchmarkDataLoader.swift
// Loads shared binary and CSV benchmark fixtures from Benchmarks/Data/
// to ensure true byte-level Apple-to-Apple parity with Python benchmarks.

import Foundation

public enum BenchmarkDataLoader {
    nonisolated(unsafe) private static var didWarnFallback = false

    /// Locates the `Benchmarks/Data` directory across runtime execution contexts
    /// (e.g. `swift run`, Xcode project, or CI runners).
    public static var dataDirectoryURL: URL? {
        let fileManager = FileManager.default

        // 1. Direct path relative to current working directory
        let cwdURL = URL(fileURLWithPath: fileManager.currentDirectoryPath)
        let candidate1 = cwdURL.appendingPathComponent("Benchmarks/Data")
        if fileManager.fileExists(atPath: candidate1.path) {
            return candidate1
        }

        // 2. Relative to the Swift project root if running from a subdirectory
        let candidate2 = cwdURL.appendingPathComponent("../Benchmarks/Data")
        if fileManager.fileExists(atPath: candidate2.path) {
            return candidate2.standardized
        }

        // 3. Fallback based on #filePath
        let sourceFile = URL(fileURLWithPath: #filePath)
        let candidate3 = sourceFile
            .deletingLastPathComponent() // Benchmarks/Swift/
            .deletingLastPathComponent() // Benchmarks/
            .appendingPathComponent("Data")
        if fileManager.fileExists(atPath: candidate3.path) {
            return candidate3.standardized
        }

        return nil
    }

    /// Loads an array of IEEE-754 little-endian `Double` values from a `.bin` file.
    public static func loadDoubleVector(
        filename: String,
        fallbackCount: Int,
        fallbackSeed: UInt64
    ) -> [Double] {
        if let dir = dataDirectoryURL {
            let fileURL = dir.appendingPathComponent(filename)
            if let data = try? Data(contentsOf: fileURL), data.count >= MemoryLayout<Double>.size {
                let count = data.count / MemoryLayout<Double>.size
                return data.withUnsafeBytes { rawBuffer in
                    let typedBuffer = rawBuffer.bindMemory(to: Double.self)
                    return Array(typedBuffer.prefix(count))
                }
            }
        }

        if !didWarnFallback {
            didWarnFallback = true
            print("⚠️ [BenchmarkDataLoader] Benchmarks/Data/\(filename) not found. Falling back to deterministic LCG generation.")
            print("   👉 Run 'python3 Benchmarks/generate_fixtures.py' for strict byte-level parity with Python.")
        }

        var rng = BenchmarkLCG(seed: fallbackSeed)
        return (0..<fallbackCount).map { _ in Double(rng.next() % 100_000) / 1000.0 - 50.0 }
    }

    /// Loads 2D matrix of shape [rows, cols] from a raw flat Double binary file.
    public static func load2DMatrix(
        filename: String,
        rows: Int,
        cols: Int,
        fallback: () -> [[Double]]
    ) -> [[Double]] {
        if let dir = dataDirectoryURL {
            let fileURL = dir.appendingPathComponent(filename)
            if let data = try? Data(contentsOf: fileURL), data.count >= rows * cols * MemoryLayout<Double>.size {
                return data.withUnsafeBytes { rawBuffer in
                    let typedBuffer = rawBuffer.bindMemory(to: Double.self)
                    var matrix = [[Double]]()
                    matrix.reserveCapacity(rows)
                    for r in 0..<rows {
                        let start = r * cols
                        let rowSlice = typedBuffer[start..<(start + cols)]
                        matrix.append(Array(rowSlice))
                    }
                    return matrix
                }
            }
        }
        return fallback()
    }

    /// Returns URL to a shared CSV fixture file or invokes the fallback generator.
    public static func csvURL(filename: String, fallbackGenerator: () throws -> URL) throws -> URL {
        if let dir = dataDirectoryURL {
            let fileURL = dir.appendingPathComponent(filename)
            if FileManager.default.fileExists(atPath: fileURL.path) {
                return fileURL
            }
        }
        return try fallbackGenerator()
    }
}
