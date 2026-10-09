import Foundation
import SwiftSciBenchmarkSupport
import SwiftDataFrame

private struct FusionSample: Encodable {
    let rows: Int
    let columns: Int
    let missing: Bool
    let mode: String
    let iteration: Int
    let seconds: Double
    let checksum: Double
    let thermalState: Int
}
private struct FusionReport: Encodable {
    let schemaVersion = 1
    let scope = "Synthetic held-out preprocessing and Float32 packing; no inference or transfer measured"
    let ownership = "All modes read the same shared immutable input; fitting excluded; output allocation included"
    let validationCases: Int
    let samples: [FusionSample]
}

func benchmarkFusedPreprocessing(output: URL) -> Int32 {
    do {
        let checks = try verifyFusedPreprocessing()
        var samples = [FusionSample]()
        let shapes: [(Int, Int)] = [(32, 8), (1024, 64), (8192, 64), (8192, 512), (65536, 64)]
        for (rows, width) in shapes {
            for missing in [false, true] {
                let training = try fusedTrialFixture(rows: 257, width: width, missing: missing)
                let plan = try FusedPreprocessingPlan(training: training)
                let input = try fusedTrialFixture(rows: rows, width: width, missing: missing, offset: 119)
                do {
                    let reference = try plan.staged(input, as: Float.self)
                    for tiled in [false, true] {
                        guard fusedTrialEqual(reference, try plan.fused(input, as: Float.self, tiled: tiled)) else {
                            throw BenchmarkFailure("Benchmark shape failed output equivalence")
                        }
                    }
                }
                for iteration in -2..<9 {
                    let modes = ["staged", "fused-row", "fused-tile"]
                    let rotation = (iteration + 2) % modes.count
                    for index in 0..<modes.count {
                        let mode = modes[(index + rotation) % modes.count]
                        let start = ContinuousClock.now
                        let values: [Float]
                        if mode == "staged" { values = try plan.staged(input, as: Float.self) }
                        else { values = try plan.fused(input, as: Float.self, tiled: mode == "fused-tile") }
                        let seconds = elapsedSeconds(since: start)
                        let checksum = values.reduce(0.0) { $0 + Double($1) }
                        if iteration >= 0 {
                            samples.append(FusionSample(rows: rows, columns: width, missing: missing,
                                mode: mode, iteration: iteration, seconds: seconds, checksum: checksum,
                                thermalState: ProcessInfo.processInfo.thermalState.rawValue))
                        }
                    }
                }
            }
        }
        let report = FusionReport(validationCases: checks, samples: samples)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: output, options: .atomic)
        print("PASS: \(checks) validation cases; \(samples.count) timing samples")
        return 0
    } catch {
        fputs("Fused preprocessing trial failed: \(error)\n", stderr)
        return 1
    }
}
