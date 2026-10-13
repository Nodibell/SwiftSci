import CoreML
import CryptoKit
import Foundation
import SwiftDataFrame
import SwiftML
import SwiftPreprocessing
import SwiftSciBenchmarkSupport

func mixedHash(_ columns: [[Double]]) -> String {
    var hash = SHA256()
    for column in columns { column.withUnsafeBytes { hash.update(bufferPointer: $0) } }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
}

struct MixedWorkloadFixture: Sendable {
    let input: PreparedNumericBatch
    let imputer: Imputer
    let scaler: StandardScaler
    let plan: StandardPreprocessingPlan
    let regression: LinearRegression
    let cpuHash: String
    let gpuHash: String
    let file: URL
    let model: URL
    let configuration: CoreMLBatchPipeline.Configuration

    static func make(_ q: MixedWorkloadRequest, file: URL, model: URL) async throws -> Self {
        let raw = try JSONDecoder().decode(FusionRawFixture.self, from: Data(contentsOf: URL(fileURLWithPath: q.rawFixture)))
        let names: [String] = (0..<54).map { "feature_\($0)" }
        guard raw.trainingColumns.count == 54, raw.heldOutColumns.count == 54,
              raw.trainingColumns.allSatisfy({ $0.count == 11340 && $0.allSatisfy(\.isFinite) }),
              raw.heldOutColumns.allSatisfy({ $0.count == 8192 && $0.allSatisfy(\.isFinite) }) else {
            throw BenchmarkFailure("Unexpected Covertype fixture")
        }
        let training = try PreparedNumericBatch(columnNames: names, columns: raw.trainingColumns)
        let columns: [[Double]] = raw.heldOutColumns.map { Array($0.prefix(q.rows)) }
        let input = try PreparedNumericBatch(columnNames: names, columns: columns)
        var imputer = Imputer(strategy: .mean)
        try imputer.fit(training)
        var scaler = StandardScaler()
        try scaler.fit(imputer.transform(training))
        let transformed = try scaler.transform(imputer.transform(input))
        let cpuHash = mixedHash(transformed.columns.map(\.values))
        // Check the prepared transform against the public row-array path before timing.
        let rowResult = try scaler.transform(imputer.transform(input.rowValues()))
        for r in 0..<q.rows { for c in 0..<54 {
            guard transformed.columns[c].values[r] == rowResult[r][c] else {
                throw BenchmarkFailure("Prepared preprocessing differs from row reference")
            }
        } }
        let weights: [Double] = (0..<54).map { c in Double(c % 7 - 3) / 64.0 }
        let regression = LinearRegression(weights: weights, bias: 0.125, device: .gpu)
        let output = try await regression.predict(features: input)
        guard await regression.resolvedDevice == .gpu else { throw BenchmarkFailure("GPU regression route unavailable") }
        // Independent Double accumulation of Float inputs and parameters, with a Float rounding bound.
        let u = Double(Float.ulpOfOne) / 2
        let operations = Double(2 * weights.count + 2)
        let gamma = operations * u / (1 - operations * u)
        for r in 0..<q.rows {
            var reference = 0.125
            var magnitude = 0.125
            for c in weights.indices {
                let term = Double(Float(columns[c][r])) * Double(Float(weights[c]))
                reference += term
                magnitude += abs(term)
            }
            guard output[r].isFinite, abs(output[r] - reference) <= gamma * magnitude else {
                throw BenchmarkFailure("GPU prediction exceeds Float rounding bound")
            }
        }
        try writeBoundedCoreMLSource(raw, rows: q.rows, batches: q.batches, to: file)
        let pool = try CoreMLMatrixPool.Configuration(maximumConcurrentPredictions: 1,
            retainedBytes: 33_554_432, requestBytes: 16_777_216)
        let config = try CoreMLBatchPipeline.Configuration(pool: pool, maximumInFlightBatches: 2,
            retainedBytes: 1_048_576, batchBytes: 33_554_432)
        return try Self(input: input, imputer: imputer, scaler: scaler,
            plan: StandardPreprocessingPlan(training: training), regression: regression,
            cpuHash: cpuHash, gpuHash: mixedHash([output]), file: file, model: model, configuration: config)
    }
}
