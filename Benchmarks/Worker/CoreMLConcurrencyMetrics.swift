import CryptoKit
import Foundation
import SwiftDataFrame
import SwiftSciBenchmarkSupport

struct CoreMLLatency: Codable, Sendable {
    var count = 0
    var totalSeconds = 0.0
    var minimumSeconds = Double.greatestFiniteMagnitude
    var maximumSeconds = 0.0
    // Upper edges: 1 microsecond * 1.08^i; last bucket includes overflow.
    var buckets = [Int](repeating: 0, count: 256)

    mutating func record(_ seconds: Double) {
        count += 1
        totalSeconds += seconds
        minimumSeconds = min(minimumSeconds, seconds)
        maximumSeconds = max(maximumSeconds, seconds)
        let raw = seconds <= 0.000001 ? 0 : Int(ceil(log(seconds / 0.000001) / log(1.08)))
        buckets[min(255, max(0, raw))] += 1
    }
}

struct CoreMLStressMemory: Codable, Sendable {
    let seconds: Double
    let residentBytes: UInt64
    let reservedBytes: Int
    let queuedCount: Int
    let thermalState: Int
}

func coreMLResultHash(_ values: PreparedNumericBatch) throws -> String {
    var digest = SHA256()
    for column in 0..<values.columnCount {
        var numbers: [Double] = []
        numbers.reserveCapacity(values.rowCount)
        for row in 0..<values.rowCount {
            guard let value = values[row, column], value.isFinite else {
                throw BenchmarkFailure("Missing or nonfinite concurrent output")
            }
            numbers.append(value)
        }
        numbers.withUnsafeBytes { digest.update(bufferPointer: $0) }
    }
    return digest.finalize().map { String(format: "%02x", $0) }.joined()
}
