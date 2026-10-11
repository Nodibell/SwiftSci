import Foundation
import SwiftDataFrame
import SwiftSciBenchmarkSupport

// Exercises the actual file adapter on the cooperative executor without Core ML.
// The fixture cycle and checks remain fixed in size while file reads repeat.
func boundedSourceLifetime(request: URL, output: URL) async -> Int32 {
    do {
        let q = try JSONDecoder().decode(BoundedCoreMLRequest.self, from: Data(contentsOf: request))
        guard q.rows == 8192, q.batches == 128 else {
            throw BenchmarkFailure("Source lifetime check requires 128 batches of 8192 rows")
        }
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        let (plan, _, _) = try prepareBoundedFixture(q, file: file)
        let measurement = Task.detached {
            let source = try BoundedCoreMLSource(url: file, names: plan.columnNames,
                rows: q.rows, batches: q.batches, delayMilliseconds: 0)
            let before = try residentBytes()
            var checksum = 0.0
            for sequence in 0..<q.batches {
                guard let batch = try await source.load(sequence), let value = batch[0, 0] else {
                    throw BenchmarkFailure("Missing lifetime-check source batch")
                }
                checksum += value
            }
            let peak = await source.sampledResidentBytes
            try await source.close()
            let growth = peak > before ? peak - before : 0
            return ["initialResidentBytes": Double(before), "peakResidentBytes": Double(peak),
                    "growthBytes": Double(growth), "checksum": checksum]
        }
        let record = try await measurement.value
        try JSONEncoder().encode(record).write(to: output, options: .atomic)
        guard record["growthBytes"]! < 67_108_864 else {
            throw BenchmarkFailure("File adapter retained more than 64 MiB across repeated decoding")
        }
        return 0
    } catch { fputs("Bounded source lifetime check failed: \(error)\n", stderr); return 1 }
}
