import CryptoKit
import Darwin
import Foundation

package struct BenchmarkRequest: Decodable, Sendable {
  package let schema_version: Int
  package let case_key: String
  package let operation: String
  package let dataset_kind: String
  package let input_path: String
  package let input_sha256: String
  package let input_bytes: Int
  package let expected_path: String
  package let expected_sha256: String
  package let rows: Int
  package let warmups: Int
  package let samples: Int
  package let atol: Double
  package let rtol: Double
}

package struct BenchmarkSample: Codable, Sendable {
  package let elapsed_ns: Int64
  package let output_sha256: String
  package let maximum_absolute_error: Double
  package let validated: Bool
  package init(elapsed: Int64, values: [Double], expected: [Double], atol: Double, rtol: Double)
    throws
  {
    guard values.count == expected.count else { throw BenchmarkFailure("Output length mismatch") }
    var maximum = 0.0
    for (a, e) in zip(values, expected) {
      guard a.isFinite && e.isFinite && abs(a - e) <= atol + rtol * abs(e) else {
        throw BenchmarkFailure("Output mismatch: \(a) versus \(e)")
      }
      maximum = max(maximum, abs(a - e))
    }
    guard elapsed > 0 else { throw BenchmarkFailure("Timer could not resolve operation") }
    elapsed_ns = elapsed
    maximum_absolute_error = maximum
    validated = true
    output_sha256 = values.withUnsafeBytes {
      SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined()
    }
  }
}

package struct BenchmarkResponse: Encodable, Sendable {
  package let schema_version = 1
  package let case_key: String
  package let status: String
  package let samples: [BenchmarkSample]
  package let peak_rss_bytes: Int64
  package let engine_version: String
  package let error: String?
  package init(key: String, samples: [BenchmarkSample], error: String? = nil) {
    case_key = key
    self.samples = samples
    self.error = error
    status = error == nil ? "passed" : "failed"
    var usage = rusage()
    getrusage(RUSAGE_SELF, &usage)
    peak_rss_bytes = Int64(usage.ru_maxrss)
    engine_version = "SwiftSci standardized worker protocol 1"
  }
}

package struct BenchmarkFailure: Error, CustomStringConvertible {
  package let description: String
  package init(_ description: String) { self.description = description }
}

package func verifiedData(path: String, sha256: String, bytes: Int? = nil) throws -> Data {
  let data = try Data(contentsOf: URL(fileURLWithPath: path))
  let actual = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  guard actual == sha256, bytes == nil || bytes == data.count else {
    throw BenchmarkFailure("Fixture identity mismatch: \(path)")
  }
  return data
}

package func decodeDoubles(_ data: Data) throws -> [Double] {
  guard data.count % 8 == 0 else { throw BenchmarkFailure("Truncated float64 fixture") }
  return data.withUnsafeBytes { bytes in
    stride(from: 0, to: data.count, by: 8).map {
      Double(
        bitPattern: UInt64(littleEndian: bytes.loadUnaligned(fromByteOffset: $0, as: UInt64.self)))
    }
  }
}
