import Darwin
import Foundation
import SwiftDataFrame
import SwiftPreprocessing
import SwiftSciBenchmarkSupport
import SwiftStats

@main struct Worker {
  enum Output {
    case frame(DataFrame)
    case values([Double])
  }
  static func main() async {
    guard CommandLine.arguments.count == 3 else {
      fputs("Expected request and response paths\n", stderr)
      exit(2)
    }
    let outputURL = URL(fileURLWithPath: CommandLine.arguments[2])
    var key = "unknown"
    do {
      let request = try JSONDecoder().decode(
        BenchmarkRequest.self,
        from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
      key = request.case_key
      guard request.schema_version == 1, request.rows > 0, request.warmups >= 0,
        request.samples > 0, request.atol >= 0, request.rtol >= 0
      else { throw BenchmarkFailure("Invalid request") }
      let input = try verifiedData(
        path: request.input_path, sha256: request.input_sha256, bytes: request.input_bytes)
      let expected = try decodeDoubles(
        verifiedData(path: request.expected_path, sha256: request.expected_sha256))
      let op = request.operation
      let frame: DataFrame?
      let x: [Double]
      if request.dataset_kind == "nist-numacc4" {
        frame = nil
        guard let text = String(data: input, encoding: .ascii) else {
          throw BenchmarkFailure("Invalid NIST encoding")
        }
        x = try text.components(separatedBy: "\n").dropFirst(60).filter {
          !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }.map {
          guard let value = Double($0.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            throw BenchmarkFailure("Invalid NIST value")
          }
          return value
        }
        guard x.count == request.rows else { throw BenchmarkFailure("NIST row count mismatch") }
      } else if op == "csv-read" {
        frame = nil
        x = []
      } else {
        let loaded = try await DataFrame(csv: URL(fileURLWithPath: request.input_path))
        guard loaded.shape.rows == request.rows else {
          throw BenchmarkFailure("Row count mismatch")
        }
        frame = loaded
        x = try loaded.toTargetVector("x")
      }
      var samples = [BenchmarkSample]()
      for index in 0..<(request.warmups + request.samples) {
        let start = ContinuousClock.now
        let output = try await execute(op, frame: frame, x: x, path: request.input_path)
        let duration = start.duration(to: .now).components
        let elapsed = duration.seconds * 1_000_000_000 + duration.attoseconds / 1_000_000_000
        let actual: [Double]
        switch output {
        case .values(let values): actual = values
        case .frame(let result):
          if op == "group-sum" {
            actual = try canonicalGroups(result)
          } else {
            actual = try result.toFlatFeatureMatrix(["id", "group", "x", "y"]).flat
          }
        }
        let sample = try BenchmarkSample(
          elapsed: elapsed, values: actual, expected: expected, atol: request.atol,
          rtol: request.rtol)
        if index >= request.warmups { samples.append(sample) }
      }
      try JSONEncoder().encode(BenchmarkResponse(key: key, samples: samples)).write(
        to: outputURL, options: .atomic)
    } catch {
      try? JSONEncoder().encode(
        BenchmarkResponse(key: key, samples: [], error: String(describing: error))
      ).write(to: outputURL, options: .atomic)
      fputs("\(error)\n", stderr)
      exit(1)
    }
  }
  static func canonicalGroups(_ result: DataFrame) throws -> [Double] {
    guard let keys = result[column: "group", as: String.self] else {
      throw BenchmarkFailure("Missing string group keys")
    }
    let sums = try result.toTargetVector("x_sum")
    let pairs = try (0..<result.shape.rows).map { row -> (Double, Double) in
      guard let text = keys.value(at: row) as? String, let key = Double(text) else {
        throw BenchmarkFailure("Invalid numeric group key")
      }
      return (key, sums[row])
    }
    return pairs.sorted { $0.0 < $1.0 }.flatMap { [$0.0, $0.1] }
  }
  @inline(never) static func execute(_ op: String, frame: DataFrame?, x: [Double], path: String)
    async throws -> Output
  {
    switch op {
    case "csv-read": return .frame(try await DataFrame(csv: URL(fileURLWithPath: path)))
    case "mean": return .values([try Stats.mean(x)])
    case "variance": return .values([try Stats.variance(x, ddof: 1)])
    case "stddev": return .values([try Stats.standardDeviation(x, ddof: 1)])
    default: break
    }
    guard let frame else { throw BenchmarkFailure("Operation requires a table") }
    switch op {
    case "filter": return .frame(try frame.filter(column: "x", where: .greaterThan(0)))
    case "sort": return .frame(try frame.sortBy("x"))
    case "group-sum": return .frame(frame.groupBy("group").agg(["x": .sum]))
    case "flat-matrix": return .values(try frame.toFlatFeatureMatrix(["x", "y"]).flat)
    case "target": return .values(try frame.toTargetVector("x"))
    case "standard-scale":
      return .values(
        try frame.standardScale(columns: ["x", "y"]).scaled.toFlatFeatureMatrix(["x", "y"]).flat)
    case "minmax-scale":
      return .values(
        try frame.minMaxScale(columns: ["x", "y"]).scaled.toFlatFeatureMatrix(["x", "y"]).flat)
    default: throw BenchmarkFailure("Unsupported operation: \(op)")
    }
  }
}
