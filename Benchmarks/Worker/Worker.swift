import Darwin
import Foundation
import SwiftDataFrame
import SwiftPreprocessing
import SwiftSciBenchmarkSupport
import SwiftStats

@main struct Worker {
  enum Output {
    case frame(DataFrame)
    case chunks([DataFrame], inputRows: [Int])
    case written(URL)
    case values([Double])
    case cosineResults([(Int, Double)])
    case text(String)
  }
  static func main() async {
    if CommandLine.arguments.count == 4 && CommandLine.arguments[1] == "--bounded-source-lifetime" {
      exit(await boundedSourceLifetime(request: URL(fileURLWithPath: CommandLine.arguments[2]),
                                       output: URL(fileURLWithPath: CommandLine.arguments[3])))
    }
    if CommandLine.arguments.count == 4 && CommandLine.arguments[1] == "--bounded-coreml-shutdown" {
      exit(await boundedCoreMLShutdown(request: URL(fileURLWithPath: CommandLine.arguments[2]),
                                      output: URL(fileURLWithPath: CommandLine.arguments[3])))
    }
    if CommandLine.arguments.count == 4 && CommandLine.arguments[1] == "--bounded-coreml-workflow" {
      exit(await boundedCoreMLWorkflow(request: URL(fileURLWithPath: CommandLine.arguments[2]),
                                      output: URL(fileURLWithPath: CommandLine.arguments[3])))
    }
    if CommandLine.arguments.count == 4 && CommandLine.arguments[1] == "--fused-coreml-workflow" {
      exit(await fusedCoreMLWorkflow(request: URL(fileURLWithPath: CommandLine.arguments[2]),
                                    output: URL(fileURLWithPath: CommandLine.arguments[3])))
    }
    if CommandLine.arguments.count == 3 && CommandLine.arguments[1] == "--fused-preprocessing-trial" {
      exit(benchmarkFusedPreprocessing(output: URL(fileURLWithPath: CommandLine.arguments[2])))
    }
    if CommandLine.arguments.count == 4 && CommandLine.arguments[1] == "--llama-inspect" {
      exit(inspectLlama(requestURL: URL(fileURLWithPath: CommandLine.arguments[2]),
                        responseURL: URL(fileURLWithPath: CommandLine.arguments[3])))
    }
    if CommandLine.arguments.count == 4 && CommandLine.arguments[1] == "--coreml-prepared-workflow" {
      let status = await preparedCoreMLWorkflow(request: URL(fileURLWithPath: CommandLine.arguments[2]),
        output: URL(fileURLWithPath: CommandLine.arguments[3]))
      exit(status)
    }
    if CommandLine.arguments.count == 4 && CommandLine.arguments[1] == "--coreml-concurrency" {
      let status = await stressCoreML(request: URL(fileURLWithPath: CommandLine.arguments[2]),
        output: URL(fileURLWithPath: CommandLine.arguments[3]))
      exit(status)
    }
    if CommandLine.arguments.count == 4 && CommandLine.arguments[1] == "--coreml-qualification" {
      let status = await qualifyCoreML(mode: CommandLine.arguments[2],
        output: URL(fileURLWithPath: CommandLine.arguments[3]))
      exit(status)
    }
    if CommandLine.arguments.count == 4 && CommandLine.arguments[1] == "--workflow-reload" {
      let status = await reloadWorkflow(requestURL: URL(fileURLWithPath: CommandLine.arguments[2]),
        responseURL: URL(fileURLWithPath: CommandLine.arguments[3]))
      exit(status)
    }
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
      let numericalInputs = try NumericalInputs.decode(
        input, operation: op, datasetKind: request.dataset_kind, rows: request.rows)
      let frame: DataFrame?
      let x: [Double]
      if numericalInputs != nil {
        frame = nil
        x = []
      } else if request.dataset_kind == "nist-univariate-v1" {
        frame = nil
        guard let skipRows = request.input_skip_rows else {
          throw BenchmarkFailure("Missing NIST data offset")
        }
        x = try decodeUnivariate(input, skipRows: skipRows, rows: request.rows)
      } else if op == "wine-pipeline" || op == "csv-read" || op == "parquet-read" || op.hasPrefix("csv-stream-") {
        frame = nil
        x = []
      } else {
        let loaded = try await DataFrame(csv: URL(fileURLWithPath: request.input_path))
        guard loaded.shape.rows == request.rows else {
          throw BenchmarkFailure("Row count mismatch")
        }
        frame = loaded
        x = op.hasPrefix("h2o-") ? [] : try loaded.toTargetVector("x")
      }
      let y = op.hasPrefix("h2o-") ? [] : try frame?.toTargetVector("y") ?? []
      let right: DataFrame? = op == "inner-join" ? try DataFrame(columns: [
        TypedColumn<Int64>(name: "id", values: (0..<request.rows).map(Int64.init)),
        TypedColumn<Double>(name: "weight", values: (0..<request.rows).map { Double($0) / 4 }),
      ]) : nil
      let coreInputs = CoreInputs(operation: op, rows: request.rows, x: x, y: y)
      var samples = [BenchmarkSample]()
      for index in 0..<(request.warmups + request.samples) {
        let start = ContinuousClock.now
        let output: Output
        if let numericalInputs {
          output = try await executeNumerical(numericalInputs, artifactDirectory: outputURL.appendingPathExtension("sample\(index)"))
        } else if op == "wine-pipeline" {
          output = try await winePipeline(path: request.input_path, rows: request.rows)
        } else if op == "csv-read" {
          output = .frame(try await DataFrame(csv: URL(fileURLWithPath: request.input_path)))
        } else if op == "sqlite-ingest" {
          output = .frame(try await sqliteIngest())
        } else if op == "parquet-read" {
          output = .frame(try await ParquetReader.read(url: URL(fileURLWithPath: request.input_path)))
        } else if op == "parquet-write" {
          guard let frame else { throw BenchmarkFailure("Missing write input") }
          let path = outputURL.appendingPathExtension("sample\(index).parquet")
          try await ParquetWriter.write(dataFrame: frame, to: path)
          output = .written(path)
        } else if op.hasPrefix("csv-stream-") {
          var chunks = [DataFrame]()
          var inputRows = [Int]()
          for try await chunk in DataFrame.readCSVStream(
            contentsOf: URL(fileURLWithPath: request.input_path), chunkSize: 10000)
          {
            inputRows.append(chunk.shape.rows)
            switch op {
            case "csv-stream-read": chunks.append(chunk)
            case "csv-stream-filter":
              chunks.append(try chunk.filter(column: "x", where: .greaterThan(0)))
            case "csv-stream-group":
              chunks.append(chunk.groupBy("group").agg(["x": .sum, "y": .mean]))
            default: throw BenchmarkFailure("Unsupported streaming operation")
            }
          }
          output = .chunks(chunks, inputRows: inputRows)
        } else {
          output = try executeCore(op, x: x, y: y, inputs: coreInputs)
            ?? execute(op, frame: frame, x: x, y: y, right: right)
        }
        let duration = start.duration(to: .now).components
        let elapsed = duration.seconds * 1_000_000_000 + duration.attoseconds / 1_000_000_000
        let actual: [Double]
        switch output {
        case .values(let values): actual = values
        case .cosineResults(let matches):
          guard case .vectorCosine(let input)? = numericalInputs else {
            throw BenchmarkFailure("Missing cosine inputs")
          }
          actual = try canonicalCosineFixtureValues(matches, entryCount: input.entryCount, expectedCount: input.topK)
        case .text(let text): actual = text.utf8.map(Double.init)
        case .frame(let result):
          actual = try canonical(result, operation: op, mixed: request.dataset_kind != "table-v1")
        case .written(let path):
          actual = try canonical(try await ParquetReader.read(url: path), operation: op, mixed: true)
        case .chunks(let chunks, let inputRows):
          let expectedSizes = stride(from: 0, to: request.rows, by: 10000).map { min(10000, request.rows - $0) }
          guard inputRows == expectedSizes else { throw BenchmarkFailure("Streaming chunk boundaries differ") }
          actual = try chunks.enumerated().flatMap { index, result -> [Double] in
            if op == "csv-stream-group" {
              let rows = try canonicalGroups(result, includeMean: true, mixed: request.dataset_kind != "table-v1")
              return stride(from: 0, to: rows.count, by: 3).flatMap {
                [Double(index)] + Array(rows[$0..<($0 + 3)])
              }
            }
            return try canonical(result, operation: op, mixed: request.dataset_kind != "table-v1")
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
  static func canonical(_ result: DataFrame, operation: String, mixed: Bool) throws -> [Double] {
    if operation.hasPrefix("h2o-") { return try canonicalH2O(result, operation: operation) }
    if operation == "sqlite-ingest" {
      guard result.columnNames == ["id", "val"] else { throw BenchmarkFailure("SQL schema mismatch") }
      return try result.toFlatFeatureMatrix(["id", "val"]).flat
    }
    if operation == "group-sum" || operation == "group-sum-mean" {
      return try canonicalGroups(result, includeMean: operation == "group-sum-mean", mixed: mixed)
    }
    let frame = operation == "inner-join" ? try result.sortBy("id") : result
    if mixed {
      let ids = try frame.toTargetVector("id")
      let xs = try frame.toTargetVector("x"), ys = try frame.toTargetVector("y")
      guard let groups = frame[column: "group", as: String.self],
        let flags = frame[column: "flag", as: Bool.self] else {
        throw BenchmarkFailure("Mixed table lost category or Boolean column")
      }
      let weights = operation == "inner-join" ? try frame.toTargetVector("weight") : []
      return try (0..<frame.shape.rows).flatMap { row -> [Double] in
        guard let label = groups.value(at: row) as? String,
          label.hasPrefix("category"), let group = Double(label.dropFirst(8)),
          let flag = flags.value(at: row) as? Bool else {
          throw BenchmarkFailure("Malformed mixed output")
        }
        return [ids[row], group, xs[row], ys[row], flag ? 1 : 0]
          + (operation == "inner-join" ? [weights[row]] : [])
      }
    }
    let columns = ["id", "group", "x", "y"] + (operation == "inner-join" ? ["weight"] : [])
    return try frame.toFlatFeatureMatrix(columns).flat
  }
  static func canonicalGroups(_ result: DataFrame, includeMean: Bool = false, mixed: Bool = false) throws -> [Double] {
    guard let keys = result[column: "group", as: String.self] else {
      throw BenchmarkFailure("Missing string group keys")
    }
    let sums = try result.toTargetVector("x_sum")
    let means = includeMean ? try result.toTargetVector("y_mean") : []
    let rows = try (0..<result.shape.rows).map { row -> [Double] in
      guard let text = keys.value(at: row) as? String,
        !mixed || text.hasPrefix("category"),
        let key = Double(mixed ? String(text.dropFirst(8)) : text) else {
        throw BenchmarkFailure("Invalid numeric group key")
      }
      return [key, sums[row]] + (includeMean ? [means[row]] : [])
    }
    return rows.sorted { $0[0] < $1[0] }.flatMap { $0 }
  }
  @inline(never) static func execute(_ op: String, frame: DataFrame?, x: [Double], y: [Double], right: DataFrame?)
    throws -> Output
  {
    switch op {
    case "pearson": return .values([try Stats.pearsonCorrelation(x, y)])
    case "spearman": return .values([try Stats.spearmanCorrelation(x, y)])
    case "mean": return .values([try Stats.mean(x)])
    case "variance": return .values([try Stats.variance(x, ddof: 1)])
    case "stddev": return .values([try Stats.standardDeviation(x, ddof: 1)])
    default: break
    }
    guard let frame else { throw BenchmarkFailure("Operation requires a table") }
    if op.hasPrefix("h2o-") { return try h2o(op, frame: frame) }
    switch op {
    case "row-sum":
      var sum = 0.0
      for row in frame.rows {
        guard let value = row.double("x") else { throw BenchmarkFailure("Missing row value") }
        sum += value
      }
      return .values([sum])
    case "inner-join":
      guard let right else { throw BenchmarkFailure("Missing right join table") }
      return .frame(try frame.joinSIMD(right, on: "id", how: .inner))
    case "group-sum-mean": return .frame(frame.groupBy("group").agg(["x": .sum, "y": .mean]))
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
