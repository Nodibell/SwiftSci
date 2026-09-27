import CoreML
import CryptoKit
import Darwin
import Foundation
import SwiftDataFrame
import SwiftML
import SwiftPreprocessing
import SwiftSciBenchmarkSupport

struct PublicWorkflowInput: Decodable {
  let operation: String
  let observations_csv: String?
  let calibration_csv: String?
  let feature_order: [String]?
  let route: String?
  let feature_names: [String]?
  let train_ids: [Int64]?
  let train_features: [[Double]]?
  let train_targets: [Double]?
  let heldout_ids: [Int64]?
  let heldout_features: [[Double]]?
  let heldout_targets: [Double]?

  static func decode(_ data: Data, operation: String, rows: Int) throws -> Self {
    let scientific = operation == "scientific-workflow"
    let keys: Set<String> = scientific
      ? ["operation", "observations_csv", "calibration_csv", "feature_order"]
      : ["operation", "route", "feature_names", "train_ids", "train_features", "train_targets", "heldout_ids", "heldout_features", "heldout_targets"]
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any], Set(object.keys) == keys else {
      throw BenchmarkFailure("Unexpected public workflow fields")
    }
    let p = try JSONDecoder().decode(Self.self, from: data)
    guard p.operation == operation, (4...128).contains(rows) else { throw BenchmarkFailure("Invalid workflow identity") }
    if scientific {
      guard let left = p.observations_csv, let right = p.calibration_csv, let order = p.feature_order,
        Set(order) == ["signal", "offset"], order.count == 2 else { throw BenchmarkFailure("Invalid scientific schema") }
      try validateCSV(left, columns: ["row_id", "site", "signal", "target", "quality"], rows: rows)
      try validateCSV(right, columns: ["calibration_id", "site", "offset"], rows: nil)
    } else {
      guard let route = p.route, ["fit", "native", "coreml"].contains(route),
        let names = p.feature_names, names.count == 2, Set(names) == ["signal", "offset"],
        let train = p.train_features, let query = p.heldout_features,
        let y = p.train_targets, let qy = p.heldout_targets,
        let ids = p.train_ids, let qids = p.heldout_ids,
        train.count == rows, y.count == rows, ids.count == rows,
        (1...128).contains(query.count), qy.count == query.count, qids.count == query.count,
        (train + query).allSatisfy({ $0.count == 2 && $0.allSatisfy({ $0.isFinite && abs($0) <= 10000 }) }),
        (y + qy).allSatisfy({ $0.isFinite && abs($0) <= 10000 }),
        Set(ids + qids).count == ids.count + qids.count,
        (ids + qids).allSatisfy({ $0 >= 0 && $0 <= Int64(Int32.max) }),
        !train.contains(where: { query.contains($0) })
      else { throw BenchmarkFailure("Invalid regression partitions or values") }
    }
    return p
  }

  static func validateCSV(_ text: String, columns: [String], rows: Int?) throws {
    let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
    guard let header = lines.first, text.utf8.count < 100000, lines.last == "", String(header) == columns.joined(separator: ","),
      (4...128).contains(lines.count - 2), rows == nil || rows == lines.count - 2 else {
      throw BenchmarkFailure("Invalid bounded workflow CSV schema")
    }
    var ids = Set<String>()
    for line in lines.dropFirst().dropLast() {
      let values = line.split(separator: ",", omittingEmptySubsequences: false)
      guard values.count == columns.count, ids.insert(String(values[0])).inserted else {
        throw BenchmarkFailure("Ragged CSV or duplicate row ID")
      }
      for (key, raw) in zip(columns, values) {
        if key == "quality" && raw.isEmpty { continue }
        guard let value = Double(raw), value.isFinite, abs(value) <= 10000 else { throw BenchmarkFailure("Invalid CSV number") }
        if ["row_id", "calibration_id", "site"].contains(key) {
          guard !raw.isEmpty, raw.utf8.allSatisfy({ (48...57).contains($0) }) else { throw BenchmarkFailure("Invalid CSV ID") }
        }
      }
    }
  }

  func execute(directory: URL) async throws -> [Double] {
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    let values = operation == "scientific-workflow"
      ? try await scientific(directory: directory) : try await regression(directory: directory)
    try JSONEncoder().encode(values).write(to: directory.appendingPathComponent("actual.json"))
    return values
  }

  private func scientific(directory: URL) async throws -> [Double] {
    guard let observations_csv, let calibration_csv, let feature_order else { throw BenchmarkFailure("Missing scientific inputs") }
    let leftURL = directory.appendingPathComponent("observations.csv")
    let rightURL = directory.appendingPathComponent("calibration.csv")
    try Data(observations_csv.utf8).write(to: leftURL)
    try Data(calibration_csv.utf8).write(to: rightURL)
    var leftOptions = CSVReadOptions()
    leftOptions.columnTypeOverrides = ["row_id": .int64, "site": .int64,
      "signal": .float64, "target": .float64, "quality": .float64]
    var rightOptions = CSVReadOptions()
    rightOptions.columnTypeOverrides = ["calibration_id": .int64, "site": .int64, "offset": .float64]
    let left = try await DataFrame(csv: leftURL, options: leftOptions)
    let right = try await DataFrame(csv: rightURL, options: rightOptions)
    let filtered = try left.filter(column: "quality", where: .greaterThanOrEqual(1))
    let joined = try filtered.join(right, on: "site", how: .inner)
      .sortBy("calibration_id", ascending: true).sortBy("row_id", ascending: true)
    let matrix = try joined.toFeatureMatrix(feature_order)
    let flat = try joined.toFlatFeatureMatrix(feature_order)
    let targets = try joined.toTargetVector("target")
    guard matrix.count >= 4, flat.rows == matrix.count, flat.cols == 2,
      flat.flat == matrix.flatMap({ $0 }) else { throw BenchmarkFailure("Scientific matrix layout differs") }
    let model = LinearRegression(device: .cpu)
    try await model.fit(features: matrix, targets: targets)
    guard await model.resolvedDevice == .cpu else { throw BenchmarkFailure("Scientific model did not use CPU") }
    let prediction = try await model.predict(features: matrix)
    let state = await model.getWeightsAndBias()
    guard let weights = state.weights, weights.count == 2, let bias = state.bias,
      prediction.count == targets.count else { throw BenchmarkFailure("Invalid scientific fitted state") }
    let residual = zip(prediction, targets).map { $0 - $1 }
    var output = [Double(left.shape.rows), Double(filtered.shape.rows), Double(joined.shape.rows), 2]
    output += try joined.toTargetVector("row_id")
    output += try joined.toTargetVector("calibration_id")
    output += flat.flat + targets + [bias] + weights + prediction + residual
    output += [residual.reduce(0) { $0 + $1 * $1 }]
    return output
  }

  private func regression(directory: URL) async throws -> [Double] {
    guard let route, let feature_names, let train_features, let train_targets, let train_ids,
      let heldout_features, let heldout_targets, let heldout_ids else { throw BenchmarkFailure("Missing regression inputs") }
    let model = LinearRegression(device: .cpu)
    let pipeline = RegressionPipeline(transformers: [StandardScaler()], estimator: model)
    try await pipeline.fit(features: train_features, targets: train_targets)
    guard await model.resolvedDevice == .cpu,
      let scaler = pipeline.transformers.first as? StandardScaler,
      let mean = scaler.mean, let std = scaler.std else { throw BenchmarkFailure("Missing fitted pipeline state") }
    let train = try scaler.transform(train_features)
    let query = try scaler.transform(heldout_features)
    let state = await model.getWeightsAndBias()
    guard let weights = state.weights, weights.count == 2, let bias = state.bias else { throw BenchmarkFailure("Missing regressor state") }
    let trainPrediction = try await pipeline.predict(features: train_features)
    let prediction = try await pipeline.predict(features: heldout_features)
    let residual = zip(prediction, heldout_targets).map { $0 - $1 }
    var output = [Double(train_features.count), Double(heldout_features.count), 2]
    output += train_ids.map(Double.init) + heldout_ids.map(Double.init) + mean + std
    output += train.flatMap({ $0 }) + query.flatMap({ $0 }) + [bias] + weights + trainPrediction + prediction + residual
    let after = await model.getWeightsAndBias()
    guard let finalScaler = pipeline.transformers.first as? StandardScaler,
      finalScaler.mean == mean, finalScaler.std == std, after.weights == weights, after.bias == bias else {
      throw BenchmarkFailure("Prediction mutated training state")
    }
    // Retain the pre-reload values even when the public export or loader fails.
    try JSONEncoder().encode(output).write(to: directory.appendingPathComponent("before-reload.json"))
    if route != "fit" {
      let modelURL = directory.appendingPathComponent(route == "native" ? "model.json" : "pipeline.mlmodel")
      if route == "native" {
        try await model.save(to: modelURL)
      } else {
        let scaledNames = feature_names.map { "scaled_" + $0 }
        let scaling = CoreMLExporter.exportBinaryStandardScaler(inputNames: feature_names, outputNames: scaledNames,
          shiftValues: mean.map { -$0 }, scaleValues: std.map { 1 / $0 })
        let fitted = try await model.exportCoreML(featureNames: scaledNames, outputName: "prediction")
        try CoreMLExporter.writePipelineRegressor(to: modelURL, inputNames: feature_names,
          outputName: "prediction", submodelsData: [scaling, fitted])
      }
      let request = WorkflowReloadRequest(route: route, model_path: modelURL.path,
        model_sha256: SHA256.hash(data: try Data(contentsOf: modelURL)).map { String(format: "%02x", $0) }.joined(), feature_names: feature_names,
        features: route == "native" ? query : heldout_features, parent_pid: Int(getpid()))
      let requestURL = directory.appendingPathComponent("reload-request.json")
      let responseURL = directory.appendingPathComponent("reload-response.json")
      try JSONEncoder().encode(request).write(to: requestURL)
      let process = Process()
      process.executableURL = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
      process.arguments = ["--workflow-reload", requestURL.path, responseURL.path]
      let logURL = directory.appendingPathComponent("reload.log")
      FileManager.default.createFile(atPath: logURL.path, contents: nil)
      let log = try FileHandle(forWritingTo: logURL)
      defer { try? log.close() }
      process.standardOutput = log; process.standardError = log
      try process.run()
      let deadline = Date().addingTimeInterval(60)
      while process.isRunning && Date() < deadline { try await Task.sleep(for: .milliseconds(25)) }
      if process.isRunning {
        kill(process.processIdentifier, SIGKILL)
        process.waitUntilExit()
        throw BenchmarkFailure("Fresh-process reload timed out")
      }
      process.waitUntilExit()
      let response = try JSONDecoder().decode(WorkflowReloadResponse.self, from: Data(contentsOf: responseURL))
      guard process.terminationStatus == 0, response.error == nil,
        response.pid == Int(process.processIdentifier), response.pid != Int(getpid()),
        response.model_sha256 == request.model_sha256, response.predictions.count == prediction.count else {
        throw BenchmarkFailure("Fresh-process \(route) reload failed: \(response.error ?? "invalid response")")
      }
      output += response.predictions
    }
    return output + [1]
  }
}

struct WorkflowReloadRequest: Codable {
  let route: String
  let model_path: String
  let model_sha256: String
  let feature_names: [String]
  let features: [[Double]]
  let parent_pid: Int
}

struct WorkflowReloadResponse: Codable {
  let pid: Int
  let model_sha256: String
  let predictions: [Double]
  let error: String?
}

extension Worker {
  static func reloadWorkflow(requestURL: URL, responseURL: URL) async -> Int32 {
    var checksum = "unknown"
    do {
      let request = try JSONDecoder().decode(WorkflowReloadRequest.self, from: Data(contentsOf: requestURL))
      checksum = request.model_sha256
      guard request.parent_pid != Int(getpid()), ["native", "coreml"].contains(request.route),
        request.feature_names.count == 2, Set(request.feature_names).count == 2,
        !request.features.isEmpty, request.features.count <= 128,
        request.features.allSatisfy({ $0.count == 2 && $0.allSatisfy(\.isFinite) }) else {
        throw BenchmarkFailure("Invalid fresh-process request")
      }
      _ = try verifiedData(path: request.model_path, sha256: checksum)
      let url = URL(fileURLWithPath: request.model_path)
      let values: [Double]
      if request.route == "native" {
        let model = try LinearRegression.load(from: url, device: .cpu)
        values = try await model.predict(features: request.features)
      } else {
        let compiled = try await MLModel.compileModel(at: url)
        defer { try? FileManager.default.removeItem(at: compiled) }
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
        let model = try MLModel(contentsOf: compiled, configuration: configuration)
        guard Set(model.modelDescription.inputDescriptionsByName.keys) == Set(request.feature_names),
          model.modelDescription.outputDescriptionsByName["prediction"] != nil else {
          throw BenchmarkFailure("Reloaded model schema differs")
        }
        values = try request.features.map { row in
          let provider = try MLDictionaryFeatureProvider(dictionary:
            Dictionary(uniqueKeysWithValues: zip(request.feature_names, row.map { NSNumber(value: $0) })))
          let result = try model.prediction(from: provider)
          guard let prediction = result.featureValue(for: "prediction"), prediction.type == .double else {
            throw BenchmarkFailure("Missing Core ML prediction")
          }
          return prediction.doubleValue
        }
      }
      guard values.count == request.features.count, values.allSatisfy(\.isFinite) else { throw BenchmarkFailure("Invalid reload predictions") }
      try JSONEncoder().encode(WorkflowReloadResponse(pid: Int(getpid()), model_sha256: checksum, predictions: values, error: nil)).write(to: responseURL)
      return 0
    } catch {
      try? JSONEncoder().encode(WorkflowReloadResponse(pid: Int(getpid()), model_sha256: checksum, predictions: [], error: String(describing: error))).write(to: responseURL)
      return 1
    }
  }
}
