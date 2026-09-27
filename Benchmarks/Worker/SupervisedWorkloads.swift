import Foundation
import SwiftML
import SwiftOptimize
import SwiftPreprocessing
import SwiftSciBenchmarkSupport

struct SupervisedFixtureInput: Decodable {
  struct Splits: Decodable {
    let train: [Int]
    let validation: [Int]
    let test: [Int]
  }

  let operation: String
  let row_ids: [String]
  let feature_names: [String]
  let features: [[Double]]
  let targets: [Double]
  let splits: Splits

  static func decode(_ data: Data, operation: String, rows: Int) throws -> Self {
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      Set(object.keys) == Set(["operation", "row_ids", "feature_names", "features", "targets", "splits"]),
      let splitObject = object["splits"] as? [String: Any],
      Set(splitObject.keys) == Set(["train", "validation", "test"])
    else { throw BenchmarkFailure("Unexpected supervised input fields") }
    let input = try JSONDecoder().decode(Self.self, from: data)
    let columns = input.feature_names.count
    guard ["supervised-scale", "supervised-ols-cpu"].contains(operation),
      input.operation == operation, rows > 0, columns > 0,
      input.row_ids.count == rows, Set(input.row_ids).count == rows,
      input.row_ids.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }),
      Set(input.feature_names).count == columns,
      input.feature_names.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }),
      input.features.count == rows,
      input.features.allSatisfy({ $0.count == columns && $0.allSatisfy(\.isFinite) }),
      input.targets.count == rows, input.targets.allSatisfy(\.isFinite)
    else { throw BenchmarkFailure("Invalid supervised input dimensions or values") }
    let partitions = [input.splits.train, input.splits.validation, input.splits.test]
    let indices = partitions.flatMap { $0 }
    guard partitions.allSatisfy({ !$0.isEmpty }), indices.count == rows,
      indices.allSatisfy({ $0 >= 0 && $0 < rows }), Set(indices).count == rows
    else { throw BenchmarkFailure("Supervised splits must be nonempty, disjoint, and exhaustive") }
    guard input.splits.train.count > columns else {
      throw BenchmarkFailure("Underdetermined supervised training split")
    }
    var featurePartitions = [[Double]: Int]()
    for (partition, indices) in partitions.enumerated() {
      for index in indices {
        let row = input.features[index]
        if let previous = featurePartitions[row], previous != partition {
          throw BenchmarkFailure("Identical feature rows cross supervised split boundaries")
        }
        featurePartitions[row] = partition
      }
    }
    if operation == "supervised-ols-cpu" {
      for indices in [input.splits.validation, input.splits.test] {
        let first = input.targets[indices[0]]
        guard indices.contains(where: { input.targets[$0] != first }) else {
          throw BenchmarkFailure("Supervised held-out targets must vary for R2")
        }
      }
    }
    return input
  }
}

extension Worker {
  @inline(never) static func executeSupervised(_ input: SupervisedFixtureInput) async throws -> Output {
    let columns = input.feature_names.count
    let partitions = [input.splits.train, input.splits.validation, input.splits.test]
    let raw = partitions.map { indices in indices.map { input.features[$0] } }
    let targets = partitions.map { indices in indices.map { input.targets[$0] } }
    var scaler = StandardScaler()
    try scaler.fit(raw[0])
    let scaled = try raw.map { try scaler.transform($0) }
    guard let means = scaler.mean, let scales = scaler.std,
      means.count == columns, means.allSatisfy(\.isFinite),
      scales.count == columns, scales.allSatisfy({ $0.isFinite && $0 > 0 }),
      zip(scaled, partitions).allSatisfy({ matrix, indices in
        matrix.count == indices.count
          && matrix.allSatisfy({ $0.count == columns && $0.allSatisfy(\.isFinite) })
      })
    else { throw BenchmarkFailure("Invalid supervised scaler output") }

    if input.operation == "supervised-scale" {
      let values = means + scales + scaled.flatMap { $0.flatMap { $0 } }
      guard values.count == 2 * columns + input.features.count * columns,
        values.allSatisfy(\.isFinite)
      else { throw BenchmarkFailure("Invalid supervised scale output layout") }
      return .values(values)
    }
    guard input.operation == "supervised-ols-cpu" else {
      throw BenchmarkFailure("Unsupported supervised operation")
    }

    let model = LinearRegression(device: .cpu)
    try await model.fit(features: scaled[0], targets: targets[0])
    guard await model.resolvedDevice == .cpu else {
      throw BenchmarkFailure("Supervised OLS did not resolve to CPU")
    }
    var predictions = [[Double]]()
    for matrix in scaled {
      predictions.append(try await model.predict(features: matrix))
    }
    let parameters = await model.getWeightsAndBias()
    guard let weights = parameters.weights, let bias = parameters.bias,
      weights.count == columns, weights.allSatisfy(\.isFinite), bias.isFinite,
      zip(predictions, targets).allSatisfy({ predicted, actual in
        predicted.count == actual.count && predicted.allSatisfy(\.isFinite)
      })
    else { throw BenchmarkFailure("Invalid supervised OLS output") }

    let trainingMean = targets[0].reduce(0, +) / Double(targets[0].count)
    guard trainingMean.isFinite else {
      throw BenchmarkFailure("Nonfinite supervised training target mean")
    }
    var scores = [Double]()
    for split in 1...2 {
      let actual = targets[split]
      let predicted = predictions[split]
      let baseline = [Double](repeating: trainingMean, count: actual.count)
      scores.append(contentsOf: [
        Metrics.rootMeanSquaredError(yTrue: actual, yPred: predicted),
        Metrics.meanAbsoluteError(yTrue: actual, yPred: predicted),
        Metrics.r2Score(yTrue: actual, yPred: predicted),
        Metrics.rootMeanSquaredError(yTrue: actual, yPred: baseline),
        Metrics.meanAbsoluteError(yTrue: actual, yPred: baseline),
        Metrics.r2Score(yTrue: actual, yPred: baseline),
      ])
    }
    let values = means + scales + [bias] + weights
      + predictions.flatMap { $0 } + [trainingMean] + scores
    guard values.count == 3 * columns + input.features.count + 14,
      values.allSatisfy(\.isFinite)
    else { throw BenchmarkFailure("Invalid supervised OLS output layout") }
    return .values(values)
  }
}
