import Foundation
import SwiftML
import SwiftSciBenchmarkSupport
import SwiftStats

private struct OLSInput: Decodable {
  let operation: String
  let features: [[Double]]
  let targets: [Double]
}

private struct ANOVAInput: Decodable {
  let operation: String
  let groups: [[Double]]
}

enum NumericalInputs {
  case ols(features: [[Double]], targets: [Double])
  case anova(groups: [[Double]])
  case pca(PCAFixtureInput)
  case naiveBayes(NBFixtureInput)
  case fixedLinear(ControlledInferenceInput)
  case fixedLogistic(ControlledInferenceInput)
  case oneCluster(ControlledKMeansInput)
  case controlledKalman(ControlledKalmanInput)
  case vectorCosine(VectorCosineFixtureInput)
  case kernelSHAP(KernelSHAPFixtureInput)

  static func decode(_ data: Data, operation: String, datasetKind: String, rows: Int)
    throws -> NumericalInputs?
  {
    let isNumerical = ["ols-cpu", "nist-anova", "pca-cpu", "multinomial-nb-cpu", "linear-fixed-cpu", "logistic-fixed-cpu", "kmeans-one-cpu", "kalman-fixed-cpu", "vector-cosine", "kernel-shap"].contains(operation)
    guard isNumerical == (datasetKind == "numerical-fixture-v1") else {
      throw BenchmarkFailure("Numerical workload/dataset mismatch")
    }
    guard isNumerical else { return nil }
    switch operation {
    case "linear-fixed-cpu": return .fixedLinear(try ControlledInferenceInput.decode(data, operation: operation, rows: rows))
    case "logistic-fixed-cpu": return .fixedLogistic(try ControlledInferenceInput.decode(data, operation: operation, rows: rows))
    case "kmeans-one-cpu": return .oneCluster(try ControlledKMeansInput.decode(data, rows: rows))
    case "kalman-fixed-cpu": return .controlledKalman(try ControlledKalmanInput.decode(data, rows: rows))
    case "vector-cosine": return .vectorCosine(try VectorCosineFixtureInput.decode(data, rows: rows))
    case "kernel-shap": return .kernelSHAP(try KernelSHAPFixtureInput.decode(data, rows: rows))
    default: break
    }
    if operation == "pca-cpu" {
      return .pca(try PCAFixtureInput.decode(data, rows: rows))
    }
    if operation == "multinomial-nb-cpu" {
      return .naiveBayes(try NBFixtureInput.decode(data, rows: rows))
    }
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw BenchmarkFailure("Numerical input must be an object")
    }
    let fields: Set<String> = operation == "ols-cpu"
      ? ["operation", "features", "targets"] : ["operation", "groups"]
    guard Set(object.keys) == fields else {
      throw BenchmarkFailure("Unexpected numerical input fields")
    }
    if operation == "ols-cpu" {
      let input = try JSONDecoder().decode(OLSInput.self, from: data)
      guard input.operation == operation else {
        throw BenchmarkFailure("Numerical input operation mismatch")
      }
      guard rows > 0, input.features.count == rows, input.targets.count == rows,
        let columns = input.features.first?.count, columns > 0,
        input.features.allSatisfy({ $0.count == columns && $0.allSatisfy(\.isFinite) }),
        input.targets.allSatisfy(\.isFinite)
      else { throw BenchmarkFailure("Invalid OLS dimensions or values") }
      guard rows > columns else { throw BenchmarkFailure("Underdetermined OLS fixture") }
      return .ols(features: input.features, targets: input.targets)
    }
    let input = try JSONDecoder().decode(ANOVAInput.self, from: data)
    guard input.operation == operation else {
      throw BenchmarkFailure("Numerical input operation mismatch")
    }
    guard input.groups.count >= 2,
      input.groups.allSatisfy({ !$0.isEmpty && $0.allSatisfy(\.isFinite) }),
      input.groups.reduce(0, { $0 + $1.count }) == rows, rows > input.groups.count
    else { throw BenchmarkFailure("Invalid ANOVA dimensions or values") }
    return .anova(groups: input.groups)
  }
}

extension Worker {
  @inline(never) static func executeNumerical(_ input: NumericalInputs) async throws -> Output {
    switch input {
    case .fixedLinear(let input): return try await executeFixedLinear(input)
    case .fixedLogistic(let input): return try await executeFixedLogistic(input)
    case .oneCluster(let input): return try await executeOneCluster(input)
    case .controlledKalman(let input): return try await executeControlledKalman(input)
    case .vectorCosine(let input): return try executeVectorCosine(input)
    case .kernelSHAP(let input): return await executeKernelSHAP(input)
    case .pca(let input): return try await executePCA(input)
    case .naiveBayes(let input): return try await executeNaiveBayes(input)
    case .ols(let features, let targets):
      let model = LinearRegression(device: .cpu)
      try await model.fit(features: features, targets: targets)
      guard await model.resolvedDevice == .cpu else {
        throw BenchmarkFailure("OLS did not resolve to CPU")
      }
      let predictions = try await model.predict(features: features)
      let parameters = await model.getWeightsAndBias()
      guard let weights = parameters.weights, let bias = parameters.bias,
        weights.count == features[0].count, predictions.count == targets.count
      else { throw BenchmarkFailure("Missing or invalid OLS result dimensions") }
      let rss = zip(targets, predictions).reduce(0.0) { sum, pair in
        let residual = pair.0 - pair.1
        return sum + residual * residual
      }
      return .values([bias] + weights + [rss] + predictions)
    case .anova(let groups):
      let result = try Stats.oneWayANOVA(groups: groups)
      return .values([result.fStatistic, Double(result.dfBetween), Double(result.dfWithin)])
    }
  }
}
