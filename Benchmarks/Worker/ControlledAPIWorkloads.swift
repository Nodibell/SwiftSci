import Foundation
import SwiftCluster
import SwiftForecast
import SwiftML
import SwiftSciBenchmarkSupport

struct ControlledInferenceInput: Decodable {
  let operation: String
  let weights: [Double]
  let bias: Double
  let features: [[Double]]

  static func decode(_ data: Data, operation: String, rows: Int) throws -> Self {
    try controlledFields(data, expected: ["operation", "weights", "bias", "features"])
    let input = try JSONDecoder().decode(Self.self, from: data)
    guard ["linear-fixed-cpu", "logistic-fixed-cpu"].contains(operation), input.operation == operation,
      rows > 0, !input.weights.isEmpty, input.weights.allSatisfy(\.isFinite), input.bias.isFinite,
      finiteMatrix(input.features, rows: rows, columns: input.weights.count)
    else { throw BenchmarkFailure("Invalid fixed-inference dimensions or values") }
    return input
  }
}

struct ControlledKMeansInput: Decodable {
  let operation: String
  let features: [[Double]]
  let query: [[Double]]
  let n_clusters: Int
  let max_iterations: Int
  let tolerance: Double
  let seed: Int

  static func decode(_ data: Data, rows: Int) throws -> Self {
    try controlledFields(data, expected: ["operation", "features", "query", "n_clusters", "max_iterations", "tolerance", "seed"])
    let input = try JSONDecoder().decode(Self.self, from: data)
    guard input.operation == "kmeans-one-cpu", rows > 0,
      let width = input.features.first?.count, width > 0,
      finiteMatrix(input.features, rows: rows, columns: width),
      !input.query.isEmpty, finiteMatrix(input.query, rows: input.query.count, columns: width),
      input.n_clusters == 1, input.max_iterations >= 2,
      input.tolerance.isFinite, input.tolerance > 0, input.seed >= 0
    else { throw BenchmarkFailure("Invalid one-cluster KMeans dimensions or controls") }
    return input
  }
}

struct ControlledKalmanInput: Decodable {
  let operation: String
  let state_size: Int
  let observation_size: Int
  let transition: [[Double]]
  let observation_matrix: [[Double]]
  let process_noise: [[Double]]
  let measurement_noise: [[Double]]
  let initial_mean: [Double]
  let initial_covariance: [[Double]]
  let observations: [[Double]]

  static func decode(_ data: Data, rows: Int) throws -> Self {
    try controlledFields(data, expected: ["operation", "state_size", "observation_size", "transition", "observation_matrix", "process_noise", "measurement_noise", "initial_mean", "initial_covariance", "observations"])
    let input = try JSONDecoder().decode(Self.self, from: data)
    let n = input.state_size
    let m = input.observation_size
    guard input.operation == "kalman-fixed-cpu", rows > 0, (1...2).contains(n), (1...2).contains(m),
      finiteMatrix(input.transition, rows: n, columns: n),
      finiteMatrix(input.observation_matrix, rows: m, columns: n),
      finiteMatrix(input.process_noise, rows: n, columns: n),
      finiteMatrix(input.measurement_noise, rows: m, columns: m),
      input.initial_mean.count == n, input.initial_mean.allSatisfy(\.isFinite),
      finiteMatrix(input.initial_covariance, rows: n, columns: n),
      finiteMatrix(input.observations, rows: rows, columns: m)
    else { throw BenchmarkFailure("Invalid controlled Kalman dimensions or values") }
    guard validCovariance(input.process_noise, size: n),
      validCovariance(input.initial_covariance, size: n),
      validCovariance(input.measurement_noise, size: m, strictlyPositive: true)
    else { throw BenchmarkFailure("Invalid controlled Kalman covariance") }
    return input
  }
}

private func validCovariance(_ value: [[Double]], size: Int, strictlyPositive: Bool = false) -> Bool {
  guard (0..<size).allSatisfy({ i in value[i][i] >= 0 && (0..<size).allSatisfy({ j in value[i][j] == value[j][i] }) }) else { return false }
  let determinant = size == 1 ? value[0][0] : value[0][0]*value[1][1]-value[0][1]*value[1][0]
  return strictlyPositive ? determinant > 0 : determinant >= 0
}

private func controlledFields(_ data: Data, expected: Set<String>) throws {
  guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
    Set(object.keys) == expected
  else { throw BenchmarkFailure("Unexpected controlled numerical input fields") }
}

private func finiteMatrix(_ matrix: [[Double]], rows: Int, columns: Int) -> Bool {
  matrix.count == rows && matrix.allSatisfy { $0.count == columns && $0.allSatisfy(\.isFinite) }
}

extension Worker {
  @inline(never) static func executeFixedLinear(_ input: ControlledInferenceInput) async throws -> Output {
    let model = LinearRegression(weights: input.weights, bias: input.bias, device: .cpu)
    let parameters = await model.getWeightsAndBias()
    let predictions = try await model.predict(features: input.features)
    guard await model.resolvedDevice == .cpu else {
      throw BenchmarkFailure("Fixed linear inference did not resolve to CPU")
    }
    guard let weights = parameters.weights, let bias = parameters.bias,
      weights.count == input.weights.count, predictions.count == input.features.count
    else { throw BenchmarkFailure("Missing or invalid fixed linear result dimensions") }
    return .values([bias] + weights + predictions)
  }

  @inline(never) static func executeFixedLogistic(_ input: ControlledInferenceInput) async throws -> Output {
    let model = LogisticRegression(weights: input.weights, bias: input.bias, device: .cpu)
    let parameters = await model.getWeightsAndBias()
    let probabilities = try await model.predictProbability(features: input.features)
    let labels = try await model.predict(features: input.features)
    guard await model.resolvedDevice == .cpu else {
      throw BenchmarkFailure("Fixed logistic inference did not resolve to CPU")
    }
    guard let weights = parameters.weights, let bias = parameters.bias,
      weights.count == input.weights.count, probabilities.count == input.features.count,
      probabilities.allSatisfy({ $0.count == 2 }), labels.count == input.features.count,
      labels.allSatisfy({ $0 == 0 || $0 == 1 })
    else { throw BenchmarkFailure("Missing or invalid fixed logistic result dimensions") }
    return .values([bias] + weights + probabilities.flatMap { $0 } + labels.map(Double.init))
  }

  @inline(never) static func executeOneCluster(_ input: ControlledKMeansInput) async throws -> Output {
    let model = try KMeans(nClusters: input.n_clusters, maxIterations: input.max_iterations,
                          tolerance: input.tolerance, seed: input.seed, device: .cpu)
    try await model.fit(features: input.features)
    guard await model.resolvedDevice == .cpu else {
      throw BenchmarkFailure("One-cluster KMeans did not resolve to CPU")
    }
    guard let centroids = await model.getCentroids(),
      finiteMatrix(centroids, rows: 1, columns: input.features[0].count)
    else { throw BenchmarkFailure("Missing or invalid KMeans centroid dimensions") }
    let labels = try await model.predict(features: input.features)
    let queryLabels = try await model.predict(features: input.query)
    guard labels.count == input.features.count, labels.allSatisfy({ $0 == 0 }),
      queryLabels.count == input.query.count, queryLabels.allSatisfy({ $0 == 0 })
    else { throw BenchmarkFailure("Invalid one-cluster labels") }
    let inertia = input.features.reduce(0.0) { total, row in
      total + zip(row, centroids[0]).reduce(0.0) { sum, pair in
        let delta = pair.0 - pair.1
        return sum + delta * delta
      }
    }
    return .values(centroids[0] + labels.map(Double.init) + queryLabels.map(Double.init) + [inertia])
  }

  @inline(never) static func executeControlledKalman(_ input: ControlledKalmanInput) async throws -> Output {
    let model = try KalmanFilter(stateSize: input.state_size, observationSize: input.observation_size)
    try await model.setTransitionMatrix(input.transition)
    try await model.setObservationMatrix(input.observation_matrix)
    try await model.setProcessNoise(input.process_noise)
    try await model.setMeasurementNoise(input.measurement_noise)
    try await model.setInitialState(mean: input.initial_mean, covariance: input.initial_covariance)
    let states = try await model.filter(observations: input.observations)
    let next = try await model.predict()
    guard states.count == input.observations.count else {
      throw BenchmarkFailure("Kalman output observation count mismatch")
    }
    var output = [Double]()
    for state in states + [next] {
      guard state.mean.count == input.state_size,
        finiteMatrix(state.covariance, rows: input.state_size, columns: input.state_size)
      else { throw BenchmarkFailure("Invalid Kalman output dimensions or covariance values") }
      output += state.mean
      output += state.covariance.flatMap { $0 }
    }
    return .values(output)
  }
}
