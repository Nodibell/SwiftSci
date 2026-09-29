import Foundation
import SwiftCluster
import SwiftNLP
import SwiftSciBenchmarkSupport

struct PCAFixtureInput: Decodable {
  let operation: String
  let features: [[Double]]
  let query: [[Double]]
  let n_components: Int

  static func decode(_ data: Data, rows: Int) throws -> Self {
    try apiInputFields(data, expected: ["operation", "features", "query", "n_components"])
    let input = try JSONDecoder().decode(Self.self, from: data)
    guard input.operation == "pca-cpu", rows >= 2, input.features.count == rows,
      let width = input.features.first?.count, width > 0,
      input.features.allSatisfy({ $0.count == width && $0.allSatisfy(\.isFinite) }),
      !input.query.isEmpty,
      input.query.allSatisfy({ $0.count == width && $0.allSatisfy(\.isFinite) }),
      input.n_components > 0, input.n_components <= min(rows, width)
    else { throw BenchmarkFailure("Invalid PCA input dimensions or values") }
    return input
  }
}

struct NBFixtureInput: Decodable {
  let operation: String
  let features: [[Double]]
  let targets: [Double]
  let query: [[Double]]
  let alpha: Double

  static func decode(_ data: Data, rows: Int) throws -> Self {
    try apiInputFields(data, expected: ["operation", "features", "targets", "query", "alpha"])
    let input = try JSONDecoder().decode(Self.self, from: data)
    guard input.operation == "multinomial-nb-cpu", rows > 1, input.features.count == rows,
      let width = input.features.first?.count, width > 0,
      input.features.allSatisfy({ $0.count == width && $0.allSatisfy({ $0.isFinite && $0 >= 0 }) }),
      input.targets.count == rows, input.targets.allSatisfy(\.isFinite), Set(input.targets).count >= 2,
      !input.query.isEmpty,
      input.query.allSatisfy({ $0.count == width && $0.allSatisfy({ $0.isFinite && $0 >= 0 }) }),
      input.alpha.isFinite, input.alpha > 0
    else { throw BenchmarkFailure("Invalid multinomial Naive Bayes input dimensions or values") }
    return input
  }
}

private func apiInputFields(_ data: Data, expected: Set<String>) throws {
  guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
    Set(object.keys) == expected
  else { throw BenchmarkFailure("Unexpected API numerical input fields") }
}

private func scoreGram(_ left: [[Double]], _ right: [[Double]]) -> [Double] {
  left.flatMap { row in
    right.map { other in
      zip(row, other).reduce(0.0) { $0 + $1.0 * $1.1 }
    }
  }
}

extension Worker {
  @inline(never) static func executePCA(_ input: PCAFixtureInput) async throws -> Output {
    let model = try PCA(nComponents: input.n_components, svdSolver: .full, device: .cpu)
    let scores = try await model.fitTransform(input.features)
    let queryScores = try await model.transform(input.query)
    guard await model.resolvedDevice == .cpu else {
      throw BenchmarkFailure("PCA did not resolve to CPU")
    }
    let width = input.features[0].count
    let k = input.n_components
    guard let mean = await model.mean, mean.count == width,
      let components = await model.components, components.count == k,
      components.allSatisfy({ $0.count == width }),
      let variance = await model.explainedVariance, variance.count == k,
      let ratios = await model.explainedVarianceRatio, ratios.count == k,
      scores.count == input.features.count, scores.allSatisfy({ $0.count == k }),
      queryScores.count == input.query.count, queryScores.allSatisfy({ $0.count == k })
    else { throw BenchmarkFailure("Missing or invalid PCA result dimensions") }
    let projectors = components.flatMap { component in
      component.flatMap { a in component.map { b in a * b } }
    }
    var output = mean
    output += variance
    output += ratios
    output += projectors
    output += scoreGram(scores, scores)
    output += scoreGram(queryScores, queryScores)
    output += scoreGram(scores, queryScores)
    return .values(output)
  }

  @inline(never) static func executeNaiveBayes(_ input: NBFixtureInput) async throws -> Output {
    let model = NaiveBayesClassifier(alpha: input.alpha)
    try await model.fit(features: input.features, targets: input.targets)
    let classes = await model.classes
    let probabilities = try await model.predictProbability(features: input.query)
    let indices = try await model.predict(features: input.query)
    guard !classes.isEmpty, classes == Array(Set(input.targets)).sorted(),
      probabilities.count == input.query.count,
      probabilities.allSatisfy({ $0.count == classes.count }),
      indices.count == input.query.count,
      indices.allSatisfy({ classes.indices.contains($0) })
    else { throw BenchmarkFailure("Missing or invalid Naive Bayes result dimensions") }
    return .values(classes + probabilities.flatMap { $0 } + indices.map(Double.init))
  }
}
