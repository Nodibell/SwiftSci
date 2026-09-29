import Foundation
import SwiftCluster
import SwiftExplain
import SwiftSciBenchmarkSupport

struct VectorCosineFixtureInput {
  let store: VectorStore
  let query: [Double]
  let topK: Int
  let entryCount: Int

  private struct Payload: Decodable {
    let operation: String
    let vectors: [[Double]]
    let query: [Double]
    let top_k: Int
  }

  static func decode(_ data: Data, rows: Int) throws -> VectorCosineFixtureInput {
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      Set(object.keys) == ["operation", "vectors", "query", "top_k"]
    else { throw BenchmarkFailure("Unexpected vector cosine fixture fields") }
    let payload = try JSONDecoder().decode(Payload.self, from: data)
    guard payload.operation == "vector-cosine", rows > 0,
      payload.vectors.count == rows, !payload.query.isEmpty,
      payload.query.allSatisfy(\.isFinite), payload.query.contains(where: { $0 != 0 }),
      payload.vectors.allSatisfy({ row in
        row.count == payload.query.count && row.allSatisfy(\.isFinite)
          && row.contains(where: { $0 != 0 })
      }), payload.top_k > 0, payload.top_k <= rows
    else { throw BenchmarkFailure("Invalid vector cosine fixture dimensions or values") }
    let store = VectorStore(metric: .cosineSimilarity)
    store.addBatch(entries: payload.vectors.enumerated().map {
      VectorEntry(id: String($0.offset), vector: $0.element)
    })
    return VectorCosineFixtureInput(
      store: store, query: payload.query, topK: payload.top_k, entryCount: rows)
  }
}

struct KernelSHAPFixtureInput: Decodable, Sendable {
  struct Polynomial: Decodable, Sendable {
    let bias: Double
    let weights: [Double]
    let interaction: Double

    func predict(_ row: [Double]) -> Double {
      bias + weights[0] * row[0] + weights[1] * row[1] + interaction * row[0] * row[1]
    }
  }

  let operation: String
  let instance: [Double]
  let background: [[Double]]
  let model: Polynomial

  static func decode(_ data: Data, rows: Int) throws -> KernelSHAPFixtureInput {
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      Set(object.keys) == ["operation", "instance", "background", "model"],
      let model = object["model"] as? [String: Any],
      Set(model.keys) == ["bias", "weights", "interaction"]
    else { throw BenchmarkFailure("Unexpected KernelSHAP fixture fields") }
    let input = try JSONDecoder().decode(Self.self, from: data)
    guard input.operation == "kernel-shap", rows > 0,
      input.background.count == rows, input.instance.count == 2,
      input.instance.allSatisfy(\.isFinite),
      input.background.allSatisfy({ $0.count == 2 && $0.allSatisfy(\.isFinite) }),
      input.model.weights.count == 2, input.model.weights.allSatisfy(\.isFinite),
      input.model.bias.isFinite, input.model.interaction.isFinite
    else { throw BenchmarkFailure("Invalid KernelSHAP fixture dimensions or values") }
    return input
  }
}

extension Worker {
  @inline(never) static func executeVectorCosine(_ input: VectorCosineFixtureInput) throws -> Output {
    let matches = input.store.search(query: input.query, topK: input.topK)
    var output = [(Int, Double)]()
    output.reserveCapacity(matches.count)
    for match in matches {
      guard let index = Int(match.id) else {
        throw BenchmarkFailure("Vector store returned a nonnumeric fixture ID")
      }
      output.append((index, match.score))
    }
    return .cosineResults(output)
  }

  @inline(never) static func executeKernelSHAP(_ input: KernelSHAPFixtureInput) async -> Output {
    let model = input.model
    let contributions = await KernelSHAP().explain(
      model: { row in model.predict(row) }, instance: input.instance,
      background: input.background, numCoalitions: 4)
    let baseline = (0..<2).map { column in
      input.background.reduce(0.0) { $0 + $1[column] } / Double(input.background.count)
    }
    return .values([model.predict(baseline)] + contributions + [model.predict(input.instance)])
  }
}

func canonicalCosineFixtureValues(
  _ matches: [(Int, Double)], entryCount: Int, expectedCount: Int
) throws -> [Double] {
  guard matches.count == expectedCount,
    Set(matches.map { $0.0 }).count == matches.count,
    matches.allSatisfy({ $0.0 >= 0 && $0.0 < entryCount && $0.1.isFinite })
  else { throw BenchmarkFailure("Invalid vector cosine result shape or identity") }
  guard zip(matches, matches.dropFirst()).allSatisfy({ $0.0.1 >= $0.1.1 }) else {
    throw BenchmarkFailure("Vector results are not closest first")
  }
  return matches.sorted {
    $0.1 == $1.1 ? $0.0 < $1.0 : $0.1 > $1.1
  }.flatMap { [Double($0.0), $0.1] }
}
