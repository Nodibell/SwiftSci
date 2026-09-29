import Foundation
import SwiftDataFrame
import SwiftPreprocessing
import SwiftSciBenchmarkSupport

extension Worker {
  static let wineFeatures = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
    "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol",
  ]

  static func winePipeline(path: String, rows: Int) async throws -> Output {
    let loaded = try await DataFrame(csv: URL(fileURLWithPath: path))
    guard loaded.shape.rows == rows, loaded.columnNames == ["id"] + wineFeatures + ["quality"] else {
      throw BenchmarkFailure("Wine input shape mismatch")
    }
    let selected = try loaded.filter(column: "quality", where: .greaterThanOrEqual(6)).sortBy("alcohol")
    let scaled = try selected.standardScale(columns: wineFeatures).scaled
    let ids = try scaled.toTargetVector("id")
    let targets = try scaled.toTargetVector("quality")
    let matrix = try scaled.toFlatFeatureMatrix(wineFeatures).flat
    return .values(ids + targets + matrix)
  }

  static func h2o(_ op: String, frame: DataFrame) throws -> Output {
    switch op {
    case "h2o-q1": return .frame(frame.groupBy("id1").agg(["v1": .sum]))
    case "h2o-q2": return .frame(frame.groupBy("id1", "id2").agg(["v1": .sum]))
    case "h2o-q3": return .frame(frame.groupBy("id3").agg(["v1": .sum, "v3": .mean]))
    case "h2o-q4": return .frame(frame.groupBy("id4").agg(["v1": .mean, "v2": .mean, "v3": .mean]))
    case "h2o-q5": return .frame(frame.groupBy("id6").agg(["v1": .sum, "v2": .sum, "v3": .sum]))
    default: throw BenchmarkFailure("Unknown H2O-derived query")
    }
  }

  static func canonicalH2O(_ result: DataFrame, operation: String) throws -> [Double] {
    let keys: [String]
    let columns: [String]
    switch operation {
    case "h2o-q1": keys = ["id1"]; columns = ["v1_sum"]
    case "h2o-q2": keys = ["id1", "id2"]; columns = ["v1_sum"]
    case "h2o-q3": keys = ["id3"]; columns = ["v1_sum", "v3_mean"]
    case "h2o-q4": keys = ["id4"]; columns = ["v1_mean", "v2_mean", "v3_mean"]
    case "h2o-q5": keys = ["id6"]; columns = ["v1_sum", "v2_sum", "v3_sum"]
    default: throw BenchmarkFailure("Unknown H2O-derived query")
    }
    guard Set(result.columnNames) == Set(keys + columns) else {
      throw BenchmarkFailure("Grouped output schema mismatch")
    }
    let measures = try columns.map { try result.toTargetVector($0) }
    let rows = try (0..<result.shape.rows).map { row -> [Double] in
      let key = try keys.map { name -> Double in
        guard let column = result[column: name, as: String.self],
          let label = column.value(at: row) as? String else {
          throw BenchmarkFailure("Missing grouped key")
        }
        let isStringKey = ["id1", "id2", "id3"].contains(name)
        guard !isStringKey || label.hasPrefix("key"),
          let value = Double(isStringKey ? String(label.dropFirst(3)) : label) else {
          throw BenchmarkFailure("Malformed grouped key")
        }
        return value
      }
      return key + measures.map { $0[row] }
    }
    return rows.sorted { $0.prefix(keys.count).lexicographicallyPrecedes($1.prefix(keys.count)) }.flatMap { $0 }
  }
}
