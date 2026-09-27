import CoreFoundation
import Foundation
import MLX
import SwiftDataFrame
import SwiftSciBenchmarkSupport

struct BoundarySweepInput {
  private struct Recipe: Decodable {
    let operation: String
    let recipe: String
    let device: String
    let dtype: String
    let stage: String
    let rows: Int
    let columns: Int
  }
  let stage: String
  let input: BoundaryInput
  let source: DataFrame?
  let prepared: BoundaryPrepared?

  static func decode(_ data: Data, rows: Int) throws -> Self {
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      Set(object.keys) == Set(["operation", "recipe", "device", "dtype", "stage", "rows", "columns"]),
      ["rows", "columns"].allSatisfy({ key in
        guard let n = object[key] as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else { return false }
        return !["f", "d"].contains(String(cString: n.objCType))
      })
    else { throw BenchmarkFailure("Unexpected boundary sweep descriptor fields") }
    let recipe = try JSONDecoder().decode(Recipe.self, from: data)
    guard recipe.operation == "dataframe-model-sweep", recipe.recipe == "dyadic-v1",
      [128, 1024, 8192].contains(recipe.rows), recipe.rows == rows, [8, 64].contains(recipe.columns),
      ["cpu", "gpu"].contains(recipe.device), ["float32", "float64"].contains(recipe.dtype),
      !(recipe.device == "gpu" && recipe.dtype == "float64"),
      ["conversion", "prepared", "pipeline"].contains(recipe.stage)
    else { throw BenchmarkFailure("Invalid bounded boundary sweep descriptor") }
    let names = (0..<recipe.columns).map { "x\($0)" }
    let input = BoundaryInput(operation: "dataframe-model", device: recipe.device, dtype: recipe.dtype,
      row_ids: (0..<rows).map { Int64(($0 * 37) % rows) }, feature_names: names,
      feature_order: Array(names.reversed()),
      features: (0..<rows).map { i in (0..<recipe.columns).map { j in Double((i * 7 + j * 3) % 19 - 9) / 8 } },
      targets: (0..<rows).map { Double($0 % 17) / 4 - 1 },
      filter_values: (0..<rows).map { Double($0 % 4) }, filter_threshold: 1,
      sort_keys: (0..<rows).map { Double($0 % 11) }, ascending: true,
      weights: (0..<recipe.columns).map { Double($0 + 1) / 8 }, bias: -0.375)
    let source = recipe.stage == "pipeline" ? nil : try input.sourceFrame()
    let prepared: BoundaryPrepared?
    if recipe.stage == "prepared", let source {
      prepared = try input.onDevice { try input.convert(source: source) }
    } else {
      prepared = nil
    }
    return Self(stage: recipe.stage, input: input, source: source, prepared: prepared)
  }

  func execute() throws -> [Double] {
    if stage == "pipeline" { return try input.execute() }
    return try input.onDevice {
      if stage == "conversion" {
        guard let source else { throw BenchmarkFailure("Missing prepared sweep frame") }
        let actual = try input.convert(source: source)
        return [Double(actual.rows), Double(actual.cols), input.dtype == "float32" ? 32 : 64]
          + actual.ids.map(Double.init) + actual.matrix + actual.target + (try input.read(actual.matrixTensor))
      }
      guard let prepared else { throw BenchmarkFailure("Missing prepared sweep tensors") }
      let actual = try input.compute(prepared)
      return [Double(prepared.matrixTensor.shape[0]), Double(prepared.matrixTensor.shape[1]),
        prepared.matrixTensor.dtype == .float32 ? 32 : 64] + actual.prediction + actual.residual
    }
  }
}

extension Worker {
  @inline(never) static func executeBoundarySweep(_ input: BoundarySweepInput) throws -> Output {
    .values(try input.execute())
  }
}
