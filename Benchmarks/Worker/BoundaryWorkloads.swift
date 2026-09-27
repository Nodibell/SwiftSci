import CoreFoundation
import Foundation
import MLX
import SwiftDataFrame
import SwiftSciBenchmarkSupport

struct BoundaryInput: Decodable {
  let operation: String
  let device: String
  let dtype: String
  let row_ids: [Int64]
  let feature_names: [String]
  let feature_order: [String]
  let features: [[Double]]
  let targets: [Double]
  let filter_values: [Double]
  let filter_threshold: Double
  let sort_keys: [Double]
  let ascending: Bool
  let weights: [Double]
  let bias: Double

  static func decode(_ data: Data, rows: Int) throws -> Self {
    let keys: Set<String> = ["operation", "device", "dtype", "row_ids", "feature_names",
      "feature_order", "features", "targets", "filter_values", "filter_threshold",
      "sort_keys", "ascending", "weights", "bias"]
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      Set(object.keys) == keys, let rawIDs = object["row_ids"] as? [Any],
      rawIDs.allSatisfy({ value in
        guard let n = value as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else { return false }
        return !["f", "d"].contains(String(cString: n.objCType))
      })
    else { throw BenchmarkFailure("Unexpected dataframe-model input fields or row IDs") }
    let input = try JSONDecoder().decode(Self.self, from: data)
    let width = input.feature_names.count
    let reserved: Set<String> = ["row_id", "target", "filter", "sort"]
    func bounded(_ values: [Double], count: Int) -> Bool {
      values.count == count && values.allSatisfy { $0.isFinite && abs($0) <= 1024 }
    }
    guard input.operation == "dataframe-model", (2...100000).contains(rows),
      ["cpu", "gpu"].contains(input.device), ["float32", "float64"].contains(input.dtype),
      !(input.device == "gpu" && input.dtype == "float64"), (1...64).contains(width),
      Set(input.feature_names).count == width,
      input.feature_names.allSatisfy({ !$0.isEmpty && !reserved.contains($0) }),
      input.feature_order.count == width, Set(input.feature_order) == Set(input.feature_names),
      input.row_ids.count == rows, Set(input.row_ids).count == rows,
      input.row_ids.allSatisfy({ $0 >= 0 && $0 <= Int64(Int32.max) }),
      input.features.count == rows, input.features.allSatisfy({ bounded($0, count: width) }),
      bounded(input.targets, count: rows), bounded(input.filter_values, count: rows),
      bounded(input.sort_keys, count: rows), bounded(input.weights, count: width),
      bounded([input.filter_threshold, input.bias], count: 2),
      input.filter_values.contains(where: { $0 >= input.filter_threshold })
    else { throw BenchmarkFailure("Invalid bounded dataframe-model contract") }
    return input
  }

  func sourceFrame() throws -> DataFrame {
    var columns: [any AnyColumn] = feature_names.enumerated().map { j, name in
      TypedColumn<Double>(name: name, values: features.map { Optional($0[j]) })
    }
    columns += [TypedColumn<Int64>(name: "row_id", values: row_ids.map(Optional.some)),
      TypedColumn<Double>(name: "target", values: targets.map(Optional.some)),
      TypedColumn<Double>(name: "filter", values: filter_values.map(Optional.some)),
      TypedColumn<Double>(name: "sort", values: sort_keys.map(Optional.some))]
    return try DataFrame(columns: columns)
  }

  func onDevice<T>(_ body: () throws -> T) rethrows -> T {
    let selected: Device = device == "cpu" ? .cpu : .gpu
    return try Device.withDefaultDevice(selected) {
      try Stream.withNewDefaultStream(device: selected) { try body() }
    }
  }

  func convert(source: DataFrame) throws -> BoundaryPrepared {
    let selected = try source.filter(column: "filter", where: .greaterThanOrEqual(filter_threshold))
      .sortBy("sort", ascending: ascending)
    let flat = try selected.toFlatFeatureMatrix(feature_order)
    let nested = try selected.toFeatureMatrix(feature_order)
    let target = try selected.toTargetVector("target")
    guard flat.rows > 0, flat.cols == feature_order.count,
      nested.count == flat.rows, nested.allSatisfy({ $0.count == flat.cols }),
      nested.flatMap({ $0 }) == flat.flat, target.count == flat.rows,
      let idColumn = selected[column: "row_id", as: Int64.self], idColumn.nullCount == 0
    else { throw BenchmarkFailure("Dataframe-model conversion dimensions or exports disagree") }
    let ids = idColumn.values.compactMap { $0 }
    func tensor(_ values: [Double], shape: [Int]) -> MLXArray {
      if dtype == "float32" { return MLXArray(values.map(Float.init), shape) }
      return values.withUnsafeBufferPointer { MLXArray($0, shape) }
    }
    let matrixTensor = tensor(flat.flat, shape: [flat.rows, flat.cols])
    let targetTensor = tensor(target, shape: [flat.rows])
    let weightTensor = tensor(weights, shape: [flat.cols])
    let biasTensor = tensor([bias], shape: [])
    let expected: DType = dtype == "float32" ? .float32 : .float64
    guard matrixTensor.shape == [flat.rows, flat.cols], targetTensor.shape == [flat.rows],
      weightTensor.shape == [flat.cols], biasTensor.shape == [],
      [matrixTensor, targetTensor, weightTensor, biasTensor].allSatisfy({ $0.dtype == expected })
    else { throw BenchmarkFailure("Dataframe-model tensor shape or dtype mismatch") }
    eval(matrixTensor, targetTensor, weightTensor, biasTensor)
    StreamOrDevice.default.stream.synchronize()
    return BoundaryPrepared(source: source, selected: selected, ids: ids, matrix: flat.flat,
      target: target, rows: flat.rows, cols: flat.cols, matrixTensor: matrixTensor,
      targetTensor: targetTensor, weightTensor: weightTensor, biasTensor: biasTensor)
  }

  func compute(_ prepared: BoundaryPrepared) throws -> (prediction: [Double], residual: [Double]) {
    let prediction = matmul(prepared.matrixTensor, prepared.weightTensor) + prepared.biasTensor
    let residual = prediction - prepared.targetTensor
    let expected: DType = dtype == "float32" ? .float32 : .float64
    guard prediction.shape == [prepared.rows], residual.shape == [prepared.rows],
      prediction.dtype == expected, residual.dtype == expected else {
      throw BenchmarkFailure("Dataframe-model arithmetic shape or dtype mismatch")
    }
    eval(prediction, residual)
    StreamOrDevice.default.stream.synchronize()
    return (try read(prediction), try read(residual))
  }

  func read(_ tensor: MLXArray) throws -> [Double] {
    let result = dtype == "float32" ? tensor.asArray(Float.self).map(Double.init) : tensor.asArray(Double.self)
    guard result.allSatisfy(\.isFinite) else { throw BenchmarkFailure("Nonfinite dataframe-model output") }
    return result
  }

  func isolation(_ prepared: BoundaryPrepared) throws -> Bool {
    var matrix = prepared.matrix
    var target = prepared.target
    matrix[0] += 17
    target[0] += 19
    let copied = try prepared.source.withColumn(feature_names[0], column:
      TypedColumn<Double>(name: "replacement", values: [Double?](repeating: -23, count: row_ids.count)))
    let sourceMatrix = try prepared.source.toFeatureMatrix(feature_names)
    let sourceTarget = try prepared.source.toTargetVector("target")
    let selectedMatrix = try prepared.selected.toFlatFeatureMatrix(feature_order)
    let selectedTarget = try prepared.selected.toTargetVector("target")
    let replacement = try copied.toTargetVector(feature_names[0])
    let tensorValues = try read(prepared.matrixTensor)
    let tensorTargets = try read(prepared.targetTensor)
    return sourceMatrix == features && sourceTarget == targets
      && selectedMatrix.flat == prepared.matrix && selectedTarget == prepared.target
      && replacement == [Double](repeating: -23, count: row_ids.count)
      && tensorValues[0] != matrix[0]
      && tensorTargets[0] != target[0]
  }

  func execute() throws -> [Double] {
    try onDevice {
      let prepared = try convert(source: sourceFrame())
      let result = try compute(prepared)
      let tensor = try read(prepared.matrixTensor)
      let isolated = try isolation(prepared)
      return [Double(prepared.rows), Double(prepared.cols), dtype == "float32" ? 32 : 64]
        + prepared.ids.map(Double.init) + prepared.matrix + prepared.target + tensor
        + result.prediction + result.residual + [isolated ? 1 : 0]
    }
  }
}

struct BoundaryPrepared {
  let source: DataFrame
  let selected: DataFrame
  let ids: [Int64]
  let matrix: [Double]
  let target: [Double]
  let rows: Int
  let cols: Int
  let matrixTensor: MLXArray
  let targetTensor: MLXArray
  let weightTensor: MLXArray
  let biasTensor: MLXArray
}

extension Worker {
  @inline(never) static func executeBoundary(_ input: BoundaryInput) throws -> Output {
    .values(try input.execute())
  }
}
