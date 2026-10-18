import Foundation
import SwiftDataFrame
import SwiftSciBenchmarkSupport

extension Worker {
  static func matchedCSVOptions(_ request: BenchmarkRequest) throws -> CSVReadOptions? {
    guard request.operation == "csv-read" || request.operation == "wine-pipeline"
      || request.operation.hasPrefix("csv-stream-") else { return nil }
    var expected = ["id": "Int64", "group": "Int64", "x": "Double", "y": "Double"]
    if request.dataset_kind == "wine-quality-v1" {
      expected = Dictionary(uniqueKeysWithValues: wineFeatures.map { ($0, "Double") })
      expected["id"] = "Int64"
      expected["quality"] = "Int64"
    } else if request.dataset_kind == "mixed-table-v1" {
      expected["group"] = "String"
      expected["flag"] = "Bool"
    } else if request.dataset_kind != "table-v1" {
      throw BenchmarkFailure("Unsupported matched CSV dataset")
    }
    guard request.csv_schema == expected else {
      throw BenchmarkFailure("CSV comparison requires the complete provided schema")
    }
    let types: [String: ColumnDType] = [
      "Int64": .int64, "Double": .float64, "String": .utf8, "Bool": .boolean,
    ]
    var options = CSVReadOptions()
    options.inferTypes = false
    options.columnTypeOverrides = expected.mapValues { types[$0]! }
    return options
  }

  static func checkedCSVSchema(_ frame: DataFrame, request: BenchmarkRequest) throws -> [String: String] {
    let actual = Dictionary(uniqueKeysWithValues: frame.columns.map { ($0.name, $0.dtype.description) })
    guard actual == request.csv_schema else { throw BenchmarkFailure("CSV input schema mismatch") }
    return actual
  }

  // Observe the reader's input types before timing transformed-output workflows.
  static func preflightCSV(_ request: BenchmarkRequest, options: CSVReadOptions) async throws -> [String: String] {
    let url = URL(fileURLWithPath: request.input_path)
    var observed = [String: String]()
    var rows = 0
    if request.operation.hasPrefix("csv-stream-") {
      for try await chunk in DataFrame.readCSVStream(contentsOf: url, chunkSize: 10000, options: options) {
        observed = try checkedCSVSchema(chunk, request: request)
        guard chunk.shape.rows == min(10000, request.rows - rows) else {
          throw BenchmarkFailure("Streaming input chunk boundaries differ")
        }
        rows += chunk.shape.rows
      }
    } else {
      let frame = try await DataFrame(csv: url, options: options)
      observed = try checkedCSVSchema(frame, request: request)
      rows = frame.shape.rows
    }
    guard rows == request.rows else { throw BenchmarkFailure("CSV input row count mismatch") }
    return observed
  }

}
