import SwiftDataFrame
import SwiftSciBenchmarkSupport

extension Worker {
  static func matchedCSVOptions(_ request: BenchmarkRequest) throws -> CSVReadOptions? {
    guard request.operation == "csv-read" else { return nil }
    var expected = ["id": "Int64", "group": "Int64", "x": "Double", "y": "Double"]
    if request.dataset_kind == "mixed-table-v1" {
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
}
