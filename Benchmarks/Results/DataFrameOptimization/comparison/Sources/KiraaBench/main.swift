import Foundation
import SwiftPandas
import Darwin

@main struct Bench {
 static func main() throws {
  let args = CommandLine.arguments
  if args[1] == "startup" { print("{}"); return }
  let op = args[1], n = Int(args[2])!, repetitions = Int(args[3])!
  let x = (0..<n).map { Double(($0 * 7919) % 100003) / 100.0 }
  let values: [Double?] = (0..<n).map { $0 % 101 == 0 ? nil : x[$0] }
  let numeric = NativeArray<Double>(x)
  var columns: [(String, Column)] = [
   ("id", Column(takingInt64s: (0..<n).map { Int64($0) })),
   ("group", Column(takingInt64s: (0..<n).map { Int64($0 % 100) })),
   ("value", Series(values).data)]
  if op == "group_two" { columns.append(("group2", Column(takingInt64s: (0..<n).map { Int64($0 % 97) }))) }
  if op == "filter_int" { columns[2] = ("value", .int64(NullableArray<Int64>((0..<n).map { $0 % 101 == 0 ? nil : Int64(($0 * 7919) % 100003) }))) }
  let df = DataFrame(columns: columns)
  var frame: DataFrame? = nil
  var numbers = [Double]()
  var times = [Double]()
  for iteration in 0..<(repetitions + 2) {
   frame = nil
   numbers = []
   let start = DispatchTime.now().uptimeNanoseconds
   switch op {
    case "csv": frame = try DataFrame.readCSV(path: args[4])
    case "filter":
     let predicate = ColumnPredicate.comparison(column: "value", op: .gt, value: .double(500.0))
     frame = df.filter(mask: predicate.evaluate(on: df))
    case "filter_int":
     let predicate = ColumnPredicate.comparison(column: "value", op: .gt, value: .double(50000.0))
     frame = df.filter(mask: predicate.evaluate(on: df))
    case "group_two": frame = df.groupBy(["group", "group2"]).sum()
    case "pipeline":
     let predicate = ColumnPredicate.comparison(column: "value", op: .gt, value: .double(500.0))
     frame = df.filter(mask: predicate.evaluate(on: df)).sortValues(by: "value", ascending: false).groupBy("group").sum()
    case "sort": frame = df.sortValues(by: "value", ascending: false)
    case "group": frame = df.groupBy("group").sum()
    case "stats": numbers = [numeric.mean(), numeric.variance(ddof: 1)]
    default: fatalError("Unknown operation")
   }
   let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
   if iteration >= 2 { times.append(seconds) }
  }
  var usage = rusage()
  getrusage(RUSAGE_SELF, &usage)
  var result: [String: Any] = ["times": times, "peak_rss_bytes": usage.ru_maxrss,
   "group_gpu_threshold": MetalDispatch.groupByThreshold,
   "metal_disabled": MetalDispatch.metalDisabled]
  if let frame {
   var data = [String: [Any]]()
   var types = [String: String]()
   for name in frame.columnNames {
    let column = frame[name].data
    types[name] = String(describing: column.dtype)
    data[name] = (0..<frame.rowCount).map { column.value(at: $0) ?? NSNull() }
   }
   if ["group", "pipeline"].contains(op) && data["group"] == nil {
    data["group"] = frame.indexLabels.map { Double($0)! as Any }
    types["group"] = "String index labels, normalized to numbers after timing"
   }
   result["result"] = data
   result["output_types"] = types
  } else { result["result"] = ["values": numbers] }
  let output = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
  FileHandle.standardOutput.write(output)
 }
}
