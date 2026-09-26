import Foundation
import SwiftDataFrame
import SwiftStats
import SwiftForecast
import Darwin

@main struct Bench {
 static func main() async throws {
  let args = CommandLine.arguments
  if args[1] == "startup" { print("{}"); return }
  let op = args[1], n = Int(args[2])!, repetitions = Int(args[3])!
  let x = (0..<n).map { Double(($0 * 7919) % 100003) / 100.0 }
  let y = (0..<n).map { x[$0] * 0.7 + Double($0 % 17) }
  let values: [Double?] = (0..<n).map { $0 % 101 == 0 ? nil : x[$0] }
  var columns: [any AnyColumn] = [
   TypedColumn<Int64>(name: "id", values: (0..<n).map { Int64($0) }),
   TypedColumn<Int64>(name: "group", values: (0..<n).map { Int64($0 % 100) }),
   TypedColumn<Double>(name: "value", values: values)]
  if op == "group_two" { columns.append(TypedColumn<Int64>(name: "group2", values: (0..<n).map { Int64($0 % 97) })) }
  if op == "filter_int" { columns[2] = TypedColumn<Int>(name: "value", values: (0..<n).map { $0 % 101 == 0 ? nil : ($0 * 7919) % 100003 }) }
  if op == "filter_int32" { columns[2] = TypedColumn<Int32>(name: "value", values: (0..<n).map { $0 % 101 == 0 ? nil : Int32(($0 * 7919) % 100003) }) }
  if op == "filter_float32" { columns[2] = TypedColumn<Float>(name: "value", values: values.map { $0.map(Float.init) }) }
  let df = try DataFrame(columns: columns)
  var frame: DataFrame? = nil
  var numbers = [Double]()
  var times = [Double]()
  for iteration in 0..<(repetitions + 2) {
   frame = nil
   numbers = []
   let start = DispatchTime.now().uptimeNanoseconds
   switch op {
    case "csv": frame = try await DataFrame(csv: URL(fileURLWithPath: args[4]))
    case "filter": frame = try df.filter(column: "value", where: .greaterThan(500.0))
    case "filter_int", "filter_int32": frame = try df.filter(column: "value", where: .greaterThan(50000))
    case "filter_float32": frame = try df.filter(column: "value", where: .greaterThan(500.0))
    case "group_two": frame = try df.groupBy("group", "group2").sum()
    case "pipeline": frame = try df.filter(column: "value", where: .greaterThan(500.0)).sortBy("value", ascending: false).groupBy("group").sum()
    case "sort": frame = try df.sortBy("value", ascending: false)
    case "group": frame = try df.groupBy("group").sum()
    case "group_checked": frame = try df.groupBy("group").sumChecked()
    case "stats": numbers = try [Stats.mean(x), Stats.variance(x, ddof: 1)]
    case "correlation": numbers = try [Stats.pearsonCorrelation(x, y)]
    case "forecast":
     let model = ExponentialSmoothing(method: .simple, alpha: 0.3)
     try await model.fit(series: x)
     numbers = try await model.forecast(horizon: 24).predictions
    default: fatalError("Unknown operation")
   }
   let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
   if iteration >= 2 { times.append(seconds) }
  }
  var usage = rusage()
  getrusage(RUSAGE_SELF, &usage)
  var result: [String: Any] = ["times": times, "peak_rss_bytes": usage.ru_maxrss]
  if let frame {
   var data = [String: [Any]]()
   for name in frame.columnNames {
    // All benchmark columns are numeric. Preserve nulls in the parity output.
    data[name] = (0..<frame.rowCount).map { index -> Any in
     if let column = frame[column: name, as: Double.self] { return column.values[index].map { $0 as Any } ?? NSNull() }
     if let column = frame[column: name, as: Int64.self] { return column.values[index].map { Double($0) as Any } ?? NSNull() }
     if ["group", "group_two", "pipeline"].contains(op), name.hasPrefix("group"), let column = frame[column: name, as: String.self] { return Double(column.values[index]!)! }
     if let column = frame[column: name, as: Int.self] { return column.values[index].map { Double($0) as Any } ?? NSNull() }
     if let column = frame[column: name, as: Int32.self] { return column.values[index].map { Double($0) as Any } ?? NSNull() }
     if let column = frame[column: name, as: Float.self] { return column.values[index].map { Double($0) as Any } ?? NSNull() }
     fatalError("Unexpected column type: \(name)")
    }
   }
   result["result"] = data
  } else { result["result"] = ["values": numbers] }
  let output = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
  FileHandle.standardOutput.write(output)
 }
}
