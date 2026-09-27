import CoreFoundation
import Foundation
import SwiftDataFrame
import SwiftSciBenchmarkSupport

private enum SemanticColumnValues {
  case int64([Int64?])
  case float64([Double?])
  case bool([Bool?])
  case utf8([String?])

  var numeric: Bool {
    switch self { case .int64, .float64: return true; default: return false }
  }

  var matrixCompatible: Bool {
    switch self { case .utf8: return false; default: return true }
  }

  func column(named name: String) -> any AnyColumn {
    switch self {
    case .int64(let values): return TypedColumn<Int64>(name: name, values: values)
    case .float64(let values): return TypedColumn<Double>(name: name, values: values)
    case .bool(let values): return TypedColumn<Bool>(name: name, values: values)
    case .utf8(let values): return TypedColumn<String>(name: name, values: values)
    }
  }
}

struct DataFrameSemanticInput {
  private struct Column {
    let name: String
    let type: String
    let values: SemanticColumnValues
  }

  private enum Action {
    case gather([Int])
    case select([String])
    case sort(String, ascending: Bool)
    case filter(String, FilterCondition)
    case matrix([String], target: String)
    case replace(String, [Int64?])
    case groupCount([String], value: String)
  }

  private let columns: [Column]
  private let action: Action

  static func decode(_ data: Data, rows: Int) throws -> Self {
    let object = try fields(JSONSerialization.jsonObject(with: data),
                            expected: ["operation", "action", "columns", "parameters"])
    guard rows > 0, object["operation"] as? String == "dataframe-semantics",
      let rawColumns = object["columns"] as? [Any], !rawColumns.isEmpty,
      let actionName = object["action"] as? String,
      let parameters = object["parameters"]
    else { throw BenchmarkFailure("Invalid dataframe semantic fixture") }
    var columns = [Column]()
    var names = Set<String>()
    for raw in rawColumns {
      let column = try fields(raw, expected: ["name", "type", "values"])
      guard let name = column["name"] as? String, !name.isEmpty,
        names.insert(name).inserted, let type = column["type"] as? String,
        let rawValues = column["values"] as? [Any], rawValues.count == rows
      else { throw BenchmarkFailure("Invalid dataframe column name, type, or length") }
      columns.append(Column(name: name, type: type, values: try values(rawValues, type: type)))
    }
    func column(_ raw: Any?) throws -> Column {
      guard let name = raw as? String, let found = columns.first(where: { $0.name == name })
      else { throw BenchmarkFailure("Unknown dataframe column") }
      return found
    }
    func columnList(_ raw: Any?, unique: Bool, nonempty: Bool) throws -> [String] {
      guard let selected = raw as? [String], !nonempty || !selected.isEmpty,
        !unique || Set(selected).count == selected.count,
        selected.allSatisfy({ names.contains($0) })
      else { throw BenchmarkFailure("Invalid dataframe column selection") }
      return selected
    }
    let action: Action
    switch actionName {
    case "gather":
      let p = try fields(parameters, expected: ["indices"])
      guard let raw = p["indices"] as? [Any] else {
        throw BenchmarkFailure("Gather indices must be an array")
      }
      let indices = try raw.map { value -> Int in
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
          !["f", "d"].contains(String(cString: number.objCType)),
          let index = Int(exactly: number.doubleValue), index >= 0, index < rows
        else { throw BenchmarkFailure("Gather index outside source rows") }
        return index
      }
      action = .gather(indices)
    case "select":
      let p = try fields(parameters, expected: ["columns"])
      action = .select(try columnList(p["columns"], unique: true, nonempty: true))
    case "sort":
      let p = try fields(parameters, expected: ["column", "ascending"])
      let source = try column(p["column"])
      guard source.values.numeric else { throw BenchmarkFailure("Sort column must be numeric") }
      if case .float64(let values) = source.values, values.contains(where: { $0?.isNaN == true }) {
        throw BenchmarkFailure("NaN sorting is outside the semantic fixture contract")
      }
      action = .sort(source.name, ascending: try boolean(p["ascending"]))
    case "filter":
      let p = try fields(parameters, expected: ["column", "predicate", "value"])
      let source = try column(p["column"])
      guard source.values.numeric, let predicate = p["predicate"] as? String,
        let rawValue = p["value"]
      else { throw BenchmarkFailure("Invalid numeric filter") }
      let condition: FilterCondition
      if predicate == "isNull" || predicate == "isNotNull" {
        guard rawValue is NSNull else { throw BenchmarkFailure("Null predicate requires null value") }
        condition = predicate == "isNull" ? .isNull : .isNotNull
      } else {
        let parsed = try values([rawValue], type: source.type)
        let threshold: any Sendable
        switch parsed {
        case .int64(let values):
          guard let value = values[0] else { throw BenchmarkFailure("Null filter threshold") }
          threshold = value
        case .float64(let values):
          guard let value = values[0] else { throw BenchmarkFailure("Null filter threshold") }
          threshold = value
        default: throw BenchmarkFailure("Invalid filter threshold type")
        }
        switch predicate {
        case "eq": condition = .equals(threshold)
        case "ne": condition = .notEquals(threshold)
        case "lt": condition = .lessThan(threshold)
        case "le": condition = .lessThanOrEqual(threshold)
        case "gt": condition = .greaterThan(threshold)
        case "ge": condition = .greaterThanOrEqual(threshold)
        default: throw BenchmarkFailure("Unknown filter predicate")
        }
      }
      action = .filter(source.name, condition)
    case "matrix":
      let p = try fields(parameters, expected: ["columns", "target"])
      let selected = try columnList(p["columns"], unique: false, nonempty: false)
      let target = try column(p["target"])
      for name in selected + [target.name] {
        guard try column(name).values.matrixCompatible else {
          throw BenchmarkFailure("Matrix requires numeric or bool columns")
        }
      }
      action = .matrix(selected, target: target.name)
    case "replace":
      let p = try fields(parameters, expected: ["column", "values"])
      let source = try column(p["column"])
      guard source.type == "int64", let raw = p["values"] as? [Any], raw.count == rows,
        case .int64(let replacement) = try values(raw, type: "int64")
      else { throw BenchmarkFailure("Replacement must preserve int64 type and row count") }
      action = .replace(source.name, replacement)
    case "group-count":
      let p = try fields(parameters, expected: ["columns", "value"])
      let keys = try columnList(p["columns"], unique: true, nonempty: true)
      let value = try column(p["value"])
      guard keys.count <= 2, !keys.contains(value.name), value.values.numeric else {
        throw BenchmarkFailure("Invalid group-count keys or value column")
      }
      for name in keys {
        guard ["utf8", "int64"].contains(try column(name).type) else {
          throw BenchmarkFailure("Group keys must be utf8 or int64")
        }
      }
      action = .groupCount(keys, value: value.name)
    default: throw BenchmarkFailure("Unknown dataframe semantic action")
    }
    return Self(columns: columns, action: action)
  }

  private static func fields(_ raw: Any, expected: Set<String>) throws -> [String: Any] {
    guard let object = raw as? [String: Any], Set(object.keys) == expected else {
      throw BenchmarkFailure("Unexpected dataframe semantic fields")
    }
    return object
  }

  private static func boolean(_ raw: Any?) throws -> Bool {
    guard let number = raw as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
      throw BenchmarkFailure("Expected JSON boolean")
    }
    return number.boolValue
  }

  private static func values(_ raw: [Any], type: String) throws -> SemanticColumnValues {
    switch type {
    case "int64":
      return .int64(try raw.map { cell in
        if cell is NSNull { return nil }
        guard let text = cell as? String, let value = Int64(text), String(value) == text else {
          throw BenchmarkFailure("Expected canonical decimal int64 string")
        }
        return value
      })
    case "float64":
      return .float64(try raw.map { cell in
        if cell is NSNull { return nil }
        if let text = cell as? String {
          switch text {
          case "NaN": return .nan
          case "+Inf": return .infinity
          case "-Inf": return -.infinity
          default: throw BenchmarkFailure("Invalid special float64 string")
          }
        }
        guard let number = cell as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
          number.doubleValue.isFinite else { throw BenchmarkFailure("Expected finite float64 number") }
        return number.doubleValue
      })
    case "bool":
      return .bool(try raw.map { $0 is NSNull ? nil : try boolean($0) })
    case "utf8":
      return .utf8(try raw.map { cell in
        if cell is NSNull { return nil }
        guard let text = cell as? String else { throw BenchmarkFailure("Expected utf8 string") }
        return text
      })
    default: throw BenchmarkFailure("Unsupported dataframe fixture type")
    }
  }

  func execute() throws -> [Double] {
    let source = try DataFrame(columns: columns.map { $0.values.column(named: $0.name) })
    switch action {
    case .gather(let indices): return try semanticFrames([source, source.gathered(at: indices)])
    case .select(let names): return try semanticFrames([source, source.select(names)])
    case .sort(let name, let ascending):
      return try semanticFrames([source, source.sortBy(name, ascending: ascending)])
    case .filter(let name, let condition):
      return try semanticFrames([source, source.filter(column: name, where: condition)])
    case .replace(let name, let values):
      guard let retained = source[column: name, as: Int64.self] else {
        throw BenchmarkFailure("Missing concrete int64 replacement source")
      }
      let selected = try source.select([name])
      var copy = source
      var extracted = retained.values
      extracted.replaceSubrange(extracted.indices, with: values)
      copy = try copy.withColumn(name, column: TypedColumn<Int64>(name: "temporary", values: extracted))
      return try semanticFrames([source, copy, selected, DataFrame(columns: [retained])])
    case .groupCount(let keys, let value):
      let grouped = keys.count == 1 ? source.groupBy(keys[0]) : source.groupBy(keys[0], keys[1])
      return try semanticFrames([source, grouped.count(), grouped.agg([value: .count])])
    case .matrix(let names, let target):
      let nested = try source.toFeatureMatrix(names)
      let flat = try source.toFlatFeatureMatrix(names)
      let vector = try source.toTargetVector(target)
      let rows = nested.count
      let cols = nested.first?.count ?? 0
      guard nested.allSatisfy({ $0.count == cols }), flat.rows == rows, flat.cols == cols,
        flat.flat.count == rows * cols else {
        throw BenchmarkFailure("Matrix exports returned inconsistent dimensions")
      }
      var result = try semanticFrames([source])
      result += [Double(rows), Double(cols)]
      for row in nested { for value in row { result += semanticFloat(value) } }
      for value in flat.flat { result += semanticFloat(value) }
      result.append(Double(vector.count))
      for value in vector { result += semanticFloat(value) }
      return result
    }
  }
}

private func semanticBits(_ bits: UInt64) -> [Double] {
  [Double(bits >> 32), Double(bits & 0xffff_ffff)]
}

private func semanticFloat(_ value: Double) -> [Double] {
  semanticBits(value.isNaN ? 0x7ff8_0000_0000_0000 : value.bitPattern)
}

private func semanticText(_ value: String) -> [Double] {
  let bytes = value.utf8.map(Double.init)
  return [Double(bytes.count)] + bytes
}

private func semanticColumn<T: SupportedType>(
  _ column: TypedColumn<T>, code: Double, payload: (T) -> [Double]
) -> [Double] {
  var result = [code] + semanticText(column.name) + [Double(column.nullCount)]
  for cell in column.values {
    if let cell { result.append(1); result += payload(cell) } else { result.append(0) }
  }
  return result
}

private func semanticFrames(_ frames: [DataFrame]) throws -> [Double] {
  var result = [Double(frames.count)]
  for frame in frames {
    result += [Double(frame.shape.rows), Double(frame.shape.columns)]
    for column in frame.columns {
      switch column {
      case let column as TypedColumn<Int64>:
        result += semanticColumn(column, code: 1) { semanticBits(UInt64(bitPattern: $0)) }
      case let column as TypedColumn<Double>:
        result += semanticColumn(column, code: 2, payload: semanticFloat)
      case let column as TypedColumn<Bool>:
        result += semanticColumn(column, code: 3) { [$0 ? 1 : 0] }
      case let column as TypedColumn<String>:
        result += semanticColumn(column, code: 4, payload: semanticText)
      default: throw BenchmarkFailure("Unsupported concrete dataframe output column")
      }
    }
  }
  return result
}

extension Worker {
  @inline(never) static func executeDataFrameSemantics(_ input: DataFrameSemanticInput) throws -> Output {
    .values(try input.execute())
  }
}
