import Foundation
import Arrow

/// A bridging layer to convert between `ArrowTable` and `DataFrame`.
internal enum ArrowTableBridge {

    /// Converts an Apache Arrow `ArrowTable` into a `DataFrame`.
    static func toDataFrame(_ arrowTable: ArrowTable, nullStrategy: ArrowNullStrategy = .preserve) throws -> DataFrame {
        var columns: [any AnyColumn] = []
        
        for arrowCol in arrowTable.columns {
            let name = arrowCol.name
            let dtype = try mapArrowType(arrowCol.type)
            let count = Int(arrowCol.length)
            
            switch dtype {
            case .int32:
                let chunked: ChunkedArray<Int32> = arrowCol.data()
                columns.append(TypedColumn(name: name, values: copyValues(
                    chunked, replacingNullWith: nullStrategy == .zero ? 0 : nil)))

            case .int64:
                let chunked: ChunkedArray<Int64> = arrowCol.data()
                columns.append(TypedColumn(name: name, values: copyValues(
                    chunked, replacingNullWith: nullStrategy == .zero ? 0 : nil)))

            case .float32:
                let chunked: ChunkedArray<Float> = arrowCol.data()
                let replacement: Float? = switch nullStrategy {
                case .preserve: nil
                case .zero: 0
                case .nan: .nan
                }
                columns.append(TypedColumn(name: name, values: copyValues(chunked, replacingNullWith: replacement)))

            case .float64:
                let chunked: ChunkedArray<Double> = arrowCol.data()
                let replacement: Double? = switch nullStrategy {
                case .preserve: nil
                case .zero: 0
                case .nan: .nan
                }
                columns.append(TypedColumn(name: name, values: copyValues(chunked, replacingNullWith: replacement)))

            case .boolean:
                let chunked: ChunkedArray<Bool> = arrowCol.data()
                columns.append(TypedColumn(name: name, values: copyValues(
                    chunked, replacingNullWith: nullStrategy == .zero ? false : nil)))

            case .utf8:
                let chunked: ChunkedArray<String> = arrowCol.data()
                columns.append(TypedColumn(name: name, values: copyValues(chunked)))

            case .date32:
                let chunked: ChunkedArray<Date32> = arrowCol.data()
                var vals = [Date?]()
                vals.reserveCapacity(count)
                for i in 0..<UInt(count) {
                    if let days = chunked[i] {
                        let sec = Double(days) * 86400.0
                        vals.append(Date(timeIntervalSince1970: sec))
                    } else {
                        vals.append(nil)
                    }
                }
                columns.append(TypedColumn<Date>(name: name, values: vals))
            }
        }
        
        return try DataFrame(columns: columns)
    }

    private static func copyValues<T>(_ chunked: ChunkedArray<T>, replacingNullWith replacement: T? = nil) -> [T?] {
        var values: [T?] = []
        values.reserveCapacity(Int(chunked.length))
        // Visit each chunk once. Global row subscripts search the chunks again
        // for every value and cannot traverse empty chunks in Arrow 21.0.0.
        for chunk in chunked.arrays {
            for index in 0..<chunk.length {
                values.append(chunk[index] ?? replacement)
            }
        }
        return values
    }

    /// Converts a `DataFrame` into an Apache Arrow `RecordBatch`.
    static func toRecordBatch(_ df: DataFrame) throws -> RecordBatch {
        let rbBuilder = RecordBatch.Builder()
        
        for col in df.columns {
            let builderType: Any.Type
            switch col.dtype {
            case .int32:   builderType = Int32.self
            case .int64:   builderType = Int64.self
            case .float32: builderType = Float.self
            case .float64: builderType = Double.self
            case .boolean: builderType = Bool.self
            case .utf8:    builderType = String.self
            case .date32:  builderType = Date.self
            }
            
            let builder = try ArrowArrayBuilders.loadBuilder(builderType)
            for i in 0..<col.count {
                builder.appendAny(col.value(at: i))
            }
            
            let holder = try builder.toHolder()
            rbBuilder.addColumn(col.name, arrowArray: holder)
        }
        
        switch rbBuilder.finish() {
        case .success(let rb):
            return rb
        case .failure(let err):
            throw SwiftMLError.unsupportedFormat("Failed to build RecordBatch: \(err)")
        }
    }

    /// Converts a `DataFrame` into an Apache Arrow `ArrowTable`.
    static func toArrowTable(_ df: DataFrame) throws -> ArrowTable {
        let rb = try toRecordBatch(df)
        switch ArrowTable.from(recordBatches: [rb]) {
        case .success(let table):
            return table
        case .failure(let err):
            throw SwiftMLError.unsupportedFormat("Failed to build ArrowTable: \(err)")
        }
    }

    
    private static func mapArrowType(_ arrowType: ArrowType) throws -> ColumnDType {
        switch arrowType.id {
        case .int32:   return .int32
        case .int64:   return .int64
        case .float:   return .float32
        case .double:  return .float64
        case .boolean: return .boolean
        case .string:  return .utf8
        case .date32:  return .date32
        default:
            throw SwiftMLError.unsupportedFormat("Unsupported Arrow type ID: \(arrowType.id)")
        }
    }
}
