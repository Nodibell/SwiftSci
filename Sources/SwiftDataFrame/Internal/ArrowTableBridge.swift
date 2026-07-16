import Foundation
import Arrow

/// A bridging layer to convert between `ArrowTable` and `DataFrame`.
internal enum ArrowTableBridge {

    /// Converts an Apache Arrow `ArrowTable` into a `DataFrame`.
    static func toDataFrame(_ arrowTable: ArrowTable) throws -> DataFrame {
        var columns: [any AnyColumn] = []
        
        for arrowCol in arrowTable.columns {
            let name = arrowCol.name
            let dtype = try mapArrowType(arrowCol.type)
            let count = Int(arrowCol.length)
            
            switch dtype {
            case .int32:
                let chunked: ChunkedArray<Int32> = arrowCol.data()
                var vals = [Int32?]()
                vals.reserveCapacity(count)
                for i in 0..<UInt(count) {
                    vals.append(chunked[i])
                }
                columns.append(TypedColumn<Int32>(name: name, values: vals))
                
            case .int64:
                let chunked: ChunkedArray<Int64> = arrowCol.data()
                var vals = [Int64?]()
                vals.reserveCapacity(count)
                for i in 0..<UInt(count) {
                    vals.append(chunked[i])
                }
                columns.append(TypedColumn<Int64>(name: name, values: vals))
                
            case .float32:
                let chunked: ChunkedArray<Float> = arrowCol.data()
                var vals = [Float?]()
                vals.reserveCapacity(count)
                for i in 0..<UInt(count) {
                    vals.append(chunked[i])
                }
                columns.append(TypedColumn<Float>(name: name, values: vals))
                
            case .float64:
                let chunked: ChunkedArray<Double> = arrowCol.data()
                var vals = [Double?]()
                vals.reserveCapacity(count)
                for i in 0..<UInt(count) {
                    vals.append(chunked[i])
                }
                columns.append(TypedColumn<Double>(name: name, values: vals))
                
            case .boolean:
                let chunked: ChunkedArray<Bool> = arrowCol.data()
                var vals = [Bool?]()
                vals.reserveCapacity(count)
                for i in 0..<UInt(count) {
                    vals.append(chunked[i])
                }
                columns.append(TypedColumn<Bool>(name: name, values: vals))
                
            case .utf8:
                let chunked: ChunkedArray<String> = arrowCol.data()
                var vals = [String?]()
                vals.reserveCapacity(count)
                for i in 0..<UInt(count) {
                    vals.append(chunked[i])
                }
                columns.append(TypedColumn<String>(name: name, values: vals))
                
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

    /// Converts a `DataFrame` into an Apache Arrow `ArrowTable`.
    static func toArrowTable(_ df: DataFrame) throws -> ArrowTable {
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
            switch ArrowTable.from(recordBatches: [rb]) {
            case .success(let table):
                return table
            case .failure(let err):
                throw DataFrameError.unsupportedFormat("Failed to build ArrowTable: \(err)")
            }
        case .failure(let err):
            throw DataFrameError.unsupportedFormat("Failed to build RecordBatch: \(err)")
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
            throw DataFrameError.unsupportedFormat("Unsupported Arrow type ID: \(arrowType.id)")
        }
    }
}
