import Foundation
import Arrow

/// Writer for Feather / Apache Arrow IPC binary file format.
public enum FeatherWriter {

    /// Writes a `DataFrame` into a Feather (.feather / .arrow) binary file.
    /// - Parameters:
    ///   - df: The `DataFrame` to write.
    ///   - url: Target file URL.
    public static func write(_ df: DataFrame, to url: URL) async throws {
        let rb = try ArrowTableBridge.toRecordBatch(df)
        let info = ArrowWriter.Info(.schema, schema: rb.schema, batches: [rb])
        let writer = ArrowWriter()
        
        switch writer.toFile(url, info: info) {
        case .success:
            return
        case .failure(let err):
            throw SwiftMLError.unsupportedFormat("Failed to write Feather file to \(url.path): \(err)")
        }
    }

    /// Serializes a `DataFrame` into raw Feather binary `Data`.
    /// - Parameter df: The `DataFrame` to serialize.
    /// - Returns: Serialized Feather `Data`.
    public static func write(_ df: DataFrame) throws -> Data {
        let rb = try ArrowTableBridge.toRecordBatch(df)
        let info = ArrowWriter.Info(.schema, schema: rb.schema, batches: [rb])
        let writer = ArrowWriter()
        
        switch writer.writeFile(info) {
        case .success(let data):
            return data
        case .failure(let err):
            throw SwiftMLError.unsupportedFormat("Failed to write Feather data: \(err)")
        }
    }

}
