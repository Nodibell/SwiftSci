import Foundation
import Arrow

/// Reader for Feather / Apache Arrow IPC binary file format.
public enum FeatherReader {

    /// Reads a Feather (.feather / .arrow) binary file into a `DataFrame`.
    /// - Parameter url: The file URL to read.
    /// - Returns: The loaded `DataFrame`.
    public static func read(url: URL) async throws -> DataFrame {
        let arrowReader = ArrowReader()
        let result = arrowReader.fromFile(url)
        
        switch result {
        case .success(let readerResult):
            guard !readerResult.batches.isEmpty else {
                return DataFrame.empty
            }
            switch ArrowTable.from(recordBatches: readerResult.batches) {
            case .success(let table):
                return try ArrowTableBridge.toDataFrame(table)
            case .failure(let err):
                throw SwiftMLError.unsupportedFormat("Failed to load ArrowTable from Feather batches: \(err)")
            }
        case .failure(let err):
            throw SwiftMLError.unsupportedFormat("Failed to read Feather file at \(url.path): \(err)")
        }
    }

    /// Reads Feather binary data into a `DataFrame`.
    /// - Parameter data: Raw Data bytes of Feather file.
    /// - Returns: The loaded `DataFrame`.
    public static func read(data: Data) async throws -> DataFrame {
        let arrowReader = ArrowReader()
        let result = arrowReader.readFile(data)
        
        switch result {
        case .success(let readerResult):
            guard !readerResult.batches.isEmpty else {
                return DataFrame.empty
            }
            switch ArrowTable.from(recordBatches: readerResult.batches) {
            case .success(let table):
                return try ArrowTableBridge.toDataFrame(table)
            case .failure(let err):
                throw SwiftMLError.unsupportedFormat("Failed to load ArrowTable from Feather batches: \(err)")
            }
        case .failure(let err):
            throw SwiftMLError.unsupportedFormat("Failed to read Feather data: \(err)")
        }
    }
}
