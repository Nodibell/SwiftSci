import Foundation
import SwiftDataFrame

extension DataFrame {
    private func extractDocuments(column name: String) throws -> [String] {
        guard let col = self[column: name, as: String.self] else {
            throw DataFrameError.columnNotFound(name)
        }
        return col.values.map { $0 ?? "" }
    }

    /// Fits a TFIDFVectorizer on the specified text column.
    public func fitTFIDF(column name: String) async throws -> TFIDFVectorizer {
        let documents = try extractDocuments(column: name)
        let vectorizer = TFIDFVectorizer()
        try await vectorizer.fit(documents)
        return vectorizer
    }
}
