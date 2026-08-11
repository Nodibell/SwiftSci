import Foundation
import SwiftDataFrame

extension DataFrame {
    /// Vectorizes text documents in a DataFrame column using TF-IDF for clustering.
    /// - Parameters:
    ///   - name: Input text column name.
    /// - Returns: 2D array of TF-IDF feature vectors [numDocuments, numFeatures].
    public func vectorizeTextColumn(column name: String) async throws -> (matrix: [[Double]], vocabulary: [String: Int]) {
        guard let col = self[column: name, as: String.self] else {
            throw SwiftMLError.columnNotFound(name)
        }
        let documents = col.values.map { $0 ?? "" }
        let vectorizer = TFIDFVectorizer()
        try await vectorizer.fit(documents)
        let matrix = try await vectorizer.transform(documents)
        let vocab = await vectorizer.vocabulary
        return (matrix, vocab)
    }
}
