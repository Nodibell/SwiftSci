import Foundation

/// HashingVectorizer converts text documents directly into fixed-size numeric feature vectors
/// using the hashing trick (FNV-1a 64-bit hash modulo nFeatures).
public final class HashingVectorizer: @unchecked Sendable {
    public let nFeatures: Int
    public let ngramRange: (min: Int, max: Int)
    public let lowercase: Bool

    public init(nFeatures: Int = 1024, ngramRange: (min: Int, max: Int) = (1, 1), lowercase: Bool = true) {
        self.nFeatures = max(1, nFeatures)
        self.ngramRange = (max(1, ngramRange.min), max(ngramRange.min, ngramRange.max))
        self.lowercase = lowercase
    }

    /// Transforms documents into fixed-width hashed feature vectors.
    public func transform(documents: [String]) -> [[Double]] {
        let tokenizer = NGramTokenizer(minN: ngramRange.min, maxN: ngramRange.max, lowercase: lowercase)
        return documents.map { doc in
            var row = [Double](repeating: 0.0, count: nFeatures)
            let tokens = tokenizer.tokenize(doc)
            for token in tokens {
                let hash = fnv1a64(token)
                let idx = Int(hash % UInt64(nFeatures))
                row[idx] += 1.0
            }
            return row
        }
    }

    private func fnv1a64(_ string: String) -> UInt64 {
        var hash: UInt64 = 14695981039346656037
        for byte in string.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 1099511628211
        }
        return hash
    }
}
