import Foundation

/// Generates n-gram token sequences from input string texts.
public struct NGramTokenizer: Sendable {
    public let minN: Int
    public let maxN: Int
    public let lowercase: Bool

    public init(minN: Int = 1, maxN: Int = 2, lowercase: Bool = true) {
        self.minN = max(1, minN)
        self.maxN = max(self.minN, maxN)
        self.lowercase = lowercase
    }

    /// Tokenizes input text into n-grams.
    public func tokenize(_ text: String) -> [String] {
        let prepared = lowercase ? text.lowercased() : text
        let unigrams = prepared.components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }

        guard !unigrams.isEmpty else { return [] }
        var result: [String] = []

        for n in minN...maxN {
            if n == 1 {
                result.append(contentsOf: unigrams)
            } else if unigrams.count >= n {
                for i in 0...(unigrams.count - n) {
                    let ngram = unigrams[i..<(i + n)].joined(separator: " ")
                    result.append(ngram)
                }
            }
        }

        return result
    }
}
