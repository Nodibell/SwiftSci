import Foundation

// MARK: - Lexical Profile Result

/// Quantitative lexical richness and diversity metrics for a natural language text corpus.
public struct LexicalProfile: Sendable, Codable, Equatable {
    /// Total number of tokens (words) analyzed across the corpus.
    public let totalTokens: Int
    /// Number of distinct (unique) lexical types / vocabulary words.
    public let uniqueTokens: Int
    /// Type-Token Ratio (TTR = uniqueTokens / totalTokens), measuring basic lexical diversity (0.0 to 1.0).
    public let typeTokenRatio: Double
    /// Ratio of hapax legomena (words occurring exactly once) relative to unique vocabulary size.
    public let hapaxLegomenaRatio: Double
    /// Shannon entropy of the token frequency distribution in bits ($-\sum p_i \log_2 p_i$).
    public let shannonEntropy: Double
    /// Average character length of tokens across the corpus.
    public let averageTokenLength: Double
    
    /// Initializes a lexical profile result.
    /// - Parameters:
    ///   - totalTokens: Total token count.
    ///   - uniqueTokens: Distinct vocabulary size.
    ///   - typeTokenRatio: Type-Token Ratio.
    ///   - hapaxLegomenaRatio: Fraction of words appearing only once.
    ///   - shannonEntropy: Information entropy in bits.
    ///   - averageTokenLength: Mean character count per token.
    public init(
        totalTokens: Int,
        uniqueTokens: Int,
        typeTokenRatio: Double,
        hapaxLegomenaRatio: Double,
        shannonEntropy: Double,
        averageTokenLength: Double
    ) {
        self.totalTokens = totalTokens
        self.uniqueTokens = uniqueTokens
        self.typeTokenRatio = typeTokenRatio
        self.hapaxLegomenaRatio = hapaxLegomenaRatio
        self.shannonEntropy = shannonEntropy
        self.averageTokenLength = averageTokenLength
    }
}

// MARK: - Corpus Lexical Profiler

/// Analytical profiler computing vocabulary breadth, lexical diversity (TTR), hapax legomena, and Shannon entropy.
public enum CorpusLexicalProfiler {
    /// Analyzes an array of pre-tokenized words.
    ///
    /// - Parameter tokens: Array of lowercased or raw token strings.
    /// - Returns: A `LexicalProfile` containing comprehensive lexical metrics.
    public static func profile(tokens: [String]) -> LexicalProfile {
        guard !tokens.isEmpty else {
            return LexicalProfile(
                totalTokens: 0,
                uniqueTokens: 0,
                typeTokenRatio: 0.0,
                hapaxLegomenaRatio: 0.0,
                shannonEntropy: 0.0,
                averageTokenLength: 0.0
            )
        }
        
        var frequencies: [String: Int] = [:]
        frequencies.reserveCapacity(tokens.count)
        var totalChars = 0
        
        for token in tokens {
            frequencies[token, default: 0] += 1
            totalChars += token.count
        }
        
        let total = tokens.count
        let unique = frequencies.count
        let ttr = Double(unique) / Double(total)
        
        var hapaxCount = 0
        var entropy = 0.0
        let log2Constant = log(2.0)
        let totalDouble = Double(total)
        
        for (_, count) in frequencies {
            if count == 1 {
                hapaxCount += 1
            }
            let p = Double(count) / totalDouble
            entropy -= p * (log(p) / log2Constant)
        }
        
        let hapaxRatio = unique > 0 ? (Double(hapaxCount) / Double(unique)) : 0.0
        let avgLength = Double(totalChars) / totalDouble
        
        return LexicalProfile(
            totalTokens: total,
            uniqueTokens: unique,
            typeTokenRatio: ttr,
            hapaxLegomenaRatio: hapaxRatio,
            shannonEntropy: entropy,
            averageTokenLength: avgLength
        )
    }
    
    /// Analyzes an array of raw text documents or sentences.
    ///
    /// - Parameters:
    ///   - texts: Array of text document strings.
    ///   - tokenizer: The tokenizer implementation to split text into words (defaults to `AppleWordTokenizer`).
    /// - Returns: A `LexicalProfile` containing comprehensive lexical metrics.
    public static func profile(
        texts: [String],
        tokenizer: Tokenizer = AppleWordTokenizer()
    ) -> LexicalProfile {
        var allTokens: [String] = []
        for text in texts {
            let tokens = tokenizer.tokenize(text: text)
            allTokens.append(contentsOf: tokens)
        }
        return profile(tokens: allTokens)
    }
}
