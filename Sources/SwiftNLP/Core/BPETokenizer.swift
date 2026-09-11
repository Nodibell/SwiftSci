import Foundation

/// A Byte Pair Encoding (BPE) subword tokenizer.
public struct BPETokenizer: Tokenizer, Sendable {
    /// The vocab.
    public let vocab: [String: Int]
    /// The merges.
    public let merges: [String: Int]
    /// The decoder.
    public let decoder: [Int: String]
    
    /// The unk token.
    public let unkToken = "<unk>"
    /// The unk token id.
    public let unkTokenId: Int
    
    /// Initializes the BPE tokenizer.
    /// - Parameters:
    ///   - vocab: A dictionary mapping subword tokens to IDs.
    ///   - merges: An array of merge rules in the format "first_token second_token".
    public init(vocab: [String: Int], merges: [String]) {
        self.vocab = vocab
        
        var mDict = [String: Int]()
        for (idx, merge) in merges.enumerated() {
            mDict[merge] = idx
        }
        self.merges = mDict
        
        self.decoder = vocab.reduce(into: [Int: String]()) { $0[$1.value] = $1.key }
        self.unkTokenId = vocab[unkToken] ?? 0
    }
    
    /// Tokenizes the given text into subword tokens.
    /// - Parameters:
    ///   - text: Input textual string to be analyzed or transformed.
    /// - Returns: Array of feature names, column identifiers, or tokens.
    public func tokenize(text: String) -> [String] {
        let words = text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        var result = [String]()
        
        for word in words {
            let tokens = bpe(word)
            result.append(contentsOf: tokens)
        }
        
        return result
    }
    
    /// Encodes the text into a sequence of token IDs.
    /// - Parameters:
    ///   - text: Input textual string to be analyzed or transformed.
    /// - Returns: Array of computed integer labels or indices.
    public func encode(text: String) -> [Int] {
        let tokens = tokenize(text: text)
        return tokens.map { vocab[$0] ?? unkTokenId }
    }
    
    /// Decodes a sequence of token IDs back into a reconstructed string with strict byte-level UTF-8 inversion.
    ///
    /// ## Grapheme & Multibyte Safety
    /// Reverses the GPT-2 byte encoder mapping back to raw bytes prior to UTF-8 decoding,
    /// ensuring intact reconstruction of Cyrillic, CJK characters, and compound emojis.
    ///
    /// - Parameter tokens: Sequence of integer token identifiers.
    /// - Returns: Reconstructed UTF-8 string with whitespace normalization.
    public func decode(tokens: [Int]) -> String {
        let subwords = tokens.compactMap { decoder[$0] }
        let joined = subwords.joined()
        
        var rawBytes = [UInt8]()
        rawBytes.reserveCapacity(joined.utf8.count)
        
        var i = joined.startIndex
        while i < joined.endIndex {
            // Check for </w> token
            if joined[i...].hasPrefix("</w>") {
                rawBytes.append(UInt8(ascii: " "))
                i = joined.index(i, offsetBy: 4)
            } else {
                let char = joined[i]
                if let byte = Self.byteDecoder[char] {
                    rawBytes.append(byte)
                } else {
                    for b in String(char).utf8 {
                        rawBytes.append(b)
                    }
                }
                i = joined.index(after: i)
            }
        }
        
        let decoded = String(decoding: rawBytes, as: UTF8.self)
        return decoded.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    // MARK: - Helper BPE Algorithm
    
    private static let byteEncoder: [UInt8: Character] = Self.makeByteEncoder()
    private static let byteDecoder: [Character: UInt8] = {
        var rev = [Character: UInt8]()
        for (b, c) in byteEncoder {
            rev[c] = b
        }
        return rev
    }()
    
    private static func makeByteEncoder() -> [UInt8: Character] {
        let bytes = Array(UInt8(ascii: "!")...UInt8(ascii: "~"))
            + Array(UInt8(0xA1)...UInt8(0xAC))
            + Array(UInt8(0xAE)...UInt8(0xFF))
        var mapping = [UInt8: Character]()
        var n: UInt32 = 0
        for b in UInt8.min...UInt8.max {
            if bytes.contains(b) {
                mapping[b] = Character(UnicodeScalar(UInt32(b))!)
            } else {
                mapping[b] = Character(UnicodeScalar(256 + n)!)
                n += 1
            }
        }
        return mapping
    }
    
    private func bpe(_ word: String) -> [String] {
        guard !word.isEmpty else { return [] }
        
        // Split word into UTF-8 bytes and encode via GPT-2 byteEncoder, appending </w> to the last byte
        var chars = word.utf8.map { String(Self.byteEncoder[$0]!) }
        chars[chars.count - 1] += "</w>"
        
        var currentWord = chars
        
        while currentWord.count > 1 {
            var pairs = [(String, String)]()
            for i in 0..<currentWord.count - 1 {
                pairs.append((currentWord[i], currentWord[i+1]))
            }
            
            var bestPair: (String, String)? = nil
            var bestRank = Int.max
            
            for pair in pairs {
                let mergeKey = "\(pair.0) \(pair.1)"
                if let rank = merges[mergeKey], rank < bestRank {
                    bestRank = rank
                    bestPair = pair
                }
            }
            
            guard let pairToMerge = bestPair else {
                break // No more merges available
            }
            
            currentWord = mergePair(currentWord, pair: pairToMerge)
        }
        
        return currentWord
    }
    
    private func mergePair(_ word: [String], pair: (String, String)) -> [String] {
        var newWord = [String]()
        var i = 0
        while i < word.count {
            if i < word.count - 1 && word[i] == pair.0 && word[i+1] == pair.1 {
                newWord.append(pair.0 + pair.1)
                i += 2
            } else {
                newWord.append(word[i])
                i += 1
            }
        }
        return newWord
    }
}
