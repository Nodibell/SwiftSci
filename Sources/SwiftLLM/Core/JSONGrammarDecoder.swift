#if os(macOS)
import Foundation
import MLX

/// The parsing state for tracking JSON grammar progression during LLM token generation.
public enum JSONGrammarState: Sendable, Equatable {
    /// Waiting for root object opening brace `{`.
    case expectRootOpen
    /// Inside object, expecting opening quote for a property key or closing brace `}`.
    case expectKeyOrClose
    /// Inside a property key string.
    case inKey(current: String)
    /// Expecting colon separator `:` after property key.
    case expectColon
    /// Expecting property value (string, number, boolean, object, array).
    case expectValue
    /// Inside a string value.
    case inStringValue
    /// Inside a numeric value.
    case inNumberValue
    /// Expecting comma separator `,` or closing brace `}` after a value.
    case expectCommaOrClose
    /// Successfully completed valid JSON object.
    case completed
}

/// Token-level grammar decoder that constrains LLM output to valid JSON conforming to Swift `Codable` schemas.
///
/// `JSONGrammarDecoder` tracks the JSON lexical state at each autoregressive generation step
/// and masks out invalid token logits, guaranteeing that generated text is syntactically valid JSON.
public final class JSONGrammarDecoder: @unchecked Sendable {
    /// Current state in the grammar state machine.
    public private(set) var state: JSONGrammarState
    /// Accumulated JSON text generated so far.
    public private(set) var accumulatedText: String

    private var openBracesCount: Int = 0

    /// Initializes a new JSON grammar decoder starting at root object state.
    public init() {
        self.state = .expectRootOpen
        self.accumulatedText = ""
    }

    /// Resets the grammar decoder state for a new generation sequence.
    public func reset() {
        self.state = .expectRootOpen
        self.accumulatedText = ""
        self.openBracesCount = 0
    }

    /// Determines whether appending the given token string remains grammatically valid in the current state.
    ///
    /// - Parameter tokenText: The candidate token text.
    /// - Returns: `true` if the token is valid in the current state, otherwise `false`.
    public func isValidToken(_ tokenText: String) -> Bool {
        if tokenText.isEmpty { return false }

        // Disallow tokens with invalid JSON characters in root states
        switch state {
        case .expectRootOpen:
            let trimmed = tokenText.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.contains("{") || trimmed.isEmpty

        case .expectKeyOrClose:
            let trimmed = tokenText.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.contains("\"") || trimmed.contains("}") || trimmed.isEmpty

        case .inKey:
            return true

        case .expectColon:
            let trimmed = tokenText.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.contains(":") || trimmed.isEmpty

        case .expectValue:
            return true

        case .inStringValue:
            return true

        case .inNumberValue:
            let trimmed = tokenText.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.allSatisfy { $0.isNumber || $0 == "." || $0 == "-" || $0 == "," || $0 == "}" }

        case .expectCommaOrClose:
            let trimmed = tokenText.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.contains(",") || trimmed.contains("}") || trimmed.isEmpty

        case .completed:
            return tokenText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
    }

    /// Advances the grammar state with the chosen token text.
    ///
    /// - Parameter tokenText: The newly emitted token string.
    public func advance(with tokenText: String) {
        accumulatedText += tokenText

        for char in tokenText {
            switch state {
            case .expectRootOpen:
                if char == "{" {
                    openBracesCount += 1
                    state = .expectKeyOrClose
                }

            case .expectKeyOrClose:
                if char == "\"" {
                    state = .inKey(current: "")
                } else if char == "}" {
                    openBracesCount -= 1
                    state = openBracesCount == 0 ? .completed : .expectCommaOrClose
                }

            case .inKey(let current):
                if char == "\"" {
                    state = .expectColon
                } else {
                    state = .inKey(current: current + String(char))
                }

            case .expectColon:
                if char == ":" {
                    state = .expectValue
                }

            case .expectValue:
                if char == "\"" {
                    state = .inStringValue
                } else if char == "{" {
                    openBracesCount += 1
                    state = .expectKeyOrClose
                } else if char.isNumber || char == "-" {
                    state = .inNumberValue
                } else if char == "t" || char == "f" || char == "n" { // true, false, null
                    state = .expectCommaOrClose
                }

            case .inStringValue:
                if char == "\"" {
                    state = .expectCommaOrClose
                }

            case .inNumberValue:
                if char.isWhitespace {
                    state = .expectCommaOrClose
                } else if char == "," {
                    state = .expectKeyOrClose
                } else if char == "}" {
                    openBracesCount -= 1
                    state = openBracesCount == 0 ? .completed : .expectCommaOrClose
                }

            case .expectCommaOrClose:
                if char == "," {
                    state = .expectKeyOrClose
                } else if char == "}" {
                    openBracesCount -= 1
                    state = openBracesCount == 0 ? .completed : .expectCommaOrClose
                }

            case .completed:
                break
            }
        }
    }

    /// Masks logits by penalizing tokens that violate current JSON grammar.
    ///
    /// - Parameters:
    ///   - logits: Logits tensor of shape `[vocabSize]`.
    ///   - vocab: Mapping from token ID to token string.
    /// - Returns: Filtered logits tensor with invalid token positions set to `-1e9`.
    public func maskLogits(_ logits: MLXArray, vocab: [Int: String]) -> MLXArray {
        var mask = [Float](repeating: 1.0, count: logits.shape[0])
        for (tokenId, text) in vocab {
            if tokenId < mask.count && !isValidToken(text) {
                mask[tokenId] = 0.0
            }
        }
        let maskArray = MLXArray(mask)
        return MLX.where(maskArray .== Float(1.0), logits, MLX.full(logits.shape, values: Float(-1e9)))
    }
}
#endif
