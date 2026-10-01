import Foundation

/// The bound that ended generation before a stop token was reached.
public enum GenerationLimit: Sendable, Equatable {
    /// The caller's maximum number of generated tokens.
    case maxTokens
    /// The configured capacity for prompt and generated tokens together.
    case contextWindow
}

/// The reason a generation operation ended.
public enum GenerationStopReason: Sendable, Equatable {
    /// A configured EOS token was encountered, or, with no EOS IDs configured,
    /// the legacy tokenizer produced empty text or the literal `<unk>`.
    case stop
    /// Generation reached the indicated bound. When both bounds coincide, `maxTokens` takes precedence.
    case length(GenerationLimit)
    /// The generation task was cancelled. A cancelled consumer may not receive this event.
    case cancelled
}

/// Final metadata for one generation operation.
public struct GenerationCompletionInfo: Sendable, Equatable {
    /// Tokens supplied to the model, including the fallback token used for an empty prompt.
    public let promptTokenCount: Int
    /// Generated non-stop tokens, including tokens buffered during UTF-8 decoding.
    /// EOS tokens are excluded. Empty/unknown tokens are excluded only when they
    /// trigger legacy stopping with no EOS IDs configured. Text chunks are not token counts.
    public let generationTokenCount: Int
    /// Why generation ended.
    public let stopReason: GenerationStopReason
}

/// A text fragment or the final completion record from a generation operation.
public enum GenerationEvent: Sendable, Equatable {
    /// Decoded text from one or more tokens.
    case chunk(String)
    /// Final metadata, emitted after any buffered text has been flushed.
    case info(GenerationCompletionInfo)

    /// Text for consumers that do not need completion metadata.
    public var chunk: String? {
        if case .chunk(let text) = self { return text }
        return nil
    }
}

/// An invalid generation request. Requests never silently discard prompt tokens.
public enum GenerationError: Error, Sendable, Equatable, LocalizedError {
    /// Output budgets must be nonnegative.
    case invalidMaxTokens(Int)
    /// Context capacity must be positive.
    case invalidContextLimit(Int)
    /// The tokenized prompt exceeds the configured capacity.
    case promptTooLong(promptTokens: Int, capacity: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidMaxTokens(let value):
            "Maximum generated tokens must be nonnegative; received \(value)."
        case .invalidContextLimit(let value):
            "Context capacity must be positive; received \(value)."
        case .promptTooLong(let count, let capacity):
            "Prompt contains \(count) tokens but the configured context capacity is \(capacity)."
        }
    }
}
