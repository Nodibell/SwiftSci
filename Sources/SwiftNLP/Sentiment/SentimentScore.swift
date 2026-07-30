import Foundation

/// Sentiment score breakdown containing positive, negative, neutral, and normalized compound metrics.
public struct SentimentScore: Sendable, Equatable, CustomStringConvertible {
    /// Negative sentiment score (0.0 to 1.0)
    public let neg: Double
    /// Neutral sentiment score (0.0 to 1.0)
    public let neu: Double
    /// Positive sentiment score (0.0 to 1.0)
    public let pos: Double
    /// Normalized compound score (-1.0 to +1.0)
    public let compound: Double

    public init(neg: Double, neu: Double, pos: Double, compound: Double) {
        self.neg = neg
        self.neu = neu
        self.pos = pos
        self.compound = compound
    }

    public var description: String {
        return "SentimentScore(compound: \(String(format: "%.4f", compound)), pos: \(String(format: "%.4f", pos)), neu: \(String(format: "%.4f", neu)), neg: \(String(format: "%.4f", neg)))"
    }
}
