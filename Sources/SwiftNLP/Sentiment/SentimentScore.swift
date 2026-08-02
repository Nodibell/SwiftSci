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

    /// Creates a sentiment score breakdown object.
    /// - Parameters:
    ///   - neg: Negative proportion score (0.0 to 1.0).
    ///   - neu: Neutral proportion score (0.0 to 1.0).
    ///   - pos: Positive proportion score (0.0 to 1.0).
    ///   - compound: Normalized compound score (-1.0 to +1.0).
    public init(neg: Double, neu: Double, pos: Double, compound: Double) {
        self.neg = neg
        self.neu = neu
        self.pos = pos
        self.compound = compound
    }

    /// Human-readable textual representation string.
    public var description: String {
        return "SentimentScore(compound: \(String(format: "%.4f", compound)), pos: \(String(format: "%.4f", pos)), neu: \(String(format: "%.4f", neu)), neg: \(String(format: "%.4f", neg)))"
    }
}
