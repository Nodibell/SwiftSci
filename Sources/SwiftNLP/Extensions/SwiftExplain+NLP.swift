import Foundation

/// Provides token-level importance explanation for text predictions.
public struct TextExplainer: Sendable {
    /// Creates a text token explainer instance.
    public init() {}

    /// Explains token importance by measuring sentiment score shift when each token is omitted.
    public func explainSentimentTokens(in text: String) -> [(token: String, impact: Double)] {
        let analyzer = VADERSentimentAnalyzer()
        let baseScore = analyzer.polarityScores(text: text).compound
        let tokens = text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }

        var importance: [(token: String, impact: Double)] = []

        for (i, targetToken) in tokens.enumerated() {
            var ablatedTokens = tokens
            ablatedTokens.remove(at: i)
            let ablatedText = ablatedTokens.joined(separator: " ")
            let ablatedScore = analyzer.polarityScores(text: ablatedText).compound
            let impact = baseScore - ablatedScore
            importance.append((token: targetToken, impact: impact))
        }

        return importance.sorted { abs($0.impact) > abs($1.impact) }
    }
}
