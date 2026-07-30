import Foundation

/// Pure Swift VADER Sentiment Analyzer implementing valence score calculation with negation and intensity boosters.
public struct VADERSentimentAnalyzer: Sendable {
    public init() {}

    /// Evaluates sentiment of the input text and returns a `SentimentScore`.
    public func polarityScores(text: String) -> SentimentScore {
        let tokens = text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        if tokens.isEmpty {
            return SentimentScore(neg: 0.0, neu: 1.0, pos: 0.0, compound: 0.0)
        }

        var valences: [Double] = []

        for (i, token) in tokens.enumerated() {
            let cleanToken = token.trimmingCharacters(in: .punctuationCharacters)
            if cleanToken.isEmpty { continue }

            if var val = VADERLexicon.valence(for: cleanToken) {
                // ALL CAPS booster
                let isAllCaps = cleanToken.count > 1 && cleanToken == cleanToken.uppercased()
                if isAllCaps {
                    val += (val > 0) ? 0.733 : -0.733
                }

                // Check preceding tokens for negations and boosters
                var scalar = 1.0
                for distance in 1...3 {
                    if i - distance >= 0 {
                        let prevToken = tokens[i - distance].lowercased().trimmingCharacters(in: .punctuationCharacters)
                        if VADERLexicon.negationWords.contains(prevToken) {
                            scalar *= -0.74
                        } else if let b = VADERLexicon.boosterDict[prevToken] {
                            scalar += (val > 0) ? b : -b
                        }
                    }
                }
                valences.append(val * scalar)
            }
        }

        if valences.isEmpty {
            return SentimentScore(neg: 0.0, neu: 1.0, pos: 0.0, compound: 0.0)
        }

        // Punctuation booster (exclamation marks)
        let exclamations = Double(text.filter { $0 == "!" }.count)
        let punctuationScalar = min(exclamations * 0.293, 0.96)

        var sumValence = valences.reduce(0.0, +)
        if sumValence > 0 {
            sumValence += punctuationScalar
        } else if sumValence < 0 {
            sumValence -= punctuationScalar
        }

        // Normalize compound score c = sum / sqrt(sum^2 + 15)
        let compound = sumValence / sqrt((sumValence * sumValence) + 15.0)

        // Compute pos, neg, neu ratios
        var posSum = 0.0
        var negSum = 0.0
        var neuCount = 0

        for v in valences {
            if v > 0 {
                posSum += (v + 1.0)
            } else if v < 0 {
                negSum += (abs(v) + 1.0)
            } else {
                neuCount += 1
            }
        }

        let total = posSum + negSum + Double(neuCount)
        let pos = total > 0 ? posSum / total : 0.0
        let neg = total > 0 ? negSum / total : 0.0
        let neu = total > 0 ? Double(neuCount) / total : 1.0

        return SentimentScore(
            neg: min(1.0, max(0.0, neg)),
            neu: min(1.0, max(0.0, neu)),
            pos: min(1.0, max(0.0, pos)),
            compound: min(1.0, max(-1.0, compound))
        )
    }
}
