import Foundation

/// Pure Swift VADER Sentiment Analyzer implementing valence score calculation with negation, intensity boosters,
/// the "but" contrast clause rule, punctuation amplification, and ALL CAPS differential weighting.
public struct VADERSentimentAnalyzer: Sendable {
    /// Creates a VADER rule-based lexicon sentiment analyzer instance.
    public init() {}

    /// Evaluates sentiment of the input text and returns a `SentimentScore`.
    /// - Parameters:
    ///   - text: Input textual string to be analyzed or transformed.
    /// - Returns: The computed SentimentScore result instance.
    public func polarityScores(text: String) -> SentimentScore {
        let tokens = text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        if tokens.isEmpty {
            return SentimentScore(neg: 0.0, neu: 1.0, pos: 0.0, compound: 0.0)
        }

        // Determine if there is ALL CAPS contrast differential in the text
        var hasLower = false
        var hasUpper = false
        for char in text {
            if char.isLowercase { hasLower = true }
            if char.isUppercase { hasUpper = true }
        }
        let isCapDiff = hasLower && hasUpper

        // Find index of "but" clause if present
        let butIndex: Int? = tokens.firstIndex { token in
            token.lowercased().trimmingCharacters(in: .punctuationCharacters) == "but"
        }

        // Initialize valences for ALL tokens (0.0 for non-sentiment/neutral tokens)
        var valences: [Double] = [Double](repeating: 0.0, count: tokens.count)

        for (i, token) in tokens.enumerated() {
            // First check if the token as-is is an emoticon (e.g. ":)", ":-D")
            var tokenValence = VADERLexicon.valence(for: token)
            let cleanToken = token.trimmingCharacters(in: .punctuationCharacters)

            if tokenValence == nil && !cleanToken.isEmpty {
                tokenValence = VADERLexicon.valence(for: cleanToken)
            }

            guard var val = tokenValence, val != 0.0 else {
                continue
            }

            // ALL CAPS booster (only active if text has mixed casing contrast)
            let isAllCaps = isCapDiff && cleanToken.count > 1 && cleanToken == cleanToken.uppercased()
            if isAllCaps {
                val += (val > 0) ? 0.733 : -0.733
            }

            // Check preceding tokens (up to 3 positions back) for negations and booster words
            var scalar = 1.0
            for distance in 1...3 {
                if i - distance >= 0 {
                    let prevToken = tokens[i - distance].lowercased().trimmingCharacters(in: .punctuationCharacters)
                    if VADERLexicon.negationWords.contains(prevToken) {
                        scalar *= -0.74
                    } else if let b = VADERLexicon.boosterDict[prevToken] {
                        let boosterScalar = isAllCaps ? (b + (b > 0 ? 0.293 : -0.293)) : b
                        scalar += (val > 0) ? boosterScalar : -boosterScalar
                    }
                }
            }

            valences[i] = val * scalar
        }

        // Apply "but" contrast rule: tokens before "but" scaled by 0.5, tokens after scaled by 1.5
        if let bIdx = butIndex {
            for i in 0..<valences.count {
                if valences[i] != 0.0 {
                    if i < bIdx {
                        valences[i] *= 0.5
                    } else if i > bIdx {
                        valences[i] *= 1.5
                    }
                }
            }
        }

        let nonZeroCount = valences.filter { $0 != 0.0 }.count
        if nonZeroCount == 0 {
            return SentimentScore(neg: 0.0, neu: 1.0, pos: 0.0, compound: 0.0)
        }

        // Punctuation booster: exclamation marks '!' and multiple question marks '?'
        let exclamations = Double(text.filter { $0 == "!" }.count)
        let epScalar = min(exclamations * 0.292, 0.96)

        let questions = Double(text.filter { $0 == "?" }.count)
        let qmScalar = questions > 1 ? min(questions * 0.18, 0.96) : 0.0
        let punctuationScalar = min(epScalar + qmScalar, 0.96)

        var sumValence = valences.reduce(0.0, +)
        if sumValence > 0 {
            sumValence += punctuationScalar
        } else if sumValence < 0 {
            sumValence -= punctuationScalar
        }

        // Normalize compound score c = sum / sqrt(sum^2 + 15)
        let compound = sumValence / sqrt((sumValence * sumValence) + 15.0)

        // Compute pos, neg, neu ratios across ALL tokens in sentence
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
