import Foundation

/// Provides static pre-sorted array structures holding VADER sentiment valence scores.
/// Using pre-sorted parallel arrays + binary search avoids Swift compiler dictionary literal type-checker overhead.
public enum VADERLexicon {
    // Pre-sorted array of lowercased VADER sentiment words
    private static let sortedWords: [String] = [
        "abandon", "abandoned", "abhor", "abhorrent", "abnormal", "abominable", "abuse", "abused",
        "accept", "accepted", "acclaim", "acclaimed", "accomplish", "accomplished", "admire", "admired",
        "adore", "adored", "affection", "affinity", "agony", "agreeable", "alarm", "alarmed",
        "amazing", "anger", "angry", "annoy", "annoyed", "appalled", "awesome", "awful",
        "bad", "beautiful", "best", "better", "bitter", "bless", "blessed", "bliss",
        "brilliant", "catastrophe", "charm", "charming", "cheer", "cheerful", "comfort", "comfortable",
        "crap", "damage", "damaged", "danger", "dangerous", "dead", "decent", "defeat",
        "delight", "delighted", "depressed", "depression", "despise", "despised", "destroy", "destroyed",
        "devastated", "disaster", "disappointed", "disgust", "disgusted", "dreadful", "enjoy", "enjoyable",
        "excellent", "excited", "exciting", "fail", "failed", "failure", "fantastic", "favor",
        "favorite", "fear", "fearful", "fine", "furious", "glad", "good", "gorgeous",
        "great", "grief", "gross", "happy", "harm", "harmful", "hate", "hated",
        "hateful", "heaven", "help", "helpful", "helpless", "horrible", "horrific", "horrifying",
        "hurt", "ideal", "ill", "impressive", "inspired", "joy", "joyful", "like",
        "liked", "love", "loved", "lovely", "mad", "mess", "nasty", "nice",
        "pain", "painful", "perfect", "pleased", "pleasure", "poor", "rubbish", "sad",
        "sadness", "sick", "smile", "splendid", "stink", "stupid", "sublime", "superb",
        "sweet", "terrible", "terrific", "thrilled", "tragic", "tragedy", "treasure", "triumph",
        "ugly", "upset", "useful", "useless", "vile", "victory", "wonderful", "worst",
        "worthless", "worthy", "wow"
    ]

    private static let valences: [Double] = [
        -1.9, -1.7, -2.9, -3.1, -1.3, -3.0, -2.1, -2.3,
        1.6, 1.8, 2.3, 2.6, 1.8, 2.2, 2.3, 2.5,
        2.9, 3.0, 2.4, 2.2, -2.8, 1.9, -1.4, -1.8,
        2.8, -2.3, -2.3, -1.8, -1.9, -2.3, 3.1, -2.7,
        -2.5, 2.9, 3.2, 1.9, -2.2, 1.8, 2.4, 2.8,
        2.8, -3.0, 1.9, 2.8, 2.3, 2.5, 1.5, 2.3,
        -2.3, -1.9, -2.0, -1.5, -2.1, -2.9, 1.9, -2.0,
        2.9, 2.8, -2.2, -2.6, -2.6, -2.8, -2.5, -2.6,
        -3.0, -3.1, -2.6, -2.6, -2.8, -2.7, 2.2, 2.5,
        3.2, 2.3, 2.6, -2.3, -2.3, -2.3, 2.8, 1.7,
        2.4, -2.2, -2.2, 1.4, -2.7, 1.9, 1.9, 3.0,
        3.1, -2.4, -2.1, 2.7, -1.9, -2.1, -2.7, -2.6,
        -2.9, 2.3, 1.7, 1.8, -2.1, -2.5, -2.7, -2.8,
        -2.4, 2.4, -1.7, 2.1, 2.3, 2.9, 2.9, 1.6,
        1.8, 3.2, 3.1, 2.8, -2.2, -1.5, -2.6, 1.9,
        -2.3, -2.4, 3.0, 2.2, 2.5, -2.0, -2.1, -2.1,
        -2.2, -1.7, 2.1, 2.7, -2.0, -2.4, 2.7, 2.8,
        2.3, -3.0, 2.6, 2.4, -2.5, -2.6, 2.7, 2.4,
        -2.0, -2.4, 1.8, -2.1, -2.8, 2.7, 3.0, -3.1,
        -2.6, 1.9, 2.8
    ]

    /// Booster words that increase or decrease valence.
    public static let boosterDict: [String: Double] = [
        "absolutely": 0.293, "incredibly": 0.293, "extremely": 0.293, "completely": 0.293,
        "very": 0.293, "so": 0.293, "totally": 0.293, "really": 0.293, "ultra": 0.293,
        "slightly": -0.293, "somewhat": -0.293, "barely": -0.293, "hardly": -0.293
    ]

    /// Negation words that flip polarity.
    public static let negationWords: Set<String> = [
        "not", "n't", "never", "no", "neither", "nor", "hardly", "scarcely", "barely"
    ]

    /// Performs binary search lookup for a word's valence score.
    public static func valence(for word: String) -> Double? {
        let target = word.lowercased()
        var low = 0
        var high = sortedWords.count - 1

        while low <= high {
            let mid = (low + high) / 2
            let current = sortedWords[mid]
            if current == target {
                return valences[mid]
            } else if current < target {
                low = mid + 1
            } else {
                high = mid - 1
            }
        }
        return nil
    }
}
