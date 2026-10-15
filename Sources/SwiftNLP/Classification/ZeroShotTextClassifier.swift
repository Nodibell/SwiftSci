import Foundation
import Accelerate

/// The output of a zero-shot text classification inference.
public struct ZeroShotClassificationResult: Sendable, Codable, Equatable {
    /// The original input text evaluated.
    public let text: String

    /// The top predicted class label among candidate labels.
    public let predictedLabel: String

    /// Calibrated prediction probabilities (or sigmoid confidence scores) keyed by candidate label.
    public let scores: [String: Double]

    /// The list of candidate labels evaluated.
    public let candidateLabels: [String]

    /// Creates a new zero-shot classification result.
    public init(text: String, predictedLabel: String, scores: [String: Double], candidateLabels: [String]) {
        self.text = text
        self.predictedLabel = predictedLabel
        self.scores = scores
        self.candidateLabels = candidateLabels
    }
}

/// A zero-shot text classifier that categorizes text into arbitrary classes without task-specific training.
///
/// `ZeroShotTextClassifier` projects both input documents and class hypothesis templates
/// (e.g., *"This text is about {}."*) into a shared semantic embedding space using a
/// ``SentenceEmbeddingProvider``. Similarity is evaluated via cosine distance, followed
/// by temperature-scaled softmax normalization (or independent sigmoid scaling for multi-label tasks).
///
/// ## Example
/// ```swift
/// import SwiftNLP
///
/// let classifier = ZeroShotTextClassifier()
/// let result = try await classifier.classify(
///     text: "Apple unveiled the M4 Max chip with breakthrough memory bandwidth.",
///     candidateLabels: ["sports", "technology", "politics", "finance"]
/// )
/// print(result.predictedLabel) // "technology"
/// print(result.scores["technology"] ?? 0.0) // > 0.90
/// ```
public final class ZeroShotTextClassifier: Sendable {

    /// The sentence embedding engine used to map text and label hypotheses into vector space.
    public let embeddingProvider: any SentenceEmbeddingProvider

    /// The hypothesis template containing `{}` where candidate labels are substituted.
    public let hypothesisTemplate: String

    /// Temperature scaling factor for the softmax/sigmoid distribution (default 0.07).
    public let temperature: Double

    /// Creates a zero-shot text classifier with a custom embedding provider.
    ///
    /// - Parameters:
    ///   - embeddingProvider: An implementation of ``SentenceEmbeddingProvider``.
    ///   - hypothesisTemplate: Template for formulating label hypotheses (default `"This text is about {}."`).
    ///   - temperature: Softmax temperature parameter (default `0.07`).
    public init(
        embeddingProvider: any SentenceEmbeddingProvider,
        hypothesisTemplate: String = "This text is about {}.",
        temperature: Double = 0.07
    ) {
        self.embeddingProvider = embeddingProvider
        self.hypothesisTemplate = hypothesisTemplate
        self.temperature = max(1e-4, temperature)
    }

    /// Creates a zero-shot text classifier with default system sentence embeddings.
    ///
    /// Automatically prefers ``AppleSentenceEmbeddingEngine`` on Apple platforms, falling
    /// back to ``LocalEmbeddingEngine`` if native platform embeddings are unavailable.
    ///
    /// - Parameters:
    ///   - hypothesisTemplate: Template for formulating label hypotheses (default `"This text is about {}."`).
    ///   - temperature: Softmax temperature parameter (default `0.07`).
    public convenience init(
        hypothesisTemplate: String = "This text is about {}.",
        temperature: Double = 0.07
    ) {
        if let apple = try? AppleSentenceEmbeddingEngine() {
            self.init(embeddingProvider: apple, hypothesisTemplate: hypothesisTemplate, temperature: temperature)
        } else {
            self.init(embeddingProvider: LocalEmbeddingEngine(dimension: 128), hypothesisTemplate: hypothesisTemplate, temperature: temperature)
        }
    }

    /// Classifies a single input string against a set of candidate labels.
    ///
    /// - Parameters:
    ///   - text: The input document to classify.
    ///   - candidateLabels: An array of candidate label strings.
    ///   - multiLabel: If `true`, probabilities are computed independently using sigmoid.
    ///     If `false` (default), probabilities sum to 1.0 via softmax.
    /// - Throws: `NLPError.invalidInput` if candidate labels are empty, or other `NLPError` if embedding fails.
    /// - Returns: A ``ZeroShotClassificationResult`` containing the predicted label and class scores.
    public func classify(
        text: String,
        candidateLabels: [String],
        multiLabel: Bool = false
    ) async throws -> ZeroShotClassificationResult {
        let results = try await classifyBatch(texts: [text], candidateLabels: candidateLabels, multiLabel: multiLabel)
        guard let first = results.first else {
            throw NLPError.invalidInput("Classification produced no results.")
        }
        return first
    }

    /// Classifies a batch of input texts against candidate labels.
    ///
    /// Candidate label hypotheses are embedded once and reused across all input texts,
    /// significantly accelerating batch inference.
    ///
    /// - Parameters:
    ///   - texts: Array of input documents to classify.
    ///   - candidateLabels: An array of candidate label strings.
    ///   - multiLabel: If `true`, probabilities are computed independently using sigmoid.
    ///     If `false` (default), probabilities sum to 1.0 via softmax.
    /// - Throws: `NLPError.invalidInput` if candidate labels are empty, or other `NLPError` if embedding fails.
    /// - Returns: Array of ``ZeroShotClassificationResult`` instances in corresponding order.
    public func classifyBatch(
        texts: [String],
        candidateLabels: [String],
        multiLabel: Bool = false
    ) async throws -> [ZeroShotClassificationResult] {
        guard !candidateLabels.isEmpty else {
            throw NLPError.invalidInput("Candidate labels array cannot be empty.")
        }

        guard !texts.isEmpty else {
            return []
        }

        // 1. Embed candidate labels using hypothesis template once for the whole batch
        let hypotheses = candidateLabels.map { label in
            hypothesisTemplate.replacingOccurrences(of: "{}", with: label)
        }
        let labelVectors = try await embeddingProvider.embedBatch(hypotheses)

        // 2. Embed all input texts
        let textVectors = try await embeddingProvider.embedBatch(texts)

        // 3. Score each text against all candidate labels
        var results: [ZeroShotClassificationResult] = []
        results.reserveCapacity(texts.count)

        for (textIndex, text) in texts.enumerated() {
            let textTrimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if textTrimmed.isEmpty {
                let uniform = 1.0 / Double(candidateLabels.count)
                var scores: [String: Double] = [:]
                for label in candidateLabels {
                    scores[label] = uniform
                }
                results.append(ZeroShotClassificationResult(
                    text: text,
                    predictedLabel: candidateLabels[0],
                    scores: scores,
                    candidateLabels: candidateLabels
                ))
                continue
            }

            let textVec = textVectors[textIndex]
            var similarities: [Double] = []
            similarities.reserveCapacity(candidateLabels.count)

            for labelVec in labelVectors {
                let sim = cosineSimilarity(textVec, labelVec)
                similarities.append(sim)
            }

            let scoresArray: [Double]
            if multiLabel {
                // Independent sigmoid for each label
                scoresArray = similarities.map { sim in
                    let logit = sim / temperature
                    return 1.0 / (1.0 + exp(-logit))
                }
            } else {
                // Softmax over all candidate labels
                let logits = similarities.map { $0 / temperature }
                let maxLogit = logits.max() ?? 0.0
                let expScores = logits.map { exp($0 - maxLogit) }
                let sumExp = expScores.reduce(0.0, +)
                if sumExp > 0.0 {
                    scoresArray = expScores.map { $0 / sumExp }
                } else {
                    let uniform = 1.0 / Double(candidateLabels.count)
                    scoresArray = [Double](repeating: uniform, count: candidateLabels.count)
                }
            }

            var bestIndex = 0
            var bestScore = -Double.infinity
            var scoresDict: [String: Double] = [:]

            for (idx, label) in candidateLabels.enumerated() {
                let s = scoresArray[idx]
                scoresDict[label] = s
                if s > bestScore {
                    bestScore = s
                    bestIndex = idx
                }
            }

            results.append(ZeroShotClassificationResult(
                text: text,
                predictedLabel: candidateLabels[bestIndex],
                scores: scoresDict,
                candidateLabels: candidateLabels
            ))
        }

        return results
    }

    // MARK: - Private SIMD Helpers

    @inline(__always)
    private func cosineSimilarity(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0.0 }
        let n = vDSP_Length(a.count)

        var dot: Double = 0.0
        vDSP_dotprD(a, 1, b, 1, &dot, n)

        var sumSqA: Double = 0.0
        var sumSqB: Double = 0.0
        vDSP_svesqD(a, 1, &sumSqA, n)
        vDSP_svesqD(b, 1, &sumSqB, n)

        let normProd = sqrt(sumSqA * sumSqB)
        if normProd > 1e-12 {
            let sim = dot / normProd
            return max(-1.0, min(1.0, sim))
        }
        return 0.0
    }
}
