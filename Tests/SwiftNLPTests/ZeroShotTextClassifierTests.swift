import XCTest
import Accelerate
import SwiftDataFrame
@testable import SwiftNLP

final class ZeroShotTextClassifierTests: XCTestCase {

    func testAppleSentenceEmbeddingEngine() throws {
        #if canImport(NaturalLanguage)
        guard let engine = try? AppleSentenceEmbeddingEngine(language: .english) else {
            throw XCTSkip("NLEmbedding not available in this environment")
        }

        XCTAssertEqual(engine.dimension, 512)

        let vector = try engine.embed("Artificial intelligence and machine learning advancements")
        XCTAssertEqual(vector.count, 512)

        // Check unit L2 norm
        var sumSq = 0.0
        vDSP_svesqD(vector, 1, &sumSq, vDSP_Length(vector.count))
        XCTAssertEqual(sqrt(sumSq), 1.0, accuracy: 1e-4)

        // Check empty text returns zeros
        let emptyVec = try engine.embed("")
        XCTAssertEqual(emptyVec.count, 512)
        XCTAssertTrue(emptyVec.allSatisfy { $0 == 0.0 })
        #else
        throw XCTSkip("NaturalLanguage framework not available")
        #endif
    }

    func testZeroShotClassificationDistinctCategories() async throws {
        let classifier = ZeroShotTextClassifier()
        let labels = ["sports", "technology", "finance"]

        let sportsText = "The Los Angeles Lakers defeated the Golden State Warriors in an intense basketball match."
        let sportsResult = try await classifier.classify(text: sportsText, candidateLabels: labels)
        XCTAssertEqual(sportsResult.predictedLabel, "sports")
        XCTAssertGreaterThan(sportsResult.scores["sports"] ?? 0.0, sportsResult.scores["technology"] ?? 0.0)
        XCTAssertGreaterThan(sportsResult.scores["sports"] ?? 0.0, sportsResult.scores["finance"] ?? 0.0)

        // Verify softmax calibration (sum of scores == 1.0)
        let sumScores = sportsResult.scores.values.reduce(0.0, +)
        XCTAssertEqual(sumScores, 1.0, accuracy: 1e-5)

        let techText = "Apple introduced the M4 chip featuring unified memory architecture and accelerated neural engine."
        let techResult = try await classifier.classify(text: techText, candidateLabels: labels)
        XCTAssertEqual(techResult.predictedLabel, "technology")
        XCTAssertGreaterThan(techResult.scores["technology"] ?? 0.0, techResult.scores["sports"] ?? 0.0)

        let financeText = "The central bank decided to increase interest rates to combat rising inflation and stabilize bond yields."
        let financeResult = try await classifier.classify(text: financeText, candidateLabels: labels)
        XCTAssertEqual(financeResult.predictedLabel, "finance")
        XCTAssertGreaterThan(financeResult.scores["finance"] ?? 0.0, financeResult.scores["sports"] ?? 0.0)
    }

    func testZeroShotBatchClassification() async throws {
        let classifier = ZeroShotTextClassifier()
        let labels = ["entertainment", "science", "travel"]

        let texts = [
            "The Hollywood movie broke box office records on its opening weekend.",
            "Astronomers discovered a new exoplanet with possible atmospheric water vapor.",
            "Exploring the scenic beaches and historical landmarks of southern Italy."
        ]

        let results = try await classifier.classifyBatch(texts: texts, candidateLabels: labels)
        XCTAssertEqual(results.count, 3)

        XCTAssertEqual(results[0].predictedLabel, "entertainment")
        XCTAssertEqual(results[1].predictedLabel, "science")
        XCTAssertEqual(results[2].predictedLabel, "travel")

        for r in results {
            let sum = r.scores.values.reduce(0.0, +)
            XCTAssertEqual(sum, 1.0, accuracy: 1e-5)
        }
    }

    func testZeroShotMultiLabelClassification() async throws {
        let classifier = ZeroShotTextClassifier()
        let labels = ["mobile software", "artificial intelligence", "cooking"]

        let text = "The new smartphone operating system features on-device machine learning models for photo search."
        let result = try await classifier.classify(text: text, candidateLabels: labels, multiLabel: true)

        // Both mobile software and AI should have high independent probabilities
        let mobileScore = result.scores["mobile software"] ?? 0.0
        let aiScore = result.scores["artificial intelligence"] ?? 0.0
        let cookingScore = result.scores["cooking"] ?? 0.0

        XCTAssertGreaterThan(mobileScore, cookingScore)
        XCTAssertGreaterThan(aiScore, cookingScore)

        // In multi-label mode, probabilities do NOT sum to 1.0
        for score in result.scores.values {
            XCTAssertGreaterThanOrEqual(score, 0.0)
            XCTAssertLessThanOrEqual(score, 1.0)
        }
    }

    func testZeroShotEdgeCases() async throws {
        let classifier = ZeroShotTextClassifier()
        let labels = ["news", "sports"]

        // Empty text handling
        let emptyResult = try await classifier.classify(text: "", candidateLabels: labels)
        XCTAssertEqual(emptyResult.scores["news"], 0.5)
        XCTAssertEqual(emptyResult.scores["sports"], 0.5)

        // Whitespace-only text handling
        let wsResult = try await classifier.classify(text: "   \n\t ", candidateLabels: labels)
        XCTAssertEqual(wsResult.scores["news"], 0.5)
        XCTAssertEqual(wsResult.scores["sports"], 0.5)

        // Empty candidate labels throws
        do {
            _ = try await classifier.classify(text: "Some text", candidateLabels: [])
            XCTFail("Expected error for empty candidate labels")
        } catch let error as NLPError {
            if case .invalidInput = error {
                // Expected
            } else {
                XCTFail("Unexpected NLPError: \(error)")
            }
        }

        // Single candidate label has 1.0 probability
        let singleResult = try await classifier.classify(text: "Any text here", candidateLabels: ["unique"])
        XCTAssertEqual(singleResult.predictedLabel, "unique")
        XCTAssertEqual(singleResult.scores["unique"] ?? 0.0, 1.0, accuracy: 1e-5)
    }

    func testZeroShotWithLocalEmbeddingEngine() async throws {
        let localEngine = LocalEmbeddingEngine(dimension: 64)
        let classifier = ZeroShotTextClassifier(embeddingProvider: localEngine)

        let labels = ["technology", "food"]
        let text = "neural network processor computation software algorithm"
        let result = try await classifier.classify(text: text, candidateLabels: labels)

        XCTAssertEqual(result.predictedLabel, "technology")
        let sum = result.scores.values.reduce(0.0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-5)
    }

    func testDataFrameZeroShotClassify() async throws {
        let texts = [
            "Lakers won the basketball game last night.",
            "Nvidia unveiled next-generation GPU hardware architecture."
        ]
        let df = try DataFrame(columns: [
            TypedColumn(name: "content", values: texts)
        ])

        let classifiedDF = try await df.zeroShotClassify(
            column: "content",
            candidateLabels: ["sports", "technology"],
            targetColumn: "predicted_category"
        )

        guard let col = classifiedDF[column: "predicted_category", as: String.self] else {
            XCTFail("predicted_category column not found")
            return
        }

        let predictions = col.values.compactMap { $0 }
        XCTAssertEqual(predictions.count, 2)
        XCTAssertEqual(predictions[0], "sports")
        XCTAssertEqual(predictions[1], "technology")
    }
}
