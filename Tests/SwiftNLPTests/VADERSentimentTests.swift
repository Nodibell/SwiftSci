import Testing
import Foundation
@testable import SwiftNLP

@Suite("VADER Sentiment Analyzer Tests")
struct VADERSentimentTests {

    @Test("VADER accounts for neutral tokens in pos/neg/neu proportions")
    func testVADERNeutralTokensAccounting() {
        let vader = VADERSentimentAnalyzer()
        let text = "The movie was absolutely wonderful."
        let score = vader.polarityScores(text: text)

        // "The", "movie", "was" are neutral tokens (valence 0.0)
        // neu must be > 0.0 rather than 0.0
        #expect(score.neu > 0.0)
        #expect(score.pos > 0.0)
        #expect(score.neg == 0.0)
        #expect(abs((score.pos + score.neg + score.neu) - 1.0) < 1e-4)

        // Purely neutral sentence should result in neu == 1.0, compound == 0.0
        let neutralText = "The cat sat on the mat."
        let neutralScore = vader.polarityScores(text: neutralText)
        #expect(neutralScore.neu == 1.0)
        #expect(neutralScore.pos == 0.0)
        #expect(neutralScore.neg == 0.0)
        #expect(neutralScore.compound == 0.0)
    }

    @Test("VADER 'but' contrast rule attenuates pre-but and amplifies post-but clauses")
    func testVADERButRule() {
        let vader = VADERSentimentAnalyzer()

        // "The movie was bad, but the ending was great."
        // bad before 'but' gets 0.5x, great after 'but' gets 1.5x -> net positive
        let butPositive = vader.polarityScores(text: "The movie was bad, but the ending was great.")
        #expect(butPositive.compound > 0.0)

        // "The movie was great, but the ending was bad."
        // great before 'but' gets 0.5x, bad after 'but' gets 1.5x -> net negative
        let butNegative = vader.polarityScores(text: "The movie was great, but the ending was bad.")
        #expect(butNegative.compound < 0.0)
    }

    @Test("VADER punctuation amplification for exclamation and question marks")
    func testVADERPunctuationAmplification() {
        let vader = VADERSentimentAnalyzer()

        let baseScore = vader.polarityScores(text: "This is great")
        let exclamScore = vader.polarityScores(text: "This is great!!!")
        #expect(exclamScore.compound > baseScore.compound)

        let questScore = vader.polarityScores(text: "This is great???")
        #expect(questScore.compound > baseScore.compound)
    }

    @Test("VADER ALL CAPS differential boost when mixed case contrast exists")
    func testVADERAllCapsDifferential() {
        let vader = VADERSentimentAnalyzer()

        let normalScore = vader.polarityScores(text: "The food was wonderful.")
        let capsWordScore = vader.polarityScores(text: "The food was WONDERFUL.")
        // Mixed-case sentence with one ALL CAPS word triggers isCapDiff boost
        #expect(capsWordScore.compound > normalScore.compound)

        // All uppercase sentence does not have differential contrast across words
        let allCapsScore = vader.polarityScores(text: "THE FOOD WAS WONDERFUL.")
        #expect(capsWordScore.compound > allCapsScore.compound)
    }
}
