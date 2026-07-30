import Testing
import Foundation
@testable import SwiftNLP
import SwiftDataFrame

@Suite("SwiftNLP 2.4.0 Feature Suite")
struct NLPv24Tests {

    @Test("Tokenizers - Word, Sentence, and Regex")
    func testTokenizers() {
        let text = "Hello world! This is SwiftSci 2.4.0. Isn't it awesome?"
        
        let wordTokenizer = AppleWordTokenizer()
        let words = wordTokenizer.tokenize(text: text)
        #expect(!words.isEmpty)
        #expect(words.contains("Hello"))
        #expect(words.contains("SwiftSci"))

        let sentenceTokenizer = SentenceTokenizer()
        let sentences = sentenceTokenizer.tokenize(text: text)
        #expect(sentences.count >= 2)

        let regexTokenizer = RegexTokenizer(pattern: #"\b\w+\b"#, gaps: false)
        let regexTokens = regexTokenizer.tokenize(text: text)
        #expect(regexTokens.contains("Hello"))
    }

    @Test("Linguistic Tagging - PorterStemmer, POSTagger, and Lemmatizer")
    func testLinguisticTagging() throws {
        let stemmer = PorterStemmer()
        #expect(stemmer.stem("running") == "run")
        #expect(stemmer.stem("capabilities") == "capabl")
        #expect(stemmer.stem("connections") == "connect")

        let posTagger = POSTagger()
        let tagged = posTagger.tag(text: "SwiftSci provides high performance machine learning algorithms.")
        #expect(!tagged.isEmpty)

        #if canImport(NaturalLanguage)
        let lemmatizer = AppleLemmaTagger()
        let lemmas = try lemmatizer.lemmatize(text: "running cars was better")
        #expect(!lemmas.isEmpty)
        #endif
    }

    @Test("Named Entity Recognition")
    func testNamedEntityRecognition() throws {
        #if canImport(NaturalLanguage)
        let ner = AppleNamedEntityRecognizer()
        let entities = try ner.extractEntities(from: "Steve Jobs co-founded Apple in Cupertino, California.")
        #expect(!entities.isEmpty)
        let categories = entities.map { $0.category }
        #expect(categories.contains(.personalName) || categories.contains(.organizationName) || categories.contains(.placeName))
        #endif
    }

    @Test("Sentiment Analysis - VADER & NLSentiment")
    func testSentimentAnalysis() throws {
        let vader = VADERSentimentAnalyzer()

        let positiveScore = vader.polarityScores(text: "SwiftSci is an amazing, fantastic, and wonderful library!!!")
        #expect(positiveScore.compound > 0.5)

        let negativeScore = vader.polarityScores(text: "This product is terrible, awful, and horrible.")
        #expect(negativeScore.compound < -0.5)

        #if canImport(NaturalLanguage)
        let nlSentiment = NLSentimentAnalyzer()
        let score = try nlSentiment.score(text: "I absolutely love writing Swift code.")
        #expect(score > 0.0)
        #endif
    }

    @Test("Language Detection & Embeddings")
    func testLanguageAndEmbeddings() throws {
        #if canImport(NaturalLanguage)
        let detector = AppleLanguageDetector()
        let lang = try detector.detectLanguage(text: "Доброго дня, як у вас справи?")
        #expect(lang == "uk" || lang == "ru")

        let nlEmbedding = AppleNLEmbedding(language: .english)
        if let vec = try nlEmbedding.vector(for: "apple") {
            #expect(!vec.isEmpty)
        }
        #endif

        let embeddingMap: [String: [Double]] = [
            "king": [0.5, 0.8, 0.1],
            "queen": [0.5, 0.7, 0.2],
            "apple": [0.1, 0.0, 0.9]
        ]
        let wordEmbeddings = WordEmbeddings(embeddings: embeddingMap)
        if let sim = wordEmbeddings.cosineSimilarity("king", "queen") {
            #expect(sim > 0.9)
        }
    }

    @Test("Naive Bayes Classifiers - Multinomial & Complement")
    func testNaiveBayesClassifiers() {
        let X = [
            [3.0, 0.0, 1.0], // sports
            [2.0, 1.0, 0.0], // sports
            [0.0, 4.0, 3.0], // tech
            [0.0, 3.0, 4.0]  // tech
        ]
        let y = ["sports", "sports", "tech", "tech"]

        var mnb = MultinomialNaiveBayes(alpha: 1.0)
        mnb.fit(X: X, y: y)

        let pred1 = mnb.predict(x: [3.0, 0.0, 0.0])
        #expect(pred1 == "sports")

        let pred2 = mnb.predict(x: [0.0, 2.0, 3.0])
        #expect(pred2 == "tech")

        var cnb = ComplementNaiveBayes(alpha: 1.0)
        cnb.fit(X: X, y: y)
        let pred3 = cnb.predict(x: [3.0, 0.0, 0.0])
        #expect(pred3 == "sports")
    }

    @Test("DataFrame NLP Extensions")
    func testDataFrameNLPExtensions() throws {
        let df = try DataFrame(columns: [
            TypedColumn<String>(name: "text", values: [
                "SwiftSci is fantastic and awesome!",
                "This component failed completely and broke."
            ])
        ])

        let tokenizedDF = try df.tokenizeColumn("text", targetColumn: "tokens")
        #expect(tokenizedDF.columnNames.contains("tokens"))

        let sentimentDF = try df.analyzeSentiment(column: "text", targetColumn: "sentiment")
        #expect(sentimentDF.columnNames.contains("sentiment"))
        guard let scoresCol = sentimentDF[column: "sentiment", as: Double.self] else {
            Issue.record("Missing sentiment column")
            return
        }
        #expect(scoresCol.values[0]! > 0.0)
        #expect(scoresCol.values[1]! < 0.0)
    }
}
