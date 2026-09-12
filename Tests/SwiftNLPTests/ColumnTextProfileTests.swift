import Foundation
import Testing
import SwiftDataFrame
import SwiftNLP
import SwiftML

@Suite("Column Text Profile & Multilingual Stopwords Tests (G-018)")
struct ColumnTextProfileTests {

    @Test("English text column profiling with stopword pruning")
    func testEnglishTextProfiling() throws {
        let texts = [
            "SwiftSci is a high performance scientific computing library.",
            "SwiftSci provides fast dataframes and advanced machine learning models.",
            "Natural language processing in Swift is fast and expressive."
        ]
        let df = try DataFrame(columns: [
            TypedColumn(name: "text", values: texts)
        ])

        let profile = try df.profileTextColumn("text", language: .english, topKTerms: 5)

        #expect(profile.columnName == "text")
        #expect(profile.totalDocuments == 3)
        #expect(profile.nonEmptyDocuments == 3)
        #expect(profile.totalWords > 0)
        #expect(profile.stopwordCount > 0) // "is", "a", "and", "in" pruned
        #expect(profile.totalWordsPruned < profile.totalWords)
        #expect(profile.uniqueWordsPruned <= profile.uniqueWords)
        #expect(profile.typeTokenRatio > 0.0 && profile.typeTokenRatio <= 1.0)
        #expect(profile.prunedTypeTokenRatio > 0.0 && profile.prunedTypeTokenRatio <= 1.0)
        #expect(profile.characterCount > 0)
        #expect(profile.meanDocumentLength > 0.0)
        #expect(profile.stdDocumentLength > 0.0)
        #expect(profile.shannonEntropy > 0.0)
        #expect(!profile.topTerms.isEmpty)
        #expect(profile.topTerms.count <= 5)
        #expect(profile.language == "english")

        // "swiftsci" and "fast" appear multiple times and should be among top terms
        let topTermStrings = profile.topTerms.map { $0.term }
        #expect(topTermStrings.contains("swiftsci") || topTermStrings.contains("fast"))
    }

    @Test("Ukrainian text column profiling with native Ukrainian stopwords")
    func testUkrainianTextProfiling() throws {
        let texts = [
            "Уряд ухвалив новий законопроект про державний бюджет на наступний рік.",
            "Новий бюджет підтримує розвиток наукових досліджень та технологій.",
            "Верховна Рада розглядає законопроект про цифрову трансформацію держави."
        ]
        let df = try DataFrame(columns: [
            TypedColumn(name: "news", values: texts)
        ])

        let profile = try df.profileTextColumn("news", language: .ukrainian, topKTerms: 10)

        #expect(profile.columnName == "news")
        #expect(profile.totalDocuments == 3)
        #expect(profile.nonEmptyDocuments == 3)
        #expect(profile.language == "ukrainian")

        // Ukrainian stopwords: "про", "на", "та" should be counted and pruned
        #expect(profile.stopwordCount >= 3)
        #expect(profile.totalWordsPruned < profile.totalWords)

        // Stopwords should NOT be present in topTerms
        let topWords = Set(profile.topTerms.map { $0.term })
        #expect(!topWords.contains("про"))
        #expect(!topWords.contains("на"))
        #expect(!topWords.contains("та"))

        // Common content words should be preserved
        #expect(topWords.contains("бюджет") || topWords.contains("законопроект") || topWords.contains("новий"))
    }

    @Test("Custom stop words injection")
    func testCustomStopWordsInjection() throws {
        let texts = [
            "apple banana cherry date",
            "apple orange grape banana",
            "cherry date apple banana"
        ]
        let df = try DataFrame(columns: [
            TypedColumn(name: "fruits", values: texts)
        ])

        let profile = try df.profileTextColumn(
            "fruits",
            language: .english,
            customStopWords: ["apple", "banana"],
            topKTerms: 10
        )

        let topWords = Set(profile.topTerms.map { $0.term })
        #expect(!topWords.contains("apple"))
        #expect(!topWords.contains("banana"))
        #expect(topWords.contains("cherry") || topWords.contains("date"))
        #expect(profile.stopwordCount >= 6) // "apple" and "banana" pruned across all documents
    }

    @Test("Missing column throws error")
    func testMissingColumnThrows() throws {
        let df = try DataFrame(columns: [
            TypedColumn(name: "id", values: [1, 2, 3])
        ])

        #expect(throws: (any Error).self) {
            _ = try df.profileTextColumn("nonexistent")
        }
    }

    @Test("Empty DataFrame text column returns zero profile")
    func testEmptyColumnProfile() throws {
        let df = try DataFrame(columns: [
            TypedColumn(name: "empty_text", values: [String]())
        ])

        let profile = try df.profileTextColumn("empty_text")
        #expect(profile.totalDocuments == 0)
        #expect(profile.nonEmptyDocuments == 0)
        #expect(profile.totalWords == 0)
        #expect(profile.uniqueWords == 0)
        #expect(profile.typeTokenRatio == 0.0)
        #expect(profile.shannonEntropy == 0.0)
        #expect(profile.topTerms.isEmpty)
    }

    @Test("ColumnTextProfile is Codable and Equatable")
    func testCodableRoundTrip() throws {
        let original = ColumnTextProfile(
            columnName: "sample",
            totalDocuments: 2,
            nonEmptyDocuments: 2,
            totalWords: 10,
            uniqueWords: 8,
            totalWordsPruned: 6,
            uniqueWordsPruned: 5,
            stopwordCount: 4,
            typeTokenRatio: 0.8,
            prunedTypeTokenRatio: 0.833,
            hapaxLegomenaCount: 4,
            hapaxLegomenaRatio: 0.5,
            characterCount: 60,
            meanDocumentLength: 30.0,
            stdDocumentLength: 2.5,
            minDocumentLength: 28,
            maxDocumentLength: 32,
            meanWordLength: 5.0,
            shannonEntropy: 2.123,
            topTerms: [TermFrequency(term: "swift", count: 3)],
            language: "english"
        )

        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ColumnTextProfile.self, from: encoded)

        #expect(decoded == original)
        #expect(decoded.topTerms == [TermFrequency(term: "swift", count: 3)])
    }
}
