import Testing
import Foundation
@testable import SwiftNLP

@Suite("SwiftNLP v3.9 Tests: N-Grams, Sublinear TF, MaxDF & TextCleaner")
struct NLPv39Tests {
    
    @Test("TFIDFVectorizer extracts unigrams and bigrams when ngramRange is 1...2")
    func testNgramRange() throws {
        let corpus = [
            "штучний інтелект змінює технології",
            "технології та штучний інтелект розвиваються"
        ]
        
        let vec = TFIDFVectorizer(ngramRange: 1...2, minDF: 1, removeStopWords: false)
        try vec.fit(corpus)
        
        let vocab = vec.vocabulary
        #expect(vocab["штучний"] != nil)
        #expect(vocab["інтелект"] != nil)
        #expect(vocab["штучний інтелект"] != nil)
        
        let matrix = try vec.transform(corpus)
        #expect(matrix.count == 2)
        #expect(matrix[0].count == vocab.count)
        
        if let bigramIdx = vocab["штучний інтелект"] {
            #expect(matrix[0][bigramIdx] > 0.0)
            #expect(matrix[1][bigramIdx] > 0.0)
        }
    }
    
    @Test("TFIDFVectorizer sublinear TF scaling dampens high counts")
    func testSublinearTF() throws {
        let corpus = [
            "новини новини новини спорт",
            "новини спорт спорт спорт"
        ]
        
        let vecStandard = TFIDFVectorizer(minDF: 1, sublinearTF: false, removeStopWords: false)
        try vecStandard.fit(corpus)
        let matStd = try vecStandard.transform(corpus)
        
        let vecSublinear = TFIDFVectorizer(minDF: 1, sublinearTF: true, removeStopWords: false)
        try vecSublinear.fit(corpus)
        let matSub = try vecSublinear.transform(corpus)
        
        let newsIdx = vecStandard.vocabulary["новини"]!
        let sportIdx = vecStandard.vocabulary["спорт"]!
        
        // In standard TF, ratio of 3 vs 1 count is 3.0 (0.75 / 0.25)
        // In sublinear TF, ratio of (1 + log(3)) / (1 + log(1)) is dampened to ~2.0986
        let ratioStd = matStd[0][newsIdx] / matStd[0][sportIdx]
        let ratioSub = matSub[0][newsIdx] / matSub[0][sportIdx]
        
        #expect(ratioStd > 2.9)
        #expect(ratioSub < ratioStd)
        #expect(ratioSub > 2.0)
    }
    
    @Test("TFIDFVectorizer prunes terms exceeding maxDF")
    func testMaxDF() throws {
        let corpus = [
            "загальне слово спорт футбол",
            "загальне слово політика вибори",
            "загальне слово бізнес інвестиції",
            "загальне слово технології штучний"
        ]
        
        // "загальне" and "слово" appear in 100% of docs (4/4 = 1.0).
        // Setting maxDF = 0.5 should exclude them from vocabulary.
        let vec = TFIDFVectorizer(minDF: 1, maxDF: 0.5, removeStopWords: false)
        try vec.fit(corpus)
        
        let vocab = vec.vocabulary
        #expect(vocab["загальне"] == nil)
        #expect(vocab["слово"] == nil)
        #expect(vocab["спорт"] != nil)
        #expect(vocab["футбол"] != nil)
    }
    
    @Test("TextCleaner strips URLs, HTML, emails and punctuation")
    func testTextCleaner() {
        let dirty = "Увага! Читайте новини на https://example.com/news або пишіть на info@news.ua. <b>Свіжі факти</b>!"
        let cleaned = TextCleaner.clean(dirty, stripURLs: true, stripEmails: true, stripHTML: true, stripPunctuation: true, lowercase: true)
        
        #expect(!cleaned.contains("https://"))
        #expect(!cleaned.contains("info@news.ua"))
        #expect(!cleaned.contains("<b>"))
        #expect(!cleaned.contains("!"))
        #expect(cleaned.contains("увага читайте новини на або пишіть на свіжі факти"))
    }
}
