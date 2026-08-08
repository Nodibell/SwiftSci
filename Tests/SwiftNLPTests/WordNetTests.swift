import Testing
import Foundation
@testable import SwiftNLP

@Suite("WordNet Tests")
struct WordNetTests {
    
    @Test("WordNet synset lookup by word lemma")
    func testSynsetLookup() throws {
        let wn = WordNet()
        let dogSynsets = wn.synsets(for: "dog")
        #expect(!dogSynsets.isEmpty)
        #expect(dogSynsets[0].id == "dog.n.01")
        #expect(dogSynsets[0].pos == .noun)
    }
    
    @Test("WordNet hypernym and hyponym hierarchy traversal")
    func testHierarchyTraversal() throws {
        let wn = WordNet()
        guard let dog = wn.synsets(for: "dog").first else {
            Issue.record("dog synset not found")
            return
        }
        
        let hypernyms = wn.hypernyms(of: dog)
        #expect(hypernyms.count == 1)
        #expect(hypernyms[0].id == "animal.n.01")
        
        let hyponyms = wn.hyponyms(of: hypernyms[0])
        #expect(hyponyms.map { $0.id }.contains("dog.n.01"))
        #expect(hyponyms.map { $0.id }.contains("cat.n.01"))
    }
    
    @Test("WordNet path distance and similarity metrics")
    func testSimilarityMetrics() throws {
        let wn = WordNet()
        guard let dog = wn.synsets(for: "dog").first,
              let cat = wn.synsets(for: "cat").first else {
            Issue.record("dog/cat synsets not found")
            return
        }
        
        let dist = wn.pathDistance(dog, cat)
        #expect(dist == 2) // dog -> animal -> cat
        
        let pathSim = wn.pathSimilarity(dog, cat)
        #expect(abs(pathSim - (1.0 / 3.0)) < 1e-5)
        
        let wupSim = wn.wupSimilarity(dog, cat)
        #expect(wupSim > 0.5)
    }

    @Test("WordNet wupSimilarity for unrelated synsets")
    func testWupSimilarityUnrelated() throws {
        let wn = WordNet()
        guard let dog = wn.synsets(for: "dog").first,
              let computer = wn.synsets(for: "computer").first else {
            Issue.record("synsets not found")
            return
        }
        let wupSim = wn.wupSimilarity(dog, computer)
        #expect(wupSim > 0.0)
    }
}
