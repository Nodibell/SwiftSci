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

    @Test("Expanded default vocabulary coverage across POS and domains")
    func testExpandedVocabularyCoverage() throws {
        let wn = WordNet()
        
        // Verbs
        let moveSynsets = wn.synsets(for: "move", pos: .verb)
        #expect(!moveSynsets.isEmpty)
        #expect(moveSynsets[0].pos == .verb)
        
        let runSynsets = wn.synsets(for: "run", pos: .verb)
        #expect(!runSynsets.isEmpty)
        
        // Adjectives
        let fastSynsets = wn.synsets(for: "fast", pos: .adjective)
        #expect(!fastSynsets.isEmpty)
        #expect(fastSynsets[0].pos == .adjective)
        
        let smartSynsets = wn.synsets(for: "smart", pos: .adjective)
        #expect(!smartSynsets.isEmpty)
        
        // Human & Professions
        let scientistSynsets = wn.synsets(for: "scientist", pos: .noun)
        #expect(!scientistSynsets.isEmpty)
        guard let scientist = scientistSynsets.first else { return }
        let scientistHypernyms = wn.hypernyms(of: scientist)
        #expect(scientistHypernyms.map { $0.id }.contains("human.n.01"))
        
        // Technology & Science
        let csSynsets = wn.synsets(for: "computer science", pos: .noun)
        #expect(!csSynsets.isEmpty)
        
        let algoSynsets = wn.synsets(for: "algorithm", pos: .noun)
        #expect(!algoSynsets.isEmpty)
    }

    @Test("Path similarity ranking: dog-cat closer than dog-computer")
    func testPathSimilarityRanking() throws {
        let wn = WordNet()
        guard let dog = wn.synsets(for: "dog").first,
              let cat = wn.synsets(for: "cat").first,
              let comp = wn.synsets(for: "computer").first else {
            Issue.record("Failed to find required synsets")
            return
        }
        
        let simDogCat = wn.pathSimilarity(dog, cat)
        let simDogComp = wn.pathSimilarity(dog, comp)
        #expect(simDogCat > simDogComp)
    }

    @Test("Princeton WordNet line and string parsing")
    func testPrincetonDataParsing() throws {
        // Sample standard Princeton WordNet data.noun lines with header comment
        let rawContent = """
          WordNet (R) 3.0 Copyright (c) 2006 by Princeton University.
          All rights reserved.
        # Format: synset_offset lex_filenum ss_type w_cnt word lex_id p_cnt [ptr_symbol synset_offset pos source/target] | gloss
        00001740 03 n 01 entity 0 003 ~ 00001930 n 0000 ~ 00002137 n 0000 ~ 04424418 n 0000 | that which is perceived or known to have its own distinct existence
        00001930 03 n 02 physical_entity 0 animate_thing 0 001 @ 00001740 n 0000 | an entity that has physical existence
        """
        
        let parsed = try WordNet.parsePrincetonData(rawContent, defaultPOS: .noun)
        #expect(parsed.count == 2)
        
        let entity = parsed[0]
        #expect(entity.id == "00001740-n")
        #expect(entity.name == "entity")
        #expect(entity.pos == .noun)
        #expect(entity.definition.contains("distinct existence"))
        #expect(entity.hyponymIDs.contains("00001930-n"))
        #expect(entity.hyponymIDs.contains("00002137-n"))
        
        let physicalEntity = parsed[1]
        #expect(physicalEntity.id == "00001930-n")
        #expect(physicalEntity.lemmas.contains("physical entity"))
        #expect(physicalEntity.lemmas.contains("animate thing"))
        #expect(physicalEntity.hypernymIDs.contains("00001740-n"))
        
        // Test WordNet instance created from parsed synsets
        let customWN = WordNet(synsets: parsed)
        let matches = customWN.synsets(for: "physical entity")
        #expect(matches.count == 1)
        #expect(matches[0].id == "00001930-n")
        
        let hypernyms = customWN.hypernyms(of: physicalEntity)
        #expect(hypernyms.count == 1)
        #expect(hypernyms[0].id == "00001740-n")
    }

    @Test("Princeton WordNet file load from temporary directory")
    func testFileLoading() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }
        
        let dataNounURL = tempDir.appendingPathComponent("data.noun")
        let nounContent = """
        00002137 03 n 01 flora 0 001 @ 00001740 n 0000 | all the plant life in a particular region or period
        """
        try nounContent.write(to: dataNounURL, atomically: true, encoding: .utf8)
        
        let synsets = try WordNet.load(fromDataFile: dataNounURL, pos: .noun)
        #expect(synsets.count == 1)
        #expect(synsets[0].lemmas.contains("flora"))
        
        let loadedWN = try WordNet.load(fromDirectory: tempDir)
        #expect(!loadedWN.synsets(for: "flora").isEmpty)
    }

    @Test("WordNet.load(fromDirectory:) throws fileNoSuchFile when no data files exist")
    func testLoadFromEmptyDirectoryThrows() throws {
        let emptyDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: emptyDir) }

        #expect(throws: CocoaError.self) {
            _ = try WordNet.load(fromDirectory: emptyDir)
        }
    }

    @Test("Princeton WordNet parser skips malformed non-comment lines")
    func testPrincetonMalformedLineSkipped() throws {
        let malformedContent = """
        invalid_short_token
        00002137 03 n 01 flora 0 001 @ 00001740 n 0000 | valid entry
        """
        let parsed = try WordNet.parsePrincetonData(malformedContent, defaultPOS: .noun)
        #expect(parsed.count == 1)
        #expect(parsed[0].name == "flora")
    }

    @Test("Princeton WordNet file load from dict subdirectory")
    func testFileLoadingFromDictSubdir() throws {
        let baseDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let dictDir = baseDir.appendingPathComponent("dict")
        try FileManager.default.createDirectory(at: dictDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: baseDir) }

        let dataNounURL = dictDir.appendingPathComponent("data.noun")
        let nounContent = """
        00002137 03 n 01 fauna 0 001 @ 00001740 n 0000 | animals of a region
        """
        try nounContent.write(to: dataNounURL, atomically: true, encoding: .utf8)

        let loadedWN = try WordNet.load(fromDirectory: baseDir)
        #expect(!loadedWN.synsets(for: "fauna").isEmpty)
    }

    @Test("Cookbook Recipe 8.4 compiles and executes properly")
    func testCookbookRecipe84() throws {
        let wordnet = WordNet()
        let dogSynsets = wordnet.synsets(for: "dog", pos: .noun)
        let catSynsets = wordnet.synsets(for: "cat", pos: .noun)
        let compSynsets = wordnet.synsets(for: "computer", pos: .noun)

        guard let dog = dogSynsets.first,
              let cat = catSynsets.first,
              let computer = compSynsets.first else {
            Issue.record("Required synsets not found")
            return
        }

        let hypernyms = wordnet.hypernyms(of: dog)
        #expect(!hypernyms.isEmpty)

        let distDogCat = wordnet.pathDistance(dog, cat)
        let simDogCat = wordnet.pathSimilarity(dog, cat)
        let wupDogCat = wordnet.wupSimilarity(dog, cat)
        let wupDogComp = wordnet.wupSimilarity(dog, computer)

        #expect(distDogCat == 2)
        #expect(simDogCat > 0.0)
        #expect(wupDogCat > wupDogComp)
    }
}
