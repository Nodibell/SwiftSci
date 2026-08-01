import Testing
import Foundation
import SwiftML
@testable import SwiftOptimize

@Suite("RandomizedSearchCV Tests")
struct RandomizedSearchCVTests {
    
    @Test("Legacy decision tree randomized search")
    func testLegacyRandomizedSearch() async throws {
        let features: [[Double]] = [
            [1.0, 2.0],
            [2.0, 1.0],
            [5.0, 6.0],
            [6.0, 5.0]
        ]
        let targets: [Double] = [0.0, 0.0, 1.0, 1.0]
        
        let search = RandomizedSearchCV(
            maxDepthValues: [2, 3, 5],
            criterionValues: [.gini],
            nIter: 2,
            nSplits: 2
        )
        
        let results = try await search.search(features: features, targets: targets)
        #expect(!results.isEmpty)
        #expect(results.count <= 2)
    }
    
    @Test("Generic randomized search with custom parameters")
    func testGenericRandomizedSearch() async throws {
        let features: [[Double]] = [
            [1.0, 2.0],
            [2.0, 1.0],
            [5.0, 6.0],
            [6.0, 5.0]
        ]
        let targets: [Double] = [0.0, 0.0, 1.0, 1.0]
        
        let candidates: [Int] = [2, 3, 4]
        let search = RandomizedSearchCV(nIter: 2, nSplits: 2)
        
        let results = try await search.searchGeneric(
            candidates: candidates,
            features: features,
            targets: targets
        ) { depth in
            DecisionTreeClassifier(maxDepth: depth)
        }
        
        #expect(!results.isEmpty)
        #expect(results.count == 2)
        #expect(results[0].meanScore >= results[1].meanScore)
    }
}
