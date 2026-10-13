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

    @Test("GenericResult checks params equality and tie-breaks by stdScore")
    func testGenericResultEqualityAndComparison() {
        let r1 = RandomizedSearchCV.GenericResult(params: "configA", meanScore: 0.85, stdScore: 0.05)
        let r2 = RandomizedSearchCV.GenericResult(params: "configA", meanScore: 0.85, stdScore: 0.05)
        let r3 = RandomizedSearchCV.GenericResult(params: "configB", meanScore: 0.85, stdScore: 0.05)
        let r4 = RandomizedSearchCV.GenericResult(params: "configA", meanScore: 0.85, stdScore: 0.02) // lower std is better

        // Same params, same scores -> equal
        #expect(r1 == r2)
        // Different params, same scores -> not equal
        #expect(r1 != r3)
        // Same params, different std -> not equal
        #expect(r1 != r4)

        // Lower stdScore means better (higher in sorting):
        // r1.stdScore = 0.05, r4.stdScore = 0.02 -> r1 < r4 (r4 is better)
        #expect(r1 < r4)
    }
}

// MARK: - Generic Regression Search & Parameter Validation

extension RandomizedSearchCVTests {

    @Test("searchGeneric supports RegressorEstimator candidates")
    func testSearchGenericRegressor() async throws {
        let n = 24
        let features = (0..<n).map { [Double($0), Double($0) * 1.5] }
        let targets = (0..<n).map { Double($0) * 3.0 + 2.5 }

        let search = RandomizedSearchCV(nIter: 3, nSplits: 3, seed: 42)
        let results = try await search.searchGeneric(
            candidates: [1, 2, 4, 8],
            features: features,
            targets: targets
        ) { depth in
            DecisionTreeRegressor(maxDepth: depth)
        }

        #expect(results.count == 3)
        for i in 1..<results.count {
            #expect(results[0].meanScore >= results[i].meanScore)
        }
    }

    @Test("RandomizedSearchCV validates nIter and candidate list")
    func testParameterValidation() async throws {
        let features: [[Double]] = [[1.0], [2.0], [3.0], [4.0]]
        let targets: [Double] = [0.0, 0.0, 1.0, 1.0]

        await #expect(throws: ValidationError.self) {
            _ = try await RandomizedSearchCV(nIter: 0, nSplits: 2).search(features: features, targets: targets)
        }

        let validSearch = RandomizedSearchCV(nIter: 2, nSplits: 2)
        let emptyCandidates: [Int] = []
        await #expect(throws: ValidationError.self) {
            _ = try await validSearch.searchGeneric(
                candidates: emptyCandidates,
                features: features,
                targets: targets
            ) { depth in
                DecisionTreeClassifier(maxDepth: depth)
            }
        }
    }
}
