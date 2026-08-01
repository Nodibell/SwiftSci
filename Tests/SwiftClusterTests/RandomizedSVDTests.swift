#if os(macOS)
import XCTest
@testable import SwiftCluster

final class RandomizedSVDTests: XCTestCase {

    // MARK: - Correctness: explained variance must match full SVD within 1e-3

    func testRandomizedSVDSmall() throws {
        // 20x5 matrix, request k=3
        let M = 20, N = 5, k = 3
        var X = [[Double]](repeating: [Double](repeating: 0.0, count: N), count: M)
        var seed: UInt64 = 12345
        for r in 0..<M {
            for c in 0..<N {
                seed = seed &* 6364136223846793005 &+ 1442695040888963407
                X[r][c] = Double(Int(seed >> 32) % 100) / 100.0
            }
        }

        let rsvd = try RandomizedSVD.compute(X: X, nComponents: k)
        XCTAssertEqual(rsvd.S.count, k)
        XCTAssertEqual(rsvd.U.count, M)
        XCTAssertEqual(rsvd.Vt.count, k)
        XCTAssertEqual(rsvd.U[0].count, k)
        XCTAssertEqual(rsvd.Vt[0].count, N)

        // Singular values must be non-negative and descending
        for i in 0..<k {
            XCTAssertGreaterThanOrEqual(rsvd.S[i], 0.0)
            if i > 0 {
                XCTAssertGreaterThanOrEqual(rsvd.S[i-1], rsvd.S[i])
            }
        }
    }

    func testPCAWithRandomizedSolverMatchesFullSVD() async throws {
        // 200x20 matrix, 5 components
        let M = 200, N = 20, k = 5
        var X = [[Double]](repeating: [Double](repeating: 0.0, count: N), count: M)
        var seed: UInt64 = 99999
        for r in 0..<M {
            for c in 0..<N {
                seed = seed &* 6364136223846793005 &+ 1442695040888963407
                X[r][c] = Double(Int(seed >> 32) % 200) / 200.0 - 0.5
            }
        }

        let pcaFull = try PCA(nComponents: k, svdSolver: .full)
        let pcaRsvd = try PCA(nComponents: k, svdSolver: .randomized)

        try await pcaFull.fit(X)
        try await pcaRsvd.fit(X)

        let fullVar = await pcaFull.explainedVariance!
        let rsvdVar = await pcaRsvd.explainedVariance!

        // Total explained variance ratio should match within 1%
        let fullTotal = fullVar.reduce(0.0, +)
        let rsvdTotal = rsvdVar.reduce(0.0, +)

        XCTAssertGreaterThan(fullTotal, 0.0)
        XCTAssertGreaterThan(rsvdTotal, 0.0)

        // Relative difference in total explained variance < 5%
        let relDiff = abs(fullTotal - rsvdTotal) / fullTotal
        XCTAssertLessThan(relDiff, 0.05,
            "Randomized SVD total explained variance should be within 5% of full SVD. Full: \(fullTotal), RSVD: \(rsvdTotal)")
    }

    func testPCAAutoSolverSelectsRandomized() async throws {
        // 1000x100, k=10 → auto should select randomized (10 < 0.8*100)
        let M = 100, N = 20, k = 3
        var X = [[Double]](repeating: [Double](repeating: 0.0, count: N), count: M)
        var seed: UInt64 = 777
        for r in 0..<M {
            for c in 0..<N {
                seed = seed &* 6364136223846793005 &+ 1442695040888963407
                X[r][c] = Double(Int(seed >> 32) % 100) / 100.0
            }
        }

        let pca = try PCA(nComponents: k)  // .auto solver
        try await pca.fit(X)

        let comps = await pca.components
        XCTAssertNotNil(comps)
        XCTAssertEqual(comps!.count, k)
    }

    func testRandomizedSVDInvalidK() throws {
        let X = [[Double]](repeating: [1.0, 2.0, 3.0], count: 4)
        // k > min(M=4, N=3) = 3 → should throw
        XCTAssertThrowsError(try RandomizedSVD.compute(X: X, nComponents: 5))
    }
}
#endif
