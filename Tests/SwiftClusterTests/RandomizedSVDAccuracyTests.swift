#if os(macOS)
import Testing
import Foundation
@testable import SwiftCluster

/// Accuracy of Randomized SVD relative to the optimal rank-k error `sigma_{k+1}` (Eckart–Young),
/// across spectra. The error bound is probabilistic (Halko–Martinsson–Tropp 2011), so these tests
/// assert (a) the residual never beats the optimum, and (b) a generous empirical factor per spectrum.
@Suite("RandomizedSVD spectral accuracy")
struct RandomizedSVDAccuracyTests {
    private struct LCG { var s: UInt64
        mutating func next() -> Double { s = s &* 6364136223846793005 &+ 1442695040888963407; return Double(s >> 11) / Double(1 << 53) - 0.5 } }

    private func orthonormal(_ n: Int, _ cols: Int, seed: UInt64) -> [[Double]] {
        var g = LCG(s: seed)
        var vs: [[Double]] = []
        while vs.count < cols {
            var v = (0..<n).map { _ in g.next() }
            for _ in 0..<2 { for u in vs { let d = zip(u, v).reduce(0) { $0 + $1.0 * $1.1 }; for i in 0..<n { v[i] -= d * u[i] } } }
            let nrm = sqrt(v.reduce(0) { $0 + $1 * $1 })
            if nrm > 1e-8 { vs.append(v.map { $0 / nrm }) }
        }
        return vs
    }

    private func build(_ sigma: [Double], rows: Int, cols: Int, seed: UInt64) -> [[Double]] {
        let U = orthonormal(rows, sigma.count, seed: seed), V = orthonormal(cols, sigma.count, seed: seed &+ 99)
        var A = [[Double]](repeating: [Double](repeating: 0, count: cols), count: rows)
        for r in 0..<sigma.count { for i in 0..<rows { let a = sigma[r] * U[r][i]; for j in 0..<cols { A[i][j] += a * V[r][j] } } }
        return A
    }

    private func spectralNorm(_ R: [[Double]]) -> Double {
        let m = R.count, n = R[0].count
        var v = (0..<n).map { Double($0 % 7) + 1 }
        var sigma = 0.0
        for _ in 0..<400 {
            var w = [Double](repeating: 0, count: m)
            for i in 0..<m { for j in 0..<n { w[i] += R[i][j] * v[j] } }
            var z = [Double](repeating: 0, count: n)
            for i in 0..<m { for j in 0..<n { z[j] += R[i][j] * w[i] } }
            let nz = sqrt(z.reduce(0) { $0 + $1 * $1 })
            if nz == 0 { return 0 }
            v = z.map { $0 / nz }
            sigma = sqrt(nz)
        }
        return sigma
    }

    private func residual(_ A: [[Double]], _ r: RandomizedSVD.Result) -> Double {
        let m = A.count, n = A[0].count, k = r.S.count
        var R = A
        for t in 0..<k { for i in 0..<m { let a = r.U[i][t] * r.S[t]; for j in 0..<n { R[i][j] -= a * r.Vt[t][j] } } }
        return spectralNorm(R)
    }

    private func check(_ sigma: [Double], k: Int, p: Int = 10, q: Int = 2, factor: Double, seeds: [UInt64] = [1, 2, 3, 4, 5]) throws {
        let A = build(sigma, rows: 80, cols: 50, seed: 11)
        let opt = sigma[k]
        for seed in seeds {
            let r = try RandomizedSVD.compute(X: A, nComponents: k, nOversamples: p, nPowerIter: q, seed: seed)
            let err = residual(A, r)
            #expect(err >= opt * (1 - 1e-6), "residual \(err) cannot beat optimum \(opt)")
            #expect(err <= opt * factor, "p=\(p) q=\(q) seed=\(seed): residual \(err) vs sigma_k+1 \(opt)")
        }
    }

    @Test func fastDecay() throws { try check((0..<50).map { pow(0.5, Double($0)) }, k: 5, factor: 1.05) }
    @Test func slowDecay() throws { try check((0..<50).map { 1.0 / pow(Double($0 + 1), 0.5) }, k: 5, factor: 1.6) }
    @Test func nearlyEqualSingularValues() throws { try check((0..<50).map { 1.0 - 0.001 * Double($0) }, k: 5, factor: 1.1) }
    @Test func illConditioned() throws { try check((0..<50).map { pow(10.0, -Double($0) * 0.3) }, k: 6, factor: 1.5) }

    @Test("More power iterations never hurt accuracy on a slow spectrum (averaged over seeds)")
    func powerIterationsImprove() throws {
        let sigma = (0..<50).map { 1.0 / pow(Double($0 + 1), 0.3) }
        let A = build(sigma, rows: 80, cols: 50, seed: 5)
        func mean(q: Int) throws -> Double {
            var t = 0.0
            for s: UInt64 in 1...8 { t += residual(A, try RandomizedSVD.compute(X: A, nComponents: 5, nOversamples: 5, nPowerIter: q, seed: s)) }
            return t / 8
        }
        let e0 = try mean(q: 0), e2 = try mean(q: 2), e4 = try mean(q: 4)
        #expect(e2 <= e0 * 1.0001)
        #expect(e4 <= e2 * 1.05)
    }

    @Test func rankDeficientMatrix() throws {
        let A = build([5, 3, 1], rows: 30, cols: 20, seed: 3)
        let r = try RandomizedSVD.compute(X: A, nComponents: 3)
        #expect(residual(A, r) < 1e-8)
    }

    @Test func invalidParametersThrow() {
        let A = build([2, 1], rows: 10, cols: 6, seed: 1)
        #expect(throws: (any Error).self) { try RandomizedSVD.compute(X: A, nComponents: 0) }
        #expect(throws: (any Error).self) { try RandomizedSVD.compute(X: A, nComponents: 7) }
        #expect(throws: (any Error).self) { try RandomizedSVD.compute(X: A, nComponents: 2, nOversamples: -1) }
        #expect(throws: (any Error).self) { try RandomizedSVD.compute(X: A, nComponents: 2, nPowerIter: -1) }
        #expect(throws: (any Error).self) { try RandomizedSVD.compute(X: [], nComponents: 1) }
    }

    @Test func kPlusPExceedingMinDimIsClampedAndExact() throws {
        let A = build([4, 3, 2, 1], rows: 12, cols: 6, seed: 2)
        let r = try RandomizedSVD.compute(X: A, nComponents: 4, nOversamples: 100)
        #expect(residual(A, r) < 1e-8)
    }
}
#endif
