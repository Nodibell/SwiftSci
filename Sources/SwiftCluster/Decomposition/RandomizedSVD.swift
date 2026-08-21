#if os(macOS)
import Foundation
import Accelerate

// MARK: - LAPACK Wrappers (RandomizedSVD)

private func dgemm_rsvd(
    _ transa: UnsafeMutablePointer<Int8>,
    _ transb: UnsafeMutablePointer<Int8>,
    _ m: UnsafeMutablePointer<Int32>,
    _ n: UnsafeMutablePointer<Int32>,
    _ k: UnsafeMutablePointer<Int32>,
    _ alpha: UnsafeMutablePointer<Double>,
    _ a: UnsafePointer<Double>,
    _ lda: UnsafeMutablePointer<Int32>,
    _ b: UnsafePointer<Double>,
    _ ldb: UnsafeMutablePointer<Int32>,
    _ beta: UnsafeMutablePointer<Double>,
    _ c: UnsafeMutablePointer<Double>,
    _ ldc: UnsafeMutablePointer<Int32>
) {
    cblas_dgemm(
        CblasColMajor,
        transa.pointee == Int8(78) ? CblasNoTrans : CblasTrans,
        transb.pointee == Int8(78) ? CblasNoTrans : CblasTrans,
        Int32(m.pointee), Int32(n.pointee), Int32(k.pointee),
        alpha.pointee,
        a, Int32(lda.pointee),
        b, Int32(ldb.pointee),
        beta.pointee,
        c, Int32(ldc.pointee)
    )
}

/// Randomized Singular Value Decomposition using the Halko et al. (2011) algorithm.
///
/// Computes an approximate rank-k SVD of an M×N matrix A:
///   A ≈ U * diag(S) * Vᵀ
///
/// where U is M×k orthonormal, S is k-vector of singular values, Vᵀ is k×N.
///
/// ## Complexity
/// O(M·N·k) — dramatically faster than full `dgesdd_` O(M·N²) when k ≪ min(M, N).
///
/// ## Algorithm (Halko 2011)
/// 1. Draw random Gaussian matrix Ω ∈ ℝ^{N×(k+p)}  (p=10 oversampling).
/// 2. Compute range sketch Y = A·Ω  via BLAS dgemm.
/// 3. Orthonormalize Y via thin QR (dgeqrf + dorgqr).
/// 4. Project B = Qᵀ·A  (small k×N matrix).
/// 5. Compute small exact SVD of B with dgesdd_.
/// 6. Back-project left singular vectors: U = Q·Ũ.
///
/// ## Reference
/// Halko N., Martinsson P.G., Tropp J.A. (2011). Finding Structure with Randomness:
/// Probabilistic Algorithms for Constructing Approximate Matrix Decompositions.
/// SIAM Review 53(2):217–288.
public enum RandomizedSVD {

    /// Result of a Randomized SVD decomposition.
    public struct Result {
        /// Left singular vectors, shape [M, k].
        public let U: [[Double]]
        /// Singular values, shape [k] (descending).
        public let S: [Double]
        /// Right singular vectors (transposed), shape [k, N].
        public let Vt: [[Double]]
    }

    /// Computes an approximate rank-k SVD of matrix X (row-major, M rows × N cols).
    ///
    /// - Parameters:
    ///   - X: Input matrix, shape [M, N], row-major.
    ///   - k: Number of singular values/vectors to compute (k).
    ///   - p: Extra random columns for accuracy (default 10).
    ///   - q: Number of power iterations for improved accuracy on slow-decay spectra (default 2).
    ///   - seed: Seed for deterministic random sketch (default 42).
    /// - Returns: `RandomizedSVD.Result` with U [M×k], S [k], Vt [k×N].
    public static func compute(
        X: [[Double]],
        nComponents k: Int,
        nOversamples p: Int = 10,
        nPowerIter q: Int = 2,
        seed: UInt64 = 42
    ) throws -> Result {
        guard !X.isEmpty, !X[0].isEmpty else {
            throw RandomizedSVDError.invalidDimensions(rows: X.count, cols: X.first?.count ?? 0)
        }
        let M = X.count
        let N = X[0].count

        guard k > 0, k <= min(M, N) else {
            throw RandomizedSVDError.invalidComponents(k: k, minDim: min(M, N))
        }

        let l = min(k + p, min(M, N)) // sketch size, bounded by matrix dimensions

        // --- Flatten X to column-major [M×N] ---
        var A = [Double](repeating: 0.0, count: M * N)
        for r in 0..<M {
            for c in 0..<N {
                A[c * M + r] = X[r][c]  // column-major
            }
        }

        // --- Step 1: Random Gaussian sketch Ω ∈ ℝ^{N×l} (column-major) ---
        let omega = randomGaussian(count: N * l, seed: seed)

        // --- Step 2: Y = A·Ω  →  shape [M×l] (column-major) ---
        // dgemm: C = alpha*A*B + beta*C
        // A is M×N colmaj, Omega is N×l colmaj → Y is M×l colmaj
        var Y = [Double](repeating: 0.0, count: M * l)
        let mInt = Int32(M), nInt = Int32(l), kInt = Int32(N)
        let alpha = 1.0, beta = 0.0
        let ldA = Int32(M), ldOm = Int32(N), ldY = Int32(M)
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    mInt, nInt, kInt,
                    alpha, A, ldA,
                    omega, ldOm,
                    beta, &Y, ldY)

        // --- Power iterations: Y = (A·Aᵀ)^q · Y for better spectral decay ---
        for _ in 0..<q {
            // Z = Aᵀ·Y  → shape [N×l]
            var Z = [Double](repeating: 0.0, count: N * l)
            let mZ = Int32(N), nZ = Int32(l), kZ = Int32(M)
            let ldAT = Int32(M), ldZ = Int32(N)
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        mZ, nZ, kZ,
                        alpha, A, ldAT,
                        Y, ldY,
                        beta, &Z, ldZ)
            // Y = A·Z  → back to [M×l]
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        mInt, nInt, kInt,
                        alpha, A, ldA,
                        Z, ldZ,
                        beta, &Y, ldY)
        }

        // --- Step 3: QR decompose Y → Q [M×l], R [l×l] ---
        var tau = [Double](repeating: 0.0, count: l)
        var infoInt = Int32(0)
        var mLap = Int32(M), nLap = Int32(l), ldLap = Int32(M)
        var lwork = Int32(-1)
        var workQ = [Double](repeating: 0.0, count: 1)
        dgeqrf_(&mLap, &nLap, &Y, &ldLap, &tau, &workQ, &lwork, &infoInt)
        lwork = Int32(workQ[0])
        var work = [Double](repeating: 0.0, count: Int(lwork))
        dgeqrf_(&mLap, &nLap, &Y, &ldLap, &tau, &work, &lwork, &infoInt)

        // Extract Q from Y (in-place Householder reflectors)
        var nQ = Int32(l)
        lwork = -1
        var workQR = [Double](repeating: 0.0, count: 1)
        dorgqr_(&mLap, &nQ, &nQ, &Y, &ldLap, &tau, &workQR, &lwork, &infoInt)
        lwork = Int32(workQR[0])
        var workQ2 = [Double](repeating: 0.0, count: Int(lwork))
        dorgqr_(&mLap, &nQ, &nQ, &Y, &ldLap, &tau, &workQ2, &lwork, &infoInt)
        let Q = Y  // Q is now M×l column-major

        // --- Step 4: B = Qᵀ·A  → shape [l×N] ---
        var B = [Double](repeating: 0.0, count: l * N)
        let mB = Int32(l), nB = Int32(N), kB = Int32(M)
        let ldQ = Int32(M), ldB = Int32(l)
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    mB, nB, kB,
                    alpha, Q, ldQ,
                    A, ldA,
                    beta, &B, ldB)

        // --- Step 5: Small exact SVD of B [l×N] ---
        var jobz: Int8 = 83  // 'S'
        var mSVD = Int32(l), nSVD = Int32(N)
        var ldB2 = Int32(l)
        let minDim = min(l, N)
        var sVals = [Double](repeating: 0.0, count: minDim)
        var uMat = [Double](repeating: 0.0, count: l * minDim)
        var vtMat = [Double](repeating: 0.0, count: minDim * N)
        var ldU = Int32(l), ldVt = Int32(minDim)
        var iwork = [Int32](repeating: 0, count: 8 * minDim)
        lwork = -1
        var workSVD = [Double](repeating: 0.0, count: 1)
        dgesdd_(&jobz, &mSVD, &nSVD, &B, &ldB2,
                &sVals, &uMat, &ldU, &vtMat, &ldVt,
                &workSVD, &lwork, &iwork, &infoInt)
        lwork = Int32(workSVD[0])
        var workFull = [Double](repeating: 0.0, count: Int(lwork))
        dgesdd_(&jobz, &mSVD, &nSVD, &B, &ldB2,
                &sVals, &uMat, &ldU, &vtMat, &ldVt,
                &workFull, &lwork, &iwork, &infoInt)

        // --- Step 6: Back-project U = Q·Ũ  → shape [M×k] ---
        var Ufull = [Double](repeating: 0.0, count: M * minDim)
        let mU = Int32(M), nU = Int32(minDim), kU = Int32(l)
        let ldU2 = Int32(M)
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    mU, nU, kU,
                    alpha, Q, ldQ,
                    uMat, ldU,
                    beta, &Ufull, ldU2)

        // --- Extract top-k components ---
        var uRows = [[Double]]()
        uRows.reserveCapacity(M)
        for r in 0..<M {
            var row = [Double](repeating: 0.0, count: k)
            for c in 0..<k { row[c] = Ufull[c * M + r] }
            uRows.append(row)
        }

        let sTop = Array(sVals.prefix(k))

        var vtRows = [[Double]]()
        vtRows.reserveCapacity(k)
        for r in 0..<k {
            var row = [Double](repeating: 0.0, count: N)
            for c in 0..<N { row[c] = vtMat[c * minDim + r] }
            vtRows.append(row)
        }

        return Result(U: uRows, S: sTop, Vt: vtRows)
    }

    // MARK: – Private Helpers

    /// Generates a pseudo-random Gaussian vector using Box-Muller transform and a simple LCG.
    private static func randomGaussian(count: Int, seed: UInt64) -> [Double] {
        var state = seed &+ 0x9E3779B97F4A7C15
        var result = [Double](repeating: 0.0, count: count)
        var i = 0
        while i < count - 1 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let u1 = Double(state >> 32) / Double(UInt32.max)
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let u2 = Double(state >> 32) / Double(UInt32.max)
            let r = (-2.0 * log(max(u1, 1e-10))).squareRoot()
            let theta = 2.0 * Double.pi * u2
            result[i]     = r * cos(theta)
            result[i + 1] = r * sin(theta)
            i += 2
        }
        if i < count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            result[i] = Double(state >> 32) / Double(UInt32.max) - 0.5
        }
        return result
    }
}

// MARK: - Error

/// Errors that can occur during Randomized SVD computation.
public enum RandomizedSVDError: Error {
    /// Raised when target rank k exceeds matrix bounds.
    case invalidComponents(k: Int, minDim: Int)
    /// Raised when matrix has zero rows or zero columns.
    case invalidDimensions(rows: Int, cols: Int)
}

#endif // os(macOS)
