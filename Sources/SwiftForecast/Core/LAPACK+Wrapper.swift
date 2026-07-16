import Foundation
import Accelerate

#if ACCELERATE_NEW_LAPACK
  #if ACCELERATE_LAPACK_ILP64
  typealias LAPACKInteger = Int
  #else
  typealias LAPACKInteger = Int32
  #endif
#else
typealias LAPACKInteger = __CLPK_integer
#endif

// Wrappers to avoid deprecation warnings
func forecast_dgemm(
    _ Order: CBLAS_ORDER, _ TransA: CBLAS_TRANSPOSE, _ TransB: CBLAS_TRANSPOSE,
    _ M: Int, _ N: Int, _ K: Int,
    _ alpha: Double, _ A: [Double], _ lda: Int,
    _ B: [Double], _ ldb: Int,
    _ beta: Double, _ C: inout [Double], _ ldc: Int
) {
    #if ACCELERATE_NEW_LAPACK
    cblas_dgemm(
        Order, TransA, TransB,
        M, N, K,
        alpha, A, lda,
        B, ldb,
        beta, &C, ldc
    )
    #else
    cblas_dgemm(
        Order, TransA, TransB,
        Int32(M), Int32(N), Int32(K),
        alpha, A, Int32(lda),
        B, Int32(ldb),
        beta, &C, Int32(ldc)
    )
    #endif
}

func forecast_dgemv(
    _ Order: CBLAS_ORDER, _ TransA: CBLAS_TRANSPOSE,
    _ M: Int, _ N: Int,
    _ alpha: Double, _ A: [Double], _ lda: Int,
    _ X: [Double], _ incX: Int,
    _ beta: Double, _ Y: inout [Double], _ incY: Int
) {
    #if ACCELERATE_NEW_LAPACK
    cblas_dgemv(
        Order, TransA,
        M, N,
        alpha, A, lda,
        X, incX,
        beta, &Y, incY
    )
    #else
    cblas_dgemv(
        Order, TransA,
        Int32(M), Int32(N),
        alpha, A, Int32(lda),
        X, Int32(incX),
        beta, &Y, Int32(incY)
    )
    #endif
}

func dgesv_wrapper(
    _ n: UnsafeMutablePointer<LAPACKInteger>,
    _ nrhs: UnsafeMutablePointer<LAPACKInteger>,
    _ a: UnsafeMutablePointer<Double>,
    _ lda: UnsafeMutablePointer<LAPACKInteger>,
    _ ipiv: UnsafeMutablePointer<LAPACKInteger>,
    _ b: UnsafeMutablePointer<Double>,
    _ ldb: UnsafeMutablePointer<LAPACKInteger>,
    _ info: UnsafeMutablePointer<LAPACKInteger>
) {
    dgesv_(n, nrhs, a, lda, ipiv, b, ldb, info)
}

func dgels_wrapper(
    _ trans: UnsafeMutablePointer<Int8>,
    _ m: UnsafeMutablePointer<LAPACKInteger>,
    _ n: UnsafeMutablePointer<LAPACKInteger>,
    _ nrhs: UnsafeMutablePointer<LAPACKInteger>,
    _ a: UnsafeMutablePointer<Double>,
    _ lda: UnsafeMutablePointer<LAPACKInteger>,
    _ b: UnsafeMutablePointer<Double>,
    _ ldb: UnsafeMutablePointer<LAPACKInteger>,
    _ work: UnsafeMutablePointer<Double>,
    _ lwork: UnsafeMutablePointer<LAPACKInteger>,
    _ info: UnsafeMutablePointer<LAPACKInteger>
) {
    dgels_(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info)
}

func dpotrf_wrapper(
    _ uplo: UnsafeMutablePointer<Int8>,
    _ n: UnsafeMutablePointer<LAPACKInteger>,
    _ a: UnsafeMutablePointer<Double>,
    _ lda: UnsafeMutablePointer<LAPACKInteger>,
    _ info: UnsafeMutablePointer<LAPACKInteger>
) {
    dpotrf_(uplo, n, a, lda, info)
}

func dpotri_wrapper(
    _ uplo: UnsafeMutablePointer<Int8>,
    _ n: UnsafeMutablePointer<LAPACKInteger>,
    _ a: UnsafeMutablePointer<Double>,
    _ lda: UnsafeMutablePointer<LAPACKInteger>,
    _ info: UnsafeMutablePointer<LAPACKInteger>
) {
    dpotri_(uplo, n, a, lda, info)
}
