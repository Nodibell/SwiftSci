import Foundation
import Accelerate
typealias LAPACKInteger = Int32

// MARK: - BLAS

@inline(__always)
func forecast_dgemm(
    _ order: CBLAS_ORDER,
    _ transA: CBLAS_TRANSPOSE,
    _ transB: CBLAS_TRANSPOSE,
    _ m: Int,
    _ n: Int,
    _ k: Int,
    _ alpha: Double,
    _ A: [Double],
    _ lda: Int,
    _ B: [Double],
    _ ldb: Int,
    _ beta: Double,
    _ C: inout [Double],
    _ ldc: Int
) {
    cblas_dgemm(
        order,
        transA,
        transB,
        Int32(m),
        Int32(n),
        Int32(k),
        alpha,
        A,
        Int32(lda),
        B,
        Int32(ldb),
        beta,
        &C,
        Int32(ldc)
    )
}

@inline(__always)
func forecast_dgemv(
    _ order: CBLAS_ORDER,
    _ trans: CBLAS_TRANSPOSE,
    _ m: Int,
    _ n: Int,
    _ alpha: Double,
    _ A: [Double],
    _ lda: Int,
    _ X: [Double],
    _ incX: Int,
    _ beta: Double,
    _ Y: inout [Double],
    _ incY: Int
) {
    cblas_dgemv(
        order,
        trans,
        Int32(m),
        Int32(n),
        alpha,
        A,
        Int32(lda),
        X,
        Int32(incX),
        beta,
        &Y,
        Int32(incY)
    )
}

// MARK: - LAPACK

@inline(__always)
func dgesv_wrapper(
    _ n: inout LAPACKInteger,
    _ nrhs: inout LAPACKInteger,
    _ a: UnsafeMutablePointer<Double>,
    _ lda: inout LAPACKInteger,
    _ ipiv: UnsafeMutablePointer<LAPACKInteger>,
    _ b: UnsafeMutablePointer<Double>,
    _ ldb: inout LAPACKInteger,
    _ info: inout LAPACKInteger
) {
    dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info)
}

@inline(__always)
func dgels_wrapper(
    _ trans: UnsafeMutablePointer<Int8>,
    _ m: inout LAPACKInteger,
    _ n: inout LAPACKInteger,
    _ nrhs: inout LAPACKInteger,
    _ a: UnsafeMutablePointer<Double>,
    _ lda: inout LAPACKInteger,
    _ b: UnsafeMutablePointer<Double>,
    _ ldb: inout LAPACKInteger,
    _ work: UnsafeMutablePointer<Double>,
    _ lwork: inout LAPACKInteger,
    _ info: inout LAPACKInteger
) {
    dgels_(trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info)
}

@inline(__always)
func dpotrf_wrapper(
    _ uplo: UnsafeMutablePointer<Int8>,
    _ n: inout LAPACKInteger,
    _ a: UnsafeMutablePointer<Double>,
    _ lda: inout LAPACKInteger,
    _ info: inout LAPACKInteger
) {
    dpotrf_(uplo, &n, a, &lda, &info)
}

@inline(__always)
func dpotri_wrapper(
    _ uplo: UnsafeMutablePointer<Int8>,
    _ n: inout LAPACKInteger,
    _ a: UnsafeMutablePointer<Double>,
    _ lda: inout LAPACKInteger,
    _ info: inout LAPACKInteger
) {
    dpotri_(uplo, &n, a, &lda, &info)
}
