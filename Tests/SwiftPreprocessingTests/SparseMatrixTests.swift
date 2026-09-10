import Testing
import Foundation
import SwiftML
@testable import SwiftPreprocessing

@Suite("SparseMatrix Tests")
struct SparseMatrixTests {

    @Test("SparseMatrix creation from dense array, nnz and sparsity")
    func testDenseToSparseConversion() {
        let dense: [[Double]] = [
            [1.0, 0.0, 0.0, 4.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0]
        ]

        let csr = SparseMatrix(dense: dense, format: .csr)
        #expect(csr.rowCount == 3)
        #expect(csr.colCount == 4)
        #expect(csr.nnz == 4)
        #expect(abs(csr.sparsity - (8.0 / 12.0)) < 1e-6)

        #expect(csr[0, 0] == 1.0)
        #expect(csr[0, 1] == 0.0)
        #expect(csr[0, 3] == 4.0)
        #expect(csr[1, 1] == 2.0)
        #expect(csr[2, 2] == 3.0)
        #expect(csr[2, 3] == 0.0)

        let recoveredDense = csr.toDense()
        #expect(recoveredDense == dense)
    }

    @Test("SparseMatrix CSC format conversion")
    func testCSCConversion() {
        let dense: [[Double]] = [
            [5.0, 0.0],
            [0.0, 8.0],
            [3.0, 0.0]
        ]

        let csr = SparseMatrix(dense: dense, format: .csr)
        let csc = csr.toCSC()
        #expect(csc.format == .csc)
        #expect(csc.rowCount == 3)
        #expect(csc.colCount == 2)
        #expect(csc.nnz == 3)

        #expect(csc[0, 0] == 5.0)
        #expect(csc[1, 1] == 8.0)
        #expect(csc[2, 0] == 3.0)
        #expect(csc[0, 1] == 0.0)

        let recovered = csc.toDense()
        #expect(recovered == dense)
    }

    @Test("SparseMatrix matrix-vector multiplication")
    func testMatrixVectorMultiplication() throws {
        let dense: [[Double]] = [
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [4.0, 0.0, 5.0]
        ]

        let csr = SparseMatrix(dense: dense, format: .csr)
        let vec: [Double] = [2.0, 3.0, 4.0]

        // Expected:
        // [1*2 + 2*3 + 0*4] = 8.0
        // [0*2 + 0*3 + 3*4] = 12.0
        // [4*2 + 0*3 + 5*4] = 28.0
        let result = try csr.multiply(vector: vec)
        #expect(result.count == 3)
        #expect(result[0] == 8.0)
        #expect(result[1] == 12.0)
        #expect(result[2] == 28.0)

        // Dimension mismatch
        #expect(throws: SwiftMLError.self) {
            _ = try csr.multiply(vector: [1.0, 2.0])
        }
    }

    @Test("SparseMatrix matrix-matrix multiplication")
    func testMatrixMatrixMultiplication() throws {
        let A_dense: [[Double]] = [
            [1.0, 0.0],
            [0.0, 2.0]
        ]
        let B_dense: [[Double]] = [
            [3.0, 4.0],
            [0.0, 5.0]
        ]

        let A = SparseMatrix(dense: A_dense)
        let B = SparseMatrix(dense: B_dense)

        // C = [[3.0, 4.0], [0.0, 10.0]]
        let C = try A.multiply(matrix: B)
        #expect(C.rowCount == 2)
        #expect(C.colCount == 2)
        #expect(C[0, 0] == 3.0)
        #expect(C[0, 1] == 4.0)
        #expect(C[1, 0] == 0.0)
        #expect(C[1, 1] == 10.0)
    }

    @Test("SparseMatrix transpose")
    func testTranspose() {
        let dense: [[Double]] = [
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 4.0]
        ]

        let csr = SparseMatrix(dense: dense)
        let transposed = csr.transpose()
        #expect(transposed.rowCount == 3)
        #expect(transposed.colCount == 2)
        #expect(transposed[0, 0] == 1.0)
        #expect(transposed[1, 0] == 2.0)
        #expect(transposed[2, 0] == 3.0)
        #expect(transposed[2, 1] == 4.0)
        #expect(transposed[0, 1] == 0.0)
    }

    @Test("Large SparseMatrix Accelerate Sparse BLAS offload")
    func testLargeAccelerateSparseBLAS() throws {
        let rows = 100
        let cols = 50
        var dense = Array(repeating: [Double](repeating: 0.0, count: cols), count: rows)

        // Diagonal-like pattern with 200 non-zeros
        for r in 0..<rows {
            let c1 = r % cols
            let c2 = (r + 1) % cols
            dense[r][c1] = Double(r + 1)
            dense[r][c2] = Double(c2 + 1)
        }

        let sparse = SparseMatrix(dense: dense)
        #expect(sparse.nnz >= 128)
        #expect(sparse.rowCount >= 64)

        let x = (0..<cols).map { Double($0 + 1) }
        let y = try sparse.multiply(vector: x)
        #expect(y.count == rows)

        // Verify first few rows against dense math
        for r in 0..<5 {
            var expected = 0.0
            for c in 0..<cols {
                expected += dense[r][c] * x[c]
            }
            #expect(abs(y[r] - expected) < 1e-9)
        }
    }
}
