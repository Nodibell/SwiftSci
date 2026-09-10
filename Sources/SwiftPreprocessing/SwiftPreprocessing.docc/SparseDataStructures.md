# Memory-Efficient Sparse Matrix Representations

Drastically reduce RAM footprint and accelerate matrix-vector products using Compressed Sparse Row (CSR) and Column (CSC) formats.

## Overview

High-dimensional data transformations—including one-hot categorical encodings, n-gram bags of words, TF-IDF vectors, and interaction graphs—typically yield matrices where 95% to 99.9% of entries are zeros.

Storing such datasets as dense arrays requires gigabytes of memory and burns CPU cycles iterating over zeros. `SparseMatrix` implements Compressed Sparse Row (CSR) and Compressed Sparse Column (CSC) compression formats, reducing memory consumption by 10x–50x.

## Storage Formats

### Compressed Sparse Row (CSR)
Stores non-zero values row-by-row:
- `values`: Non-zero numerical entries.
- `colIndices`: Column indices corresponding to each entry in `values`.
- `rowPointers`: Offsets into `values` marking where each row begins (length: `rowCount + 1`).

```
Dense 3x3:
[ 10.0,  0.0,  0.0 ]
[  0.0, 20.0, 30.0 ]
[  0.0,  0.0, 40.0 ]

CSR Representation:
values:      [10.0, 20.0, 30.0, 40.0]
colIndices:  [0, 1, 2, 2]
rowPointers: [0, 1, 3, 4]
```

### Compressed Sparse Column (CSC)
Stores non-zero values column-by-column, optimal for column slicing and sparse-sparse matrix multiplication.

## Apple Accelerate Sparse BLAS

When multiplying large sparse matrices by dense vectors ($y = A \cdot x$), `SparseMatrix` delegates to Apple's native Accelerate Sparse BLAS C routines:
- `sparse_matrix_create_double`
- `sparse_insert_row_double`
- `sparse_commit`
- `sparse_matrix_vector_product_dense_double`

These routines utilize SIMD hardware instructions on Apple Silicon CPUs, delivering maximum throughput.

## Example Usage

```swift
import SwiftPreprocessing

// 1. Create a sparse matrix from a dense array
let dense: [[Double]] = [
    [1.0, 0.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 3.0]
]
let sparse = SparseMatrix(dense: dense, format: .csr)
print("Sparsity: \(sparse.sparsity)") // ~ 0.667

// 2. Accelerated matrix-vector product
let x = [1.0, 2.0, 3.0]
let y = try sparse.multiply(vector: x)
print(y) // [1.0, 4.0, 9.0]

// 3. Matrix Transpose & CSC conversion
let csc = sparse.toCSC()
let transposed = sparse.transpose()
```

## Topics

### Sparse Data Types
- ``SparseMatrix``
- ``SparseFormat``
