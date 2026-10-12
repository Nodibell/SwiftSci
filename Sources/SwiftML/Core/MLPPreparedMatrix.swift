import Accelerate
import SwiftDataFrame

func mlpValidateMatrix(_ matrix: PreparedNumericMatrix, layers: [LayerWeights]) throws {
    let expected = layers[0].inDim
    guard matrix.columnCount == expected else {
        throw SwiftMLError.dimensionMismatch(expected: expected, got: matrix.columnCount)
    }
}

func mlpMatrixRow(_ matrix: PreparedNumericMatrix, row: Int) -> [Double] {
    switch matrix.order {
    case .rowMajor:
        let start = row * matrix.columnCount
        return Array(matrix.values[start..<(start + matrix.columnCount)])
    case .columnMajor:
        return (0..<matrix.columnCount).map { matrix.values[$0 * matrix.rowCount + row] }
    }
}

func mlpPredictRow(_ values: [Double], layers: [LayerWeights], activate: (Double) -> Double) -> [Double] {
    var current = values
    for index in layers.indices {
        var next = layers[index].b
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            1, Int32(layers[index].outDim), Int32(layers[index].inDim), 1,
            current, Int32(layers[index].inDim), layers[index].W, Int32(layers[index].outDim),
            1, &next, Int32(layers[index].outDim))
        if index < layers.count - 1 {
            for column in 0..<layers[index].outDim { next[column] = activate(next[column]) }
        }
        current = next
    }
    return current
}
