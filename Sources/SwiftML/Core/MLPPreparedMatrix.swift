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
    for (index, layer) in layers.enumerated() {
        var next = layer.b
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            1, Int32(layer.outDim), Int32(layer.inDim), 1,
            current, Int32(layer.inDim), layer.W, Int32(layer.outDim),
            1, &next, Int32(layer.outDim))
        if index < layers.count - 1 {
            for column in 0..<layer.outDim { next[column] = activate(next[column]) }
        }
        current = next
    }
    return current
}
