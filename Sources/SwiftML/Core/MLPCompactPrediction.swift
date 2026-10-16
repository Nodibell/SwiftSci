import SwiftDataFrame

func mlpCompactPrediction(features: PreparedNumericBatch, layers: [LayerWeights],
                          activate: (Double) -> Double) -> [Double] {
    let width = layers.reduce(0) { max($0, max($1.inDim, $1.outDim)) }
    if features.rowCount >= mlpPredictionBatchRows && width > 0 && width <= 512 {
        return mlpBatchedPrediction(rowCount: features.rowCount, width: width, layers: layers,
            fill: { start, count, buffer, stride in
                for column in features.columns.indices {
                    features.columns[column].values.withUnsafeBufferPointer { source in
                        for row in 0..<count {
                            buffer[row * stride + column] = source[start + row]
                        }
                    }
                }
            }, activate: activate, finish: { $0[0] })
    }
    var input = [Double](repeating: 0, count: features.columnCount)
    var result: [Double] = []
    result.reserveCapacity(features.rowCount)
    for row in 0..<features.rowCount {
        for column in features.columns.indices { input[column] = features.columns[column].values[row] }
        result.append(mlpPredictRow(input, layers: layers, activate: activate)[0])
    }
    return result
}
