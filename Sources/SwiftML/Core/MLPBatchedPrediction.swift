import Accelerate
import SwiftDataFrame

let mlpPredictionBatchRows = 32

// Per-call scratch is capped at 256 KiB. Small requests and wider networks keep
// the row path; the measured crossover does not justify batching them.
func mlpBatchedPrediction<Result>(
    features: [[Double]], layers: [LayerWeights],
    activate: (Double) -> Double,
    finish: (UnsafeBufferPointer<Double>) -> Result
) throws -> [Result]? {
    guard features.count >= mlpPredictionBatchRows else { return nil }
    let width = layers.reduce(0) { max($0, max($1.inDim, $1.outDim)) }
    guard width > 0, width <= 512, let firstLayer = layers.first else { return nil }
    guard firstLayer.inDim > 0 else { throw SwiftMLError.emptyInput }
    for row in features where row.count != firstLayer.inDim {
        throw SwiftMLError.dimensionMismatch(expected: firstLayer.inDim, got: row.count)
    }
    return mlpBatchedPrediction(rowCount: features.count, width: width, layers: layers,
        fill: { start, count, buffer, stride in
            for row in 0..<count {
                features[start + row].withUnsafeBufferPointer { source in
                    (buffer.baseAddress! + row * stride).update(from: source.baseAddress!, count: firstLayer.inDim)
                }
            }
        }, activate: activate, finish: finish)
}

func mlpBatchedPrediction<Result>(
    matrix: PreparedNumericMatrix, layers: [LayerWeights],
    activate: (Double) -> Double,
    finish: (UnsafeBufferPointer<Double>) -> Result
) -> [Result]? {
    guard matrix.rowCount >= mlpPredictionBatchRows else { return nil }
    let width = layers.reduce(0) { max($0, max($1.inDim, $1.outDim)) }
    guard width > 0, width <= 512, !layers.isEmpty else { return nil }
    return matrix.values.withUnsafeBufferPointer { source in
        mlpBatchedPrediction(rowCount: matrix.rowCount, width: width, layers: layers,
            fill: { start, count, buffer, stride in
                switch matrix.order {
                case .rowMajor:
                    for row in 0..<count {
                        (buffer.baseAddress! + row * stride).update(
                            from: source.baseAddress! + (start + row) * matrix.columnCount,
                            count: matrix.columnCount)
                    }
                case .columnMajor:
                    for column in 0..<matrix.columnCount {
                        for row in 0..<count {
                            buffer[row * stride + column] = source[column * matrix.rowCount + start + row]
                        }
                    }
                }
            }, activate: activate, finish: finish)
    }
}

private func mlpBatchedPrediction<Result>(
    rowCount: Int, width: Int, layers: [LayerWeights],
    fill: (Int, Int, UnsafeMutableBufferPointer<Double>, Int) -> Void,
    activate: (Double) -> Double,
    finish: (UnsafeBufferPointer<Double>) -> Result
) -> [Result] {
    let lastLayer = layers.last!
    let batchRows = mlpPredictionBatchRows
    var first = [Double](repeating: 0, count: batchRows * width)
    var second = [Double](repeating: 0, count: batchRows * width)
    var results: [Result] = []
    results.reserveCapacity(rowCount)
    first.withUnsafeMutableBufferPointer { a in
        second.withUnsafeMutableBufferPointer { b in
            for start in stride(from: 0, to: rowCount, by: batchRows) {
                let count = min(batchRows, rowCount - start)
                fill(start, count, a, width)
                var input = a.baseAddress!, output = b.baseAddress!
                for (index, layer) in layers.enumerated() {
                    for row in 0..<count {
                        for col in 0..<layer.outDim { output[row * width + col] = layer.b[col] }
                    }
                    layer.W.withUnsafeBufferPointer { weights in
                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            Int32(count), Int32(layer.outDim), Int32(layer.inDim), 1,
                            input, Int32(width), weights.baseAddress!, Int32(layer.outDim),
                            1, output, Int32(width))
                    }
                    if index < layers.count - 1 {
                        for row in 0..<count {
                            for col in 0..<layer.outDim {
                                let offset = row * width + col
                                output[offset] = activate(output[offset])
                            }
                        }
                    }
                    swap(&input, &output)
                }
                for row in 0..<count {
                    let logits = UnsafeBufferPointer(start: input + row * width, count: lastLayer.outDim)
                    results.append(finish(logits))
                }
            }
        }
    }
    return results
}
