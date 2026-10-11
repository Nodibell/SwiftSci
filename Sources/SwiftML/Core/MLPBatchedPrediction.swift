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
    guard width > 0, width <= 512, let firstLayer = layers.first,
          let lastLayer = layers.last else { return nil }
    for row in features where row.count != firstLayer.inDim {
        throw SwiftMLError.dimensionMismatch(expected: firstLayer.inDim, got: row.count)
    }
    let batchRows = mlpPredictionBatchRows
    var first = [Double](repeating: 0, count: batchRows * width)
    var second = [Double](repeating: 0, count: batchRows * width)
    var results: [Result] = []
    results.reserveCapacity(features.count)
    first.withUnsafeMutableBufferPointer { a in
        second.withUnsafeMutableBufferPointer { b in
            for start in stride(from: 0, to: features.count, by: batchRows) {
                let count = min(batchRows, features.count - start)
                for row in 0..<count {
                    features[start + row].withUnsafeBufferPointer { source in
                        (a.baseAddress! + row * width).update(from: source.baseAddress!, count: firstLayer.inDim)
                    }
                }
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
