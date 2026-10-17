import SwiftDataFrame
import SwiftPreprocessing

func mlpPreprocessedPrediction<Result>(features: PreparedNumericBatch,
    preprocessing: StandardPreprocessingPlan, layers: [LayerWeights],
    activate: (Double) -> Double, finish: (UnsafeBufferPointer<Double>) -> Result
) throws -> [Result] {
    guard let first = layers.first else { throw SwiftMLError.modelNotFitted }
    guard features.columnCount == first.inDim else {
        throw SwiftMLError.dimensionMismatch(expected: first.inDim, got: features.columnCount)
    }
    let width = layers.reduce(0) { max($0, max($1.inDim, $1.outDim)) }
    let batched = features.rowCount >= mlpPredictionBatchRows && width > 0 && width <= 512
    var writer = try preprocessing.makeRowWriter(features, maximumRows: batched ? mlpPredictionBatchRows : 1)
    if batched {
        return try mlpBatchedPrediction(rowCount: features.rowCount, width: width, layers: layers,
            fill: { start, count, buffer, stride in
                try writer.fillDouble(rows: start..<(start + count), into: buffer, stride: stride)
            }, activate: activate, finish: finish)
    }
    var input = [Double](repeating: 0, count: features.columnCount)
    var results: [Result] = []
    results.reserveCapacity(features.rowCount)
    for row in 0..<features.rowCount {
        try input.withUnsafeMutableBufferPointer {
            try writer.fillDouble(rows: row..<(row + 1), into: $0, stride: features.columnCount)
        }
        let prediction = mlpPredictRow(input, layers: layers, activate: activate)
        results.append(prediction.withUnsafeBufferPointer(finish))
    }
    return results
}
