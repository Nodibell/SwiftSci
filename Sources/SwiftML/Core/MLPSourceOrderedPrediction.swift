import SwiftDataFrame
import SwiftPreprocessing

extension MLPRegressor {
    /// Orders independent CPU rows and restores logical occurrence order before return.
    package func predictSourceOrdered(features: IndexedNumericBatch,
                                      preprocessing: StandardPreprocessingPlan) async throws -> [Double] {
        try Task.checkCancellation()
        // Keep every row in the same BLAS batch-size class, including the original tail.
        let prefix = mlpSourceOrderPrefix(rowCount: features.rowCount)
        let ordered = try features.sourceOrdered(prefixCount: prefix)
        try Task.checkCancellation()
        let prediction = try await predict(features: ordered.batch, preprocessing: preprocessing)
        guard let positions = ordered.logicalPositions else { return prediction }
        var result = [Double](repeating: 0, count: prediction.count)
        for r in positions.indices { result[positions[r]] = prediction[r] }
        return result
    }
}

func mlpSourceOrderPrefix(rowCount: Int) -> Int {
    rowCount >= mlpPredictionBatchRows ? rowCount - rowCount % mlpPredictionBatchRows : rowCount
}
