import Testing
import Accelerate
import SwiftDataFrame
@testable import SwiftPreprocessing

@Suite("Prepared numerical pipeline")
struct PreparedPipelineTests {
    @Test func selectionScaleAndMultiplyPreserveRowIdentity() throws {
        let frame = try DataFrame(columns: [TypedColumn<Double>(name: "x", values: [7, 1, 9, 3, 5]),
                                            TypedColumn<Double>(name: "y", values: [14, 3, 4, 8, 2])])
        // A filter selection followed by a stable sort permutation, with a repeated row.
        let indices = [1, 3, 3, 4, 0]
        let source = try frame.prepareNumericBatch(["x","y"]).selectingRows(indices)
        var scaler = StandardScaler(), reference = StandardScaler()
        let transformed = try scaler.fitTransform(source)
        let expected = try reference.fitTransform(frame.gathered(at: indices).toFeatureMatrix(["x","y"]))
        let weights = [2.0, -3.0]
        for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
            let matrix = try transformed.matrix(order: order, missing: .reject)
            var prediction = [Double](repeating: 0, count: indices.count)
            cblas_dgemv(order == .rowMajor ? CblasRowMajor : CblasColMajor, CblasNoTrans,
                        Int32(matrix.rowCount), Int32(matrix.columnCount), 1, matrix.values,
                        Int32(order == .rowMajor ? matrix.columnCount : matrix.rowCount), weights, 1, 0, &prediction, 1)
            for row in prediction.indices { #expect(abs(prediction[row] - (expected[row][0] * 2 - expected[row][1] * 3)) < 1e-12) }
            #expect(matrix.originalRowIndices == indices)
        }
        #expect(source.originalRowIndices == indices && source[0,0] == 1)
    }
}
