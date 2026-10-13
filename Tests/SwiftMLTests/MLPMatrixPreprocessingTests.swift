import Testing
import SwiftDataFrame
import SwiftPreprocessing
import SwiftML

@Suite("Fitted preprocessing through matrix prediction")
struct MLPMatrixPreprocessingTests {
    @Test func heldOutWorkflowKeepsTrainingStatisticsAndSelection() async throws {
        let training = matrixMLPFeatures(19, width: 7)
        var imputer = Imputer(), scaler = StandardScaler()
        try imputer.fit(training)
        let imputedTraining = try imputer.transform(training)
        try scaler.fit(imputedTraining)
        let preparedTraining = try scaler.transform(imputedTraining)
        let targets: [Double] = training.map { 0.5 * $0[0] + 1 }
        let model = MLPRegressor(hiddenLayerSizes: [9], maxIter: 2, seed: 42, requestedDevice: .cpu)
        try await model.fit(features: preparedTraining, targets: targets)
        var heldOut = matrixMLPFeatures(65, width: 7)
        for r in heldOut.indices { heldOut[r][0] += 7; heldOut[r][2] = .nan }
        let names: [String] = (0..<7).map { "x\($0)" }
        let columns: [TypedColumn<Double>] = (0..<7).map { c in
            let values: [Double] = heldOut.map { $0[c] }
            return TypedColumn<Double>(name: names[c], values: values)
        }
        let frame = try DataFrame(columns: columns.reversed())
        let selections: [[Int]] = [[], [64, 1, 1], [64, 2, 2] + Array(0..<33)]
        let means = scaler.mean, statistics = imputer.statistics
        for selected in selections {
            let referenceRows: [[Double]] = selected.map { heldOut[$0] }
            let expected = try await model.predict(features: scaler.transform(imputer.transform(referenceRows)))
            for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
                let raw = try frame.prepareNumericMatrix(names, rows: selected, order: order)
                let imputed = try imputer.transform(consuming: consume raw)
                let scaled = try scaler.transform(consuming: consume imputed)
                matrixMLPExpectClose(try await model.predict(features: scaled), expected)
                #expect(scaled.originalRowIndices == selected && scaled.columnNames == names)
            }
        }
        #expect(scaler.mean == means && imputer.statistics == statistics)
    }
}
