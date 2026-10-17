import Testing
import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftML

@Suite("Indexed CPU prediction", .serialized)
struct MLPIndexedPredictionTests {
    @Test(arguments: [1, 7, 64, 128, 512, 513])
    func predictionsMatchGatheredInput(width: Int) async throws {
        let trainingRows = matrixMLPFeatures(8, width: width)
        let training = try matrixMLPBatch(trainingRows, width: width)
        let source = try matrixMLPBatch(matrixMLPFeatures(97, width: width), width: width)
        let plan = try StandardPreprocessingPlan(training: training)
        let model = MLPRegressor(hiddenLayerSizes: [9, 5], maxIter: 1, seed: 42, requestedDevice: .cpu)
        try await model.fit(features: trainingRows, targets: trainingRows.map { $0[0] })
        for count in [0, 1, 31, 32, 33, 65] {
            let indices = (0..<count).map { ($0 * 17) % 43 }
            var view = try IndexedNumericBatch(source, selectingRows: indices)
            let snapshot = view
            var gathered = try source.selectingRows(indices)
            if count > 0 {
                let rows = [0, count / 2, count - 1, count / 2]
                let values: [Double?] = [nil, -0.0, 42, .nan]
                try view.updateColumn(at: 0, rows: rows, values: values)
                try gathered.updateColumn(at: 0, rows: rows, values: values)
            }
            let expected = try await model.predict(features: gathered, preprocessing: plan)
            let actual = try await model.predict(features: view, preprocessing: plan)
            #expect(actual.allSatisfy { $0.isFinite })
            #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
            #expect(snapshot.editedColumns.isEmpty)
            #expect(view.originalRowIndices == gathered.originalRowIndices)
        }
    }

    @Test func cancellationAndRecovery() async throws {
        let rows = matrixMLPFeatures(65, width: 7)
        let source = try matrixMLPBatch(rows, width: 7)
        let view = try IndexedNumericBatch(source, selectingRows: Array((0..<65).reversed()))
        let plan = try StandardPreprocessingPlan(training: source)
        let model = MLPRegressor(hiddenLayerSizes: [9], maxIter: 1, seed: 42, requestedDevice: .cpu)
        await #expect(throws: SwiftMLError.self) { try await model.predict(features: view, preprocessing: plan) }
        try await model.fit(features: rows, targets: rows.map { $0[0] })
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await model.predict(features: view, preprocessing: plan)
        }
        await #expect(throws: CancellationError.self) { try await task.value }
        let expected = try await model.predict(features: source.selectingRows(Array((0..<65).reversed())), preprocessing: plan)
        #expect(try await model.predict(features: view, preprocessing: plan).map(\.bitPattern) == expected.map(\.bitPattern))
    }
}
