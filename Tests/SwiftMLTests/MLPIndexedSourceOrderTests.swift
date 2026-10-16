import Testing
import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftML

@Suite("Source-ordered CPU prediction", .serialized)
struct MLPIndexedSourceOrderTests {
    @Test(arguments: [1, 7, 128, 512, 513])
    func predictionsRestoreLogicalOrderExactly(width: Int) async throws {
        let rows = matrixMLPFeatures(8, width: width)
        let training = try matrixMLPBatch(rows, width: width)
        let source = try matrixMLPBatch(matrixMLPFeatures(1091, width: width), width: width)
        let plan = try StandardPreprocessingPlan(training: training)
        let model = MLPRegressor(hiddenLayerSizes: [9, 5], maxIter: 1, seed: 42, requestedDevice: .cpu)
        try await model.fit(features: rows, targets: rows.map { $0[0] })
        for count in [0, 1, 31, 32, 33, 65, 1031] {
            for repeated in [false, true] {
                let indices = (0..<count).map { ($0 * 17) % (repeated ? 13 : 1091) }
                var view = try IndexedNumericBatch(source, selectingRows: indices)
                var gathered = try source.selectingRows(indices)
                if count > 0 {
                    let edited = [0, count / 2, count - 1, count / 2]
                    let values: [Double?] = [nil, -0.0, 42, .nan]
                    for c in 0..<width where c == 0 || c == width - 1 {
                        try view.updateColumn(at: c, rows: edited, values: values)
                        try gathered.updateColumn(at: c, rows: edited, values: values)
                    }
                }
                let expected = try await model.predict(features: gathered, preprocessing: plan)
                let actual = try await model.predictSourceOrdered(features: view, preprocessing: plan)
                #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
            }
        }
    }

    @Test func cancellationAndRecovery() async throws {
        let rows = matrixMLPFeatures(65, width: 7)
        let source = try matrixMLPBatch(rows, width: 7)
        let view = try IndexedNumericBatch(source, selectingRows: Array((0..<65).reversed()))
        let plan = try StandardPreprocessingPlan(training: source)
        let model = MLPRegressor(hiddenLayerSizes: [9], maxIter: 1, seed: 42, requestedDevice: .cpu)
        await #expect(throws: SwiftMLError.self) { try await model.predictSourceOrdered(features: view, preprocessing: plan) }
        try await model.fit(features: rows, targets: rows.map { $0[0] })
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await model.predictSourceOrdered(features: view, preprocessing: plan)
        }
        await #expect(throws: CancellationError.self) { try await task.value }
        let actual = try await model.predictSourceOrdered(features: view, preprocessing: plan)
        let expected = try await model.predict(features: view, preprocessing: plan)
        #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
    }
}
