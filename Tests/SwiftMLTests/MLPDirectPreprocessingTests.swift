import Testing
import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftML

@Suite("Direct preprocessing MLP trial", .serialized)
struct MLPDirectPreprocessingTests {
    @Test(arguments: [1, 7, 64, 129, 512, 513])
    func bitwisePredictionsAndSnapshots(width: Int) async throws {
        let rows = matrixMLPFeatures(8, width: width)
        let training = try matrixMLPBatch(rows, width: width)
        let plan = try StandardPreprocessingPlan(training: training)
        let model = MLPRegressor(hiddenLayerSizes: [9, 5], maxIter: 1, seed: 42, requestedDevice: .cpu)
        try await model.fit(features: rows, targets: rows.map { $0[0] })
        for count in [0, 1, 31, 32, 33, 65] {
            let columns: [[Double]] = (0..<width).map { c -> [Double] in
                (0..<count).map { r -> Double in
                    let missing: Bool = (r + c) % 11 == 0
                    let value: Double = Double(r - c) / 16
                    return missing ? .nan : value
                }
            }
            let input = try PreparedNumericBatch(columnNames: training.columnNames, columns: columns)
            let snapshot = input
            let bits = snapshot.columns.map { $0.values.map(\.bitPattern) }
            let matrix = try plan.preparedMatrix(input)
            let expected = try await model.predict(features: matrix)
            let actual = try await model.predict(features: input, preprocessing: plan)
            #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
            #expect(snapshot.columns.map { $0.values.map(\.bitPattern) } == bits)
        }
    }

    @Test func cancellationAndRecovery() async throws {
        let rows = matrixMLPFeatures(65, width: 7)
        let input = try matrixMLPBatch(rows, width: 7)
        let plan = try StandardPreprocessingPlan(training: input)
        let model = MLPRegressor(hiddenLayerSizes: [9], maxIter: 1, seed: 42, requestedDevice: .cpu)
        await #expect(throws: SwiftMLError.self) { try await model.predict(features: input, preprocessing: plan) }
        try await model.fit(features: rows, targets: rows.map { $0[0] })
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await model.predict(features: input, preprocessing: plan)
        }
        await #expect(throws: CancellationError.self) { try await task.value }
        let expected = try await model.predict(features: plan.preparedMatrix(input))
        #expect(try await model.predict(features: input, preprocessing: plan).map(\.bitPattern)
                == expected.map(\.bitPattern))
    }

    @Test func fillErrorStopsBatchLoop() throws {
        enum Stop: Error { case requested }
        let layers = [LayerWeights(W: [1], b: [0], inDim: 1, outDim: 1)]
        var calls = 0
        #expect(throws: Stop.self) {
            try mlpBatchedPrediction(rowCount: 65, width: 1, layers: layers,
                fill: { _, count, buffer, _ in
                    calls += 1
                    if calls == 2 { throw Stop.requested }
                    for r in 0..<count { buffer[r] = 1 }
                }, activate: { $0 }, finish: { $0[0] })
        }
        #expect(calls == 2)
    }
}
