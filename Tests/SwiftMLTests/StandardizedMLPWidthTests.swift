import Testing
import SwiftML
import SwiftPreprocessing
import SwiftDataFrame

@Suite("Public standardized MLP network width")
struct StandardizedMLPWidthTests {
    @Test(arguments: [512, 513])
    func batchingBoundaryMatchesPublicReference(_ width: Int) async throws {
        let rows: [[Double]] = matrixMLPFeatures(4, width: width)
        let training = try matrixMLPBatch(rows, width: width)
        let targets: [Double] = [1, 2, 3, 4]
        var pipeline = Pipeline(steps: [Imputer(), StandardScaler()])
        try pipeline.fit(training)
        let clean = try pipeline.transform(training)
        let model = MLPRegressor(hiddenLayerSizes: [], maxIter: 1, seed: 42, requestedDevice: .cpu)
        let reference = MLPRegressor(hiddenLayerSizes: [], maxIter: 1, seed: 42, requestedDevice: .cpu)
        let fitted = try await model.fitStandardized(features: training, targets: targets)
        try await reference.fit(features: clean.rowValues(), targets: targets)
        let source = try matrixMLPBatch(matrixMLPFeatures(65, width: width), width: width)
        let indices: [Int] = [64, 2, 2] + Array(0..<30)
        let selected = try source.selectingRows(indices)
        let expected = try await reference.predict(features: pipeline.transform(selected))
        let budget = try MemoryBudget(limit: 32 * 1024 * 1024)
        let request = try MLPPredictionRequest(source: source, selectingRows: indices)
        let input = try await fitted.prepare(request, budget: budget)
        let actual = try await fitted.predict(input, budget: budget)
        #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
        #expect(await budget.reservedBytes == 0)
    }
}
