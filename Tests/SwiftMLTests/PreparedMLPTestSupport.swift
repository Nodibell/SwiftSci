import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftML

struct PreparedMLPTestContext {
    let source: PreparedNumericBatch
    let plan: StandardPreprocessingPlan
    let model: MLPRegressor
    let fitted: MLPCPUPredictor
}

func preparedMLPTestContext(width: Int = 7) async throws -> PreparedMLPTestContext {
    let rows = matrixMLPFeatures(8, width: width)
    let training = try matrixMLPBatch(rows, width: width)
    let source = try matrixMLPBatch(matrixMLPFeatures(1091, width: width), width: width)
    let plan = try StandardPreprocessingPlan(training: training)
    let model = MLPRegressor(hiddenLayerSizes: [9, 5], maxIter: 1, seed: 42, requestedDevice: .cpu)
    try await model.fit(features: rows, targets: rows.map { $0[0] })
    let fitted = try await model.makeCPUPredictor()
    return PreparedMLPTestContext(source: source, plan: plan, model: model, fitted: fitted)
}

func preparedMLPWaitUntil(_ predicate: () async -> Bool) async -> Bool {
    let deadline = ContinuousClock.now.advanced(by: .seconds(5))
    while ContinuousClock.now < deadline {
        if await predicate() { return true }
        await Task.yield()
    }
    return false
}
