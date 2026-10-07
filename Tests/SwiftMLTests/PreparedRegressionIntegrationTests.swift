import Testing
import SwiftML
import SwiftPreprocessing

private enum LegacyCall: Error { case forbidden }
private struct PreparedOnlyTransformer: PreprocessingTransformer {
    mutating func fit(_ data: [[Double]]) throws { throw LegacyCall.forbidden }
    func transform(_ data: [[Double]]) throws -> [[Double]] { throw LegacyCall.forbidden }
    mutating func fit(_ data: PreparedNumericBatch) throws {}
    func transform(_ data: PreparedNumericBatch) throws -> PreparedNumericBatch { data }
}
private actor PreparedOnlyRegressor: RegressorEstimator {
    func fit(features: [[Double]], targets: [Double]) async throws { throw LegacyCall.forbidden }
    func predict(features: [[Double]]) async throws -> [Double] { throw LegacyCall.forbidden }
    func fit(features: PreparedNumericBatch, targets: [Double]) async throws {
        #expect((0..<features.rowCount).map { features[$0,0]! * 2 } == targets)
    }
    func predict(features: PreparedNumericBatch) async throws -> [Double] {
        (0..<features.rowCount).map { features[$0,0]! * 2 }
    }
}
private actor LegacyRegressor: RegressorEstimator {
    func fit(features: [[Double]], targets: [Double]) async throws {}
    func predict(features: [[Double]]) async throws -> [Double] { features.map { $0[0] } }
}

@Suite("Prepared regression integration")
struct PreparedRegressionIntegrationTests {
    @Test func dispatchAndAlignedSelection() async throws {
        var batch = try PreparedNumericBatch(columnNames:["x","target"],columns:[[1,2,3],[2,4,6]])
        let snapshot = try PreparedSupervisedBatch(batch,targetColumn:"target").selectingRows([2,0,2])
        try batch.updateColumn(at:0,rows:[2],values:[99])
        let pipeline: any RegressorEstimator = RegressionPipeline(transformers:[PreparedOnlyTransformer()],estimator:PreparedOnlyRegressor())
        let budget = try MemoryBudget(limit:4096), estimate = try MemoryEstimate(capacities:[1024,1024])
        try await pipeline.fit(snapshot,budget:budget,estimate:estimate)
        let prediction = try await pipeline.predict(features:snapshot.features,budget:budget,estimate:estimate)
        #expect(prediction == [6,2,6])
        #expect(snapshot.features.originalRowIndices == [2,0,2])
        #expect(await budget.reservedBytes == 0)
    }

    @Test func legacyConformersRemainUsable() async throws {
        let batch = try PreparedNumericBatch(columnNames:["x"],columns:[[1,2,3]])
        let pipeline = RegressionPipeline(transformers:[MinMaxScaler()],estimator:LegacyRegressor())
        try await pipeline.fit(features:batch,targets:[1,2,3])
        #expect(try await pipeline.predict(features:batch) == [0,0.5,1])
    }

    @Test func exactCPUAgreementAndSchema() async throws {
        let rows = (0..<128).map { i in [Double(i%7),Double((i*13)%17)] }
        let targets = rows.enumerated().map { i,row in row[0]*2-row[1]*3+7+Double(i%5) }
        let batch = try PreparedNumericBatch(columnNames:["a","b"],columns:[rows.map{$0[0]},rows.map{$0[1]}])
        let model = LinearRegression(device:.cpu), reference = LinearRegression(device:.cpu)
        let pipeline = RegressionPipeline(transformers:[StandardScaler()],estimator:model)
        let legacy = RegressionPipeline(transformers:[StandardScaler()],estimator:reference)
        try await pipeline.fit(features:batch,targets:targets)
        try await legacy.fit(features:rows,targets:targets)
        let actual = try await pipeline.predict(features:batch)
        let expected = try await legacy.predict(features:rows)
        #expect(actual == expected)
        let a = await model.getWeightsAndBias(), b = await reference.getWeightsAndBias()
        #expect(a.weights == b.weights && a.bias == b.bias)
        let reversed = try PreparedNumericBatch(columnNames:["b","a"],columns:[rows.map{$0[1]},rows.map{$0[0]}])
        await #expect(throws:(any Error).self) { _ = try await pipeline.predict(features:reversed) }
    }

    @Test func rejectedRefitPreservesSchema() async throws {
        let batch = try PreparedNumericBatch(columnNames:["a","b"],columns:[[1,2,3,4],[4,1,3,2]])
        let reversed = try PreparedNumericBatch(columnNames:["b","a"],columns:[[4,1,3,2],[1,2,3,4]])
        let targets = [6.0,5,9,10]
        let model = LinearRegression(device:.cpu)
        let pipeline = RegressionPipeline(transformers:[StandardScaler()],estimator:model)
        try await pipeline.fit(features:batch,targets:targets)
        let original = try await pipeline.predict(features:batch)
        await #expect(throws:(any Error).self) { try await pipeline.fit(features:batch,targets:[]) }
        #expect(try await pipeline.predict(features:batch) == original)
        await #expect(throws:(any Error).self) { _ = try await pipeline.predict(features:reversed) }
        await #expect(throws:(any Error).self) { _ = try await model.predict(features:reversed) }
        let erased: any RegressorEstimator = model
        await #expect(throws:(any Error).self) { _ = try await erased.predict(features:reversed) }
        await #expect(throws:(any Error).self) { try await pipeline.fit(features:batch.rowValues(),targets:[]) }
        // Transformers may have changed before the estimator rejects a refit.
        // Both prediction interfaces must stay unavailable until another fit succeeds.
        await #expect(throws:(any Error).self) { _ = try await pipeline.predict(features:batch) }
        await #expect(throws:(any Error).self) { _ = try await pipeline.predict(features:batch.rowValues()) }
        try await pipeline.fit(features:batch,targets:targets)
        #expect(try await pipeline.predict(features:batch) == original)
    }

    @Test func legacyInterceptOnlyAndNaNPrediction() async throws {
        let rows: [[Double]] = [[],[],[],[]]
        let model = LinearRegression(device:.cpu)
        try await model.fit(features:rows,targets:[1,3,5,7])
        let predicted = try await model.predict(features:rows)
        #expect(predicted.allSatisfy { abs($0-4) < 1e-12 })
        try await model.fitCPUGradientDescent(features:rows,targets:[1,3,5,7],learningRate:0.1,epochs:200)
        #expect(try await model.predict(features:rows).allSatisfy { abs($0-4) < 1e-12 })
        let pretrained = LinearRegression(weights:[1],bias:0,device:.cpu)
        #expect(try await pretrained.predict(features:[[.nan]])[0].isNaN)
    }

    @Test func budgetReleasesOnInvalidInput() async throws {
        let budget = try MemoryBudget(limit:100), estimate = try MemoryEstimate(capacities:[100])
        let model: any RegressorEstimator = LinearRegression(device:.cpu)
        let invalid = try PreparedNumericBatch(columnNames:["x"],columns:[[1,.nan]])
        await #expect(throws:(any Error).self) { try await model.fit(features:invalid,targets:[1,2],budget:budget,estimate:estimate) }
        #expect(await budget.reservedBytes == 0)
        let valid = try PreparedNumericBatch(columnNames:["x"],columns:[[1,2]])
        await #expect(throws:MemoryAdmissionError.exceedsLimit(required:101,limit:100)) {
            try await model.fit(features:valid,targets:[1,2],budget:budget,estimate:MemoryEstimate(capacities:[101]))
        }
        #expect(await budget.reservedBytes == 0)
    }
}
