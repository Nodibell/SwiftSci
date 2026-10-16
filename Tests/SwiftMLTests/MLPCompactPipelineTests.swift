import Testing
import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftML

@Suite("Compact MLP regression pipeline integration")
struct MLPCompactPipelineTests {
    @Test func fittedPipelineSelectionAndRefit() async throws {
        let rows = matrixMLPFeatures(65, width: 7)
        var source = try matrixMLPBatch(rows, width: 7)
        let targets: [Double] = rows.map { $0[0] - $0[1] }
        let native = MLPRegressor(hiddenLayerSizes: [9], maxIter: 2, seed: 42, requestedDevice: .cpu)
        let reference = MLPRegressor(hiddenLayerSizes: [9], maxIter: 2, seed: 42, requestedDevice: .cpu)
        let pipeline = RegressionPipeline(transformers: [StandardScaler()], estimator: native)
        let legacy = RegressionPipeline(transformers: [StandardScaler()], estimator: reference)
        try await pipeline.fit(features: source, targets: targets)
        try await legacy.fit(features: source, targets: targets)
        let positions = [64, 2, 2, 0] + Array(0..<33)
        let selected = try source.selectingRows(positions)
        let expected = try await legacy.predict(features: selected.rowValues())
        try source.updateColumn(at: 0, rows: [2], values: [99])
        let actual = try await pipeline.predict(features: selected)
        #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
        #expect(selected.originalRowIndices == positions)
        let names: [String] = Array(selected.columnNames.reversed())
        let columns: [[Double]] = (0..<7).reversed().map { column -> [Double] in
            positions.map { position -> Double in rows[position][column] }
        }
        let reversed = try PreparedNumericBatch(columnNames: names, columns: columns)
        await #expect(throws: SwiftMLError.self) { try await pipeline.predict(features: reversed) }
        let newTargets: [Double] = targets.map { -$0 + 1 }
        try await pipeline.fit(features: selected, targets: positions.map { newTargets[$0] })
        let revised = try await pipeline.predict(features: selected)
        #expect(revised != actual)
        let direct = try await native.predict(features: try pipeline.transformers[0].transform(selected))
        #expect(revised.map(\.bitPattern) == direct.map(\.bitPattern))
    }
}
