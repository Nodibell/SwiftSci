import Testing
import SwiftDataFrame
@testable import SwiftML

@Suite("Compact MLP input boundaries")
struct MLPCompactBatchBoundaryTests {
    @Test func fittedAndFiniteChecks() async throws {
        let model = MLPRegressor(hiddenLayerSizes: [], maxIter: 1, seed: 42, requestedDevice: .cpu)
        let empty = try matrixMLPBatch([], width: 2)
        let erased: any RegressorEstimator = model
        await #expect(throws: SwiftMLError.modelNotFitted) { try await erased.predict(features: empty) }
        let rows: [[Double]] = [[1, 2], [3, 4]]
        try await model.fit(features: rows, targets: [1, 2])
        #expect(try await erased.predict(features: empty).isEmpty)
        for value in [Double.nan, .infinity, -.infinity] {
            let batch = try PreparedNumericBatch(columnNames: ["x0", "x1"], columns: [[value], [2]])
            await #expect(throws: SwiftMLError.self) { try await erased.predict(features: batch) }
        }
        var missing = try matrixMLPBatch(rows, width: 2)
        try missing.updateColumn(at: 0, rows: [0], values: [nil])
        await #expect(throws: SwiftMLError.self) { try await erased.predict(features: missing) }
    }
    @Test func rejectsWrongNonemptyWidth() async throws {
        let model = MLPRegressor(hiddenLayerSizes: [], maxIter: 1, seed: 42, requestedDevice: .cpu)
        try await model.fit(features: [[1, 2], [3, 4]], targets: [1, 2])
        for count in [1, 33] {
            let bad = try matrixMLPBatch(matrixMLPFeatures(count, width: 3), width: 3)
            await #expect(throws: SwiftMLError.dimensionMismatch(expected: 2, got: 3)) {
                try await model.predict(features: bad)
            }
        }
    }
}
