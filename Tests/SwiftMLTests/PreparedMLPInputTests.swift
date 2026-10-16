import Testing
import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftML

@Suite("Reusable source-ordered MLP inputs", .serialized)
struct PreparedMLPInputTests {
    @Test(arguments: [1, 7, 128, 512, 513])
    func repeatedPredictionsPreserveExactLogicalOrder(width: Int) async throws {
        let c = try await preparedMLPTestContext(width: width)
        let budget = try MemoryBudget(limit: 128 * 1024 * 1024)
        for count in [0, 1, 31, 32, 33, 65, 1031] {
            let indices = (0..<count).map { ($0 * 17) % 43 }
            var view = try IndexedNumericBatch(c.source, selectingRows: indices)
            var gathered = try c.source.selectingRows(indices)
            if count > 0 {
                let rows = [0, count / 2, count - 1, count / 2]
                let values: [Double?] = [nil, -0.0, 42, .nan]
                for column in 0..<width where column == 0 || column == width - 1 {
                    try view.updateColumn(at: column, rows: rows, values: values)
                    try gathered.updateColumn(at: column, rows: rows, values: values)
                }
            }
            let expected = try await c.model.predict(features: gathered, preprocessing: c.plan)
            let input = try await PreparedMLPInput.prepare(features: view, preprocessing: c.plan, budget: budget)
            #expect(input.rowCount == count)
            #expect(input.retainedBytes <= input.preparationPeakBytes)
            #expect(await budget.reservedBytes == 0)
            for _ in 0..<3 {
                let callerManaged = try c.fitted.predictExternallyAdmitted(features: input)
                #expect(callerManaged.map(\.bitPattern) == expected.map(\.bitPattern))
                let result = try await c.model.predict(features: input, budget: budget)
                #expect(result.map(\.bitPattern) == expected.map(\.bitPattern))
                #expect(await budget.reservedBytes == 0)
            }
        }
    }

    @Test func laterEditsAndOwnerReleaseDoNotChangeThePreparedSnapshot() async throws {
        let c = try await preparedMLPTestContext()
        let budget = try MemoryBudget(limit: 128 * 1024 * 1024)
        var source = c.source
        var view = try IndexedNumericBatch(source, selectingRows: [64, 1, 64, 0, 33])
        try view.updateColumn(at: 0, rows: [0, 2], values: [-0.0, 42])
        let expected = try await c.model.predictSourceOrdered(features: view, preprocessing: c.plan)
        var input: PreparedMLPInput? = try await PreparedMLPInput.prepare(features: view, preprocessing: c.plan, budget: budget)
        weak var witness = input
        try view.updateColumn(at: 0, rows: [0, 1, 2], values: [nil, .nan, 999])
        try source.updateColumn(at: 1, rows: [64], values: [999])
        let actual = try await c.model.predict(features: input!, budget: budget)
        #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
        input = nil
        #expect(witness == nil)
        #expect(await budget.reservedBytes == 0)
    }
}
