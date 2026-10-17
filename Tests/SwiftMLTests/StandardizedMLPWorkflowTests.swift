import Testing
import SwiftML
import SwiftPreprocessing
import SwiftDataFrame

@Suite("Public standardized MLP workflow")
struct StandardizedMLPWorkflowTests {
    @Test func trainingOnlyStatisticsAndPublicReuse() async throws {
        let training = try publicMLPTraining()
        let model = publicMLPModel()
        let fitted = try await model.fitStandardized(features: training.batch, targets: training.targets)
        let reference = try await publicMLPReference(training.batch, targets: training.targets)
        let budget = try MemoryBudget(limit: 16 * 1024 * 1024)
        #expect(fitted.columnNames == training.batch.columnNames)
        #expect(fitted.retainedBytes > 0)
        for count in [0, 1, 31, 32, 33, 65] {
            let rows: [[Double]] = matrixMLPFeatures(count, width: 7).map { row -> [Double] in
                row.map { value -> Double in value + 100 }
            }
            let batch = try matrixMLPBatch(rows, width: 7)
            let request = MLPPredictionRequest(source: batch)
            let quote = try fitted.preparationEstimate(for: request)
            let input = try await fitted.prepare(request, budget: budget)
            let expected = try await reference.predict(batch)
            let actual = try await fitted.predict(input, budget: budget)
            #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
            #expect(try await fitted.predict(input, budget: budget) == actual)
            #expect(input.rowCount == count && input.retainedBytes <= quote.bytes)
            #expect(input.preparationPeakBytes == quote.bytes)
        }
        #expect(await budget.reservedBytes == 0)
        #expect(await budget.queuedCount == 0)
    }
}
