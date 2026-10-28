#if os(macOS)
import Testing
import Foundation
import SwiftDataFrame
import SwiftPreprocessing
import SwiftCluster
import SwiftWorkflow

@Suite("Fitted workflow persistence", .serialized)
struct WorkflowPersistenceTests {
    private func fixture() throws -> PreparedSupervisedBatch {
        var batch = try PreparedNumericBatch(columnNames: ["signal", "companion", "constant", "class"], columns: [
            [-4, -3, -2, -1, 1, 2, 3, 4, -2.5, 2.5, 0.5, -0.5],
            [-7, -5, .nan, -1, 2, 3, 6, 8, -4.5, 5, .nan, 1],
            Array(repeating: 7, count: 12), [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0]])
        try batch.updateColumn(at: 1, rows: [2], values: [nil])
        return try PreparedSupervisedBatch(batch, targetColumn: "class")
    }

    @Test func roundTripPreservesParametersPredictionsAndRows() async throws {
        let source = try fixture()
        let original = source.features.columnValues(at: 1).map { $0?.bitPattern }
        let training = try source.selectingRows(Array(0..<8))
        let query = try source.selectingRows([11, 10, 9, 8, 10])
        let workflow = try await PCAClassificationWorkflow.train(training: training)
        let before = try await workflow.predict(query.features)
        #expect(before.sourceRowIndices == [11, 10, 9, 8, 10])
        #expect(before.labels == [0, 1, 1, 0, 1])
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: file) }
        try workflow.save(to: file)
        let bytes = try Data(contentsOf: file)
        try workflow.save(to: file)
        #expect(try Data(contentsOf: file) == bytes)
        let loaded = try PCAClassificationWorkflow.load(from: file)
        let after = try await loaded.predict(query.features)
        #expect(after.sourceRowIndices == before.sourceRowIndices)
        #expect(after.labels == before.labels)
        #expect(after.probabilities == before.probabilities)
        #expect(loaded.featureNames == ["signal", "companion", "constant"])
        #expect(source.features.columnValues(at: 1).map { $0?.bitPattern } == original)
        let empty = try source.features.selectingRows([])
        #expect(try await loaded.predict(empty).labels.isEmpty)
    }

    @Test func rejectsIncompatiblePreparedSchemasAndInfinity() async throws {
        let source = try fixture()
        let workflow = try await PCAClassificationWorkflow.train(training: source.selectingRows(Array(0..<8)))
        let wrong = try PreparedNumericBatch(columnNames: ["companion", "signal", "constant"], columns: [[1], [2], [7]])
        await #expect(throws: SwiftMLError.self) { _ = try await workflow.predict(wrong) }
        let infinite = try PreparedNumericBatch(columnNames: workflow.featureNames, columns: [[.infinity], [2], [7]])
        await #expect(throws: SwiftMLError.self) { _ = try await workflow.predict(infinite) }
    }

    @Test func rejectsUnknownVersionsAndCorruptConnections() async throws {
        let source = try fixture()
        let workflow = try await PCAClassificationWorkflow.train(training: source.selectingRows(Array(0..<8)))
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: file) }
        try workflow.save(to: file)
        let valid = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
        #expect(valid["trainingTargets"] == nil && valid["rawRows"] == nil)
        let mutations: [(String, Any)] = [("version", 99), ("weights", [1.0]),
            ("scalingStd", [1.0, 0.0, 1.0]), ("featureNames", ["signal", "signal", "constant"]),
            ("pcaComponents", [[1.0, 0.0]]), ("pcaTotalVariance", -1.0)]
        for (key, value) in mutations {
            var bad = valid
            bad[key] = value
            try JSONSerialization.data(withJSONObject: bad).write(to: file)
            #expect(throws: SwiftMLError.self) { _ = try PCAClassificationWorkflow.load(from: file) }
        }
        try Data("{\"version\":99}".utf8).write(to: file)
        #expect(throws: SwiftMLError.self) { _ = try PCAClassificationWorkflow.load(from: file) }
    }

    @Test func restoresPreprocessingAndCompletePCAProjection() async throws {
        let imputer = try Imputer(statistics: [2, 7])
        #expect(try imputer.transform([[.nan, 7]]) == [[2, 7]])
        let scaler = try StandardScaler(mean: [2, 7], std: [2, 1])
        #expect(try scaler.transform([[4, 7]]) == [[1, 0]])
        #expect(throws: SwiftMLError.self) { _ = try StandardScaler(mean: [1], std: [0]) }
        #expect(throws: SwiftMLError.self) { _ = try Imputer(statistics: [.nan]) }
        let features = try PreparedNumericBatch(columnNames: ["a", "a"], columns: [[16, 4, 14, 6], [6, -10, -5, 1]])
        let pca = try PCA(nComponents: 1, svdSolver: .full, device: .cpu)
        let expected = try await pca.fitTransform(features)
        let parameters = try await pca.parameters()
        let restored = PCA(parameters: parameters)
        let actual = try await restored.transform(features)
        #expect(actual.rowValues() == expected.rowValues())
        let restoredRatio = await restored.explainedVarianceRatio
        let expectedRatio = await pca.explainedVarianceRatio
        #expect(restoredRatio == expectedRatio)
        #expect(parameters.totalVariance > parameters.explainedVariance[0])
        #expect(throws: SwiftMLError.self) {
            _ = try PCA.Parameters(mean: [0], components: [[.nan]], explainedVariance: [1], totalVariance: 1)
        }
    }
}
#endif
