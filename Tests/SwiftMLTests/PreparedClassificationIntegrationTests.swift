import Testing
import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftML

private enum PreparedClassificationFailure: Error {
    case legacyPath
    case unexpectedInput
}

private struct ConsumingPreparedIdentity: PreprocessingTransformer {
    var supportsNativePreparedBatches: Bool { true }

    mutating func fit(_ data: [[Double]]) throws {
        throw PreparedClassificationFailure.legacyPath
    }

    func transform(_ data: [[Double]]) throws -> [[Double]] {
        throw PreparedClassificationFailure.legacyPath
    }

    mutating func fit(_ data: PreparedNumericBatch) throws {}

    func transform(_ data: PreparedNumericBatch) throws -> PreparedNumericBatch {
        throw PreparedClassificationFailure.legacyPath
    }

    func transform(consuming data: consuming PreparedNumericBatch) throws -> PreparedNumericBatch {
        data
    }
}

private struct ReorderingPreparedTransformer: PreprocessingTransformer {
    var supportsNativePreparedBatches: Bool { true }

    mutating func fit(_ data: [[Double]]) throws {}
    func transform(_ data: [[Double]]) throws -> [[Double]] { Array(data.reversed()) }
    mutating func fit(_ data: PreparedNumericBatch) throws {}
    func transform(_ data: PreparedNumericBatch) throws -> PreparedNumericBatch {
        try data.selectingRows(Array((0..<data.rowCount).reversed()))
    }
}

private struct DroppingPreparedTransformer: PreprocessingTransformer {
    var supportsNativePreparedBatches: Bool { true }

    mutating func fit(_ data: [[Double]]) throws {}
    func transform(_ data: [[Double]]) throws -> [[Double]] { Array(data.dropLast()) }
    mutating func fit(_ data: PreparedNumericBatch) throws {}
    func transform(_ data: PreparedNumericBatch) throws -> PreparedNumericBatch {
        try data.selectingRows(Array(0..<max(0, data.rowCount - 1)))
    }
}

private actor InconsistentProbabilityClassifier: ClassifierEstimator {
    func fit(features: [[Double]], targets: [Double]) async throws {}
    func predict(features: [[Double]]) async throws -> [Int] {
        Array(repeating: 0, count: features.count)
    }
    func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        features.indices.map { $0 == 0 ? [0.5, 0.5] : [1.0] }
    }
}

private actor PreparedOnlyClassifier: ClassifierEstimator {
    private var fittedTargets: [Double]?
    private var fittedRows: [[Double]]?

    func fit(features: [[Double]], targets: [Double]) async throws {
        throw PreparedClassificationFailure.legacyPath
    }

    func predict(features: [[Double]]) async throws -> [Int] {
        throw PreparedClassificationFailure.legacyPath
    }

    func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        throw PreparedClassificationFailure.legacyPath
    }

    func fit(features: PreparedNumericBatch, targets: [Double]) async throws {
        fittedRows = features.rowValues()
        fittedTargets = targets
    }

    func predict(features: PreparedNumericBatch) async throws -> [Int] {
        features.rowValues().map { $0[0] > 1.5 ? 1 : 0 }
    }

    func predictProbability(features: PreparedNumericBatch) async throws -> [[Double]] {
        features.rowValues().map { row in
            let positive = row[0] / 4.0
            return [1.0 - positive, positive]
        }
    }

    func recordedFit() -> (rows: [[Double]]?, targets: [Double]?) {
        (fittedRows, fittedTargets)
    }
}

private actor LegacyClassifier: ClassifierEstimator {
    private var fitted = false

    func fit(features: [[Double]], targets: [Double]) async throws {
        guard features.count == targets.count else {
            throw PreparedClassificationFailure.unexpectedInput
        }
        fitted = true
    }

    func predict(features: [[Double]]) async throws -> [Int] {
        guard fitted else { throw SwiftMLError.modelNotFitted }
        return features.map { $0[0] >= 0 ? 1 : 0 }
    }
}

private func preparedClassificationBatch() throws -> PreparedNumericBatch {
    let x: [Double] = [1.0, 2.0, 3.0, 4.0]
    let y: [Double] = [0.0, 0.0, 1.0, 1.0]
    return try PreparedNumericBatch(columnNames: ["x", "target"], columns: [x, y])
}

@Suite("Prepared classification integration", .serialized)
struct PreparedClassificationIntegrationTests {
    @Test("Erased prepared dispatch preserves supervised selection and snapshots")
    func nativeDispatchSelectionAndSnapshot() async throws {
        var source = try preparedClassificationBatch()
        let selected = try PreparedSupervisedBatch(source, targetColumn: "target")
            .selectingRows([2, 0, 2])
        try source.updateColumn(at: 0, rows: [2], values: [99.0])

        let estimator = PreparedOnlyClassifier()
        let erased: any ClassifierEstimator = ClassificationPipeline(
            transformers: [ConsumingPreparedIdentity()],
            estimator: estimator
        )

        try await erased.fit(selected)
        let labels = try await erased.predict(features: selected.features)
        let probabilities = try await erased.predictProbability(features: selected.features)
        let recorded = await estimator.recordedFit()

        #expect(selected.features.originalRowIndices == [2, 0, 2])
        #expect(selected.features.rowValues() == [[3.0], [1.0], [3.0]])
        #expect(recorded.rows == [[3.0], [1.0], [3.0]])
        #expect(recorded.targets == [1.0, 0.0, 1.0])
        #expect(labels == [1, 0, 1])
        #expect(probabilities == [[0.25, 0.75], [0.75, 0.25], [0.25, 0.75]])
    }

    @Test("Consuming prepared input reaches consuming transformer methods")
    func consumingPipelinePath() async throws {
        let features = try PreparedNumericBatch(
            columnNames: ["x"],
            columns: [[1.0, 2.0, 3.0]]
        )
        let pipeline = ClassificationPipeline(
            transformers: [ConsumingPreparedIdentity()],
            estimator: PreparedOnlyClassifier()
        )

        try await pipeline.fit(consuming: consume features, targets: [0.0, 1.0, 1.0])
        let predictionInput = try PreparedNumericBatch(
            columnNames: ["x"],
            columns: [[1.0, 2.0, 3.0]]
        )
        let labels = try await pipeline.predict(consuming: consume predictionInput)
        #expect(labels == [0, 1, 1])

        let probabilityInput = try PreparedNumericBatch(
            columnNames: ["x"],
            columns: [[1.0, 2.0, 3.0]]
        )
        let probabilities = try await pipeline.predictProbability(consuming: consume probabilityInput)
        #expect(probabilities.count == 3)
    }

    @Test("Legacy classifiers use compatibility defaults and keep unsupported probabilities")
    func legacyCompatibility() async throws {
        let features = try PreparedNumericBatch(
            columnNames: ["x"],
            columns: [[-2.0, 0.0, 3.0]]
        )
        let classifier: any ClassifierEstimator = LegacyClassifier()

        try await classifier.fit(features: features, targets: [0.0, 1.0, 1.0])
        #expect(try await classifier.predict(features: features) == [0, 1, 1])
        await #expect(throws: SwiftMLError.self) {
            _ = try await classifier.predictProbability(features: features)
        }
    }

    @Test("Prepared StandardScaler fits training rows and matches row workflow exactly")
    func trainOnlyScalingMatchesRows() async throws {
        let trainingRows: [[Double]] = [
            [1.0, 10.0], [2.0, 13.0], [4.0, 12.0], [8.0, 20.0], [9.0, 18.0]
        ]
        let targets: [Double] = [0.0, 0.0, 0.25, 0.75, 1.0]
        let heldOutRows: [[Double]] = [[100.0, -50.0], [3.0, 15.0]]
        let training = try PreparedNumericBatch(
            columnNames: ["a", "b"],
            columns: [trainingRows.map { $0[0] }, trainingRows.map { $0[1] }]
        )
        let heldOut = try PreparedNumericBatch(
            columnNames: ["a", "b"],
            columns: [heldOutRows.map { $0[0] }, heldOutRows.map { $0[1] }]
        )

        let preparedModel = LogisticRegression(device: .cpu)
        let rowModel = LogisticRegression(device: .cpu)
        let preparedPipeline = ClassificationPipeline(
            transformers: [StandardScaler()], estimator: preparedModel
        )
        let rowPipeline = ClassificationPipeline(
            transformers: [StandardScaler()], estimator: rowModel
        )

        try await preparedPipeline.fit(features: training, targets: targets)
        try await rowPipeline.fit(features: trainingRows, targets: targets)

        let preparedLabels = try await preparedPipeline.predict(features: heldOut)
        let rowLabels = try await rowPipeline.predict(features: heldOutRows)
        let preparedProbabilities = try await preparedPipeline.predictProbability(features: heldOut)
        let rowProbabilities = try await rowPipeline.predictProbability(features: heldOutRows)
        #expect(preparedLabels == rowLabels)
        #expect(preparedProbabilities == rowProbabilities)
        let preparedParameters = await preparedModel.getWeightsAndBias()
        let rowParameters = await rowModel.getWeightsAndBias()
        #expect(preparedParameters.weights == rowParameters.weights)
        #expect(preparedParameters.bias == rowParameters.bias)
    }

    @Test("Model and pipeline reject reordered prepared schemas")
    func preparedSchema() async throws {
        let rows: [[Double]] = [[-2.0, 1.0], [-1.0, 2.0], [1.0, 3.0], [2.0, 4.0]]
        let targets: [Double] = [0.0, 0.0, 1.0, 1.0]
        let batch = try PreparedNumericBatch(
            columnNames: ["a", "b"],
            columns: [rows.map { $0[0] }, rows.map { $0[1] }]
        )
        let reversed = try PreparedNumericBatch(
            columnNames: ["b", "a"],
            columns: [rows.map { $0[1] }, rows.map { $0[0] }]
        )

        let model = LogisticRegression(device: .cpu)
        try await model.fit(features: batch, targets: targets, learningRate: 0.2, epochs: 4)
        let erasedModel: any ClassifierEstimator = model
        await #expect(throws: SwiftMLError.self) {
            _ = try await erasedModel.predict(features: reversed)
        }

        let pipeline = ClassificationPipeline(estimator: LogisticRegression(device: .cpu))
        try await pipeline.fit(features: batch, targets: targets)
        await #expect(throws: SwiftMLError.self) {
            _ = try await pipeline.predictProbability(features: reversed)
        }
    }

    @Test("Prevalidation preserves a fit and row reordering invalidates every prediction route")
    func failedRefitRules() async throws {
        let rows: [[Double]] = [[-2.0], [-1.0], [1.0], [2.0]]
        let targets: [Double] = [0.0, 0.0, 1.0, 1.0]
        let batch = try PreparedNumericBatch(
            columnNames: ["x"],
            columns: [rows.map { $0[0] }]
        )
        let pipeline = ClassificationPipeline(
            transformers: [StandardScaler()],
            estimator: LogisticRegression(device: .cpu)
        )
        try await pipeline.fit(features: batch, targets: targets)
        let original = try await pipeline.predict(features: batch)

        await #expect(throws: SwiftMLError.self) {
            try await pipeline.fit(features: batch, targets: [])
        }
        #expect(try await pipeline.predict(features: batch) == original)

        pipeline.transformers = [ReorderingPreparedTransformer()]
        await #expect(throws: SwiftMLError.self) {
            try await pipeline.fit(features: batch, targets: targets)
        }
        await #expect(throws: SwiftMLError.modelNotFitted) {
            _ = try await pipeline.predict(features: batch)
        }
        await #expect(throws: SwiftMLError.modelNotFitted) {
            _ = try await pipeline.predictProbability(features: batch)
        }
        await #expect(throws: SwiftMLError.modelNotFitted) {
            _ = try await pipeline.predict(features: rows)
        }
        await #expect(throws: SwiftMLError.modelNotFitted) {
            _ = try await pipeline.predictProbability(features: rows)
        }
    }

    @Test("Prepared transformers cannot change row count or identity")
    func transformerRowContracts() async throws {
        let batch = try PreparedNumericBatch(
            columnNames: ["x"],
            columns: [[1.0, 2.0, 3.0]]
        )
        let targets: [Double] = [0.0, 1.0, 1.0]
        let invalidTransformers: [any PreprocessingTransformer] = [
            DroppingPreparedTransformer(),
            ReorderingPreparedTransformer(),
        ]

        for transformer in invalidTransformers {
            let pipeline = ClassificationPipeline(
                transformers: [transformer],
                estimator: PreparedOnlyClassifier()
            )
            await #expect(throws: SwiftMLError.self) {
                try await pipeline.fit(features: batch, targets: targets)
            }
        }
    }

    @Test("Pipeline rejects inconsistent probability widths")
    func probabilityShape() async throws {
        let batch = try PreparedNumericBatch(
            columnNames: ["x"],
            columns: [[1.0, 2.0, 3.0]]
        )
        let pipeline = ClassificationPipeline(estimator: InconsistentProbabilityClassifier())
        try await pipeline.fit(features: batch, targets: [0.0, 1.0, 1.0])
        await #expect(throws: SwiftMLError.self) {
            _ = try await pipeline.predictProbability(features: batch)
        }
    }

    @Test("Prepared classifier budget reservations release on success and failure")
    func budgetRelease() async throws {
        let features = try PreparedNumericBatch(
            columnNames: ["x"],
            columns: [[1.0, 2.0, 3.0]]
        )
        let classifier: any ClassifierEstimator = ClassificationPipeline(estimator: PreparedOnlyClassifier())
        let budget = try MemoryBudget(limit: 128)
        let estimate = try MemoryEstimate(capacities: [64])

        await #expect(throws: SwiftMLError.self) {
            try await classifier.fit(
                features: features,
                targets: [],
                budget: budget,
                estimate: estimate
            )
        }
        #expect(await budget.reservedBytes == 0)

        try await classifier.fit(
            features: features,
            targets: [0.0, 1.0, 1.0],
            budget: budget,
            estimate: estimate
        )
        #expect(try await classifier.predict(
            features: features,
            budget: budget,
            estimate: estimate
        ) == [0, 1, 1])
        #expect(try await classifier.predictProbability(
            features: features,
            budget: budget,
            estimate: estimate
        ).count == 3)
        #expect(await budget.reservedBytes == 0)
    }
}
