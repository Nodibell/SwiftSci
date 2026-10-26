import Testing
import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftML

private actor ClassificationOperationGate {
    private var entries = 0
    private var open = false
    private var blocked: [CheckedContinuation<Void, Never>] = []
    private var observers: [(count: Int, continuation: CheckedContinuation<Void, Never>)] = []

    func enter() async {
        entries += 1
        let ready = observers.filter { $0.count <= entries }
        observers.removeAll { $0.count <= entries }
        for observer in ready { observer.continuation.resume() }
        guard !open else { return }
        await withCheckedContinuation { blocked.append($0) }
    }

    func waitForEntries(_ count: Int) async {
        guard entries < count else { return }
        await withCheckedContinuation { observers.append((count, $0)) }
    }

    func release() {
        open = true
        let continuations = blocked
        blocked.removeAll()
        for continuation in continuations { continuation.resume() }
    }
}

private actor GatedPipelineClassifier: ClassifierEstimator {
    let fitGate: ClassificationOperationGate?
    let predictionGate: ClassificationOperationGate?

    init(fitGate: ClassificationOperationGate? = nil, predictionGate: ClassificationOperationGate? = nil) {
        self.fitGate = fitGate
        self.predictionGate = predictionGate
    }

    func fit(features: [[Double]], targets: [Double]) async throws {
        guard features.allSatisfy({ $0.allSatisfy(\.isFinite) }) else { throw SwiftMLError.containsNaN }
        if let fitGate { await fitGate.enter() }
    }

    func predict(features: [[Double]]) async throws -> [Int] {
        if let predictionGate { await predictionGate.enter() }
        return features.map { $0[0] > 0.5 ? 1 : 0 }
    }

    func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        if let predictionGate { await predictionGate.enter() }
        return features.map { $0[0] > 0.5 ? [0.25, 0.75] : [0.75, 0.25] }
    }
}

private struct PipelineOffset: PreprocessingTransformer {
    let value: Double
    mutating func fit(_ data: [[Double]]) throws {}
    func transform(_ data: [[Double]]) throws -> [[Double]] {
        data.map { row in row.map { $0 + value } }
    }
}

@Suite("Classification pipeline synchronized lifecycle")
struct ClassificationPipelineConcurrencyTests {
    @Test("An active fit excludes competing fits and both prediction routes", arguments: [false, true])
    func fitExcludesCompetingOperations(prepared: Bool) async throws {
        let gate = ClassificationOperationGate()
        let pipeline = ClassificationPipeline(estimator: GatedPipelineClassifier(fitGate: gate))
        let batch = try PreparedNumericBatch(columnNames: ["x"], columns: [[0, 1]])
        let fitting = Task {
            if prepared {
                try await pipeline.fit(features: batch, targets: [0, 1])
            } else {
                try await pipeline.fit(features: [[0], [1]], targets: [0, 1])
            }
        }
        await gate.waitForEntries(1)
        await #expect(throws: SwiftMLError.self) {
            try await pipeline.fit(features: batch, targets: [0, 1])
        }
        await #expect(throws: SwiftMLError.self) {
            try await pipeline.fit(features: [[0], [1]], targets: [0, 1])
        }
        await #expect(throws: SwiftMLError.modelNotFitted) {
            _ = try await pipeline.predict(features: batch)
        }
        await #expect(throws: SwiftMLError.modelNotFitted) {
            _ = try await pipeline.predictProbability(features: [[0], [1]])
        }
        await gate.release()
        try await fitting.value
        #expect(try await pipeline.predict(features: batch) == [0, 1])
    }

    @Test("Configuration changes reject a stale fit commit and require recovery", arguments: [false, true])
    func configurationInvalidatesSuspendedFit(prepared: Bool) async throws {
        let gate = ClassificationOperationGate()
        let pipeline = ClassificationPipeline(estimator: GatedPipelineClassifier(fitGate: gate))
        let batch = try PreparedNumericBatch(columnNames: ["x"], columns: [[0, 1]])
        let fitting = Task {
            if prepared {
                try await pipeline.fit(features: batch, targets: [0, 1])
            } else {
                try await pipeline.fit(features: [[0], [1]], targets: [0, 1])
            }
        }
        await gate.waitForEntries(1)
        pipeline.transformers = [PipelineOffset(value: 2)]
        #expect(pipeline.transformers.count == 1)
        await gate.release()
        await #expect(throws: SwiftMLError.self) { try await fitting.value }
        await #expect(throws: SwiftMLError.modelNotFitted) {
            _ = try await pipeline.predict(features: batch)
        }
        await #expect(throws: SwiftMLError.modelNotFitted) {
            _ = try await pipeline.predictProbability(features: [[0], [1]])
        }
        try await pipeline.fit(features: batch, targets: [0, 1])
        #expect(try await pipeline.predict(features: batch) == [1, 1])
        #expect(pipeline.transformers.count == 1)
    }

    @Test("Concurrent predictions retain snapshots and exclude a fit")
    func concurrentPredictionsRetainConfiguration() async throws {
        let gate = ClassificationOperationGate()
        let pipeline = ClassificationPipeline(estimator: GatedPipelineClassifier(predictionGate: gate))
        let batch = try PreparedNumericBatch(columnNames: ["x"], columns: [[0, 1]])
        try await pipeline.fit(features: batch, targets: [0, 1])
        let labels = Task { try await pipeline.predict(features: batch) }
        let probabilities = Task { try await pipeline.predictProbability(features: [[0], [1]]) }
        await gate.waitForEntries(2)
        await #expect(throws: SwiftMLError.self) {
            try await pipeline.fit(features: batch, targets: [0, 1])
        }
        pipeline.transformers = [PipelineOffset(value: 2)]
        await gate.release()
        #expect(try await labels.value == [0, 1])
        #expect(try await probabilities.value == [[0.75, 0.25], [0.25, 0.75]])
        await #expect(throws: SwiftMLError.modelNotFitted) {
            _ = try await pipeline.predict(features: batch)
        }
        try await pipeline.fit(features: batch, targets: [0, 1])
        #expect(try await pipeline.predict(features: batch) == [1, 1])
    }

    @Test("Configuration before the first fit preserves supplied-model delegation")
    func prefitConfigurationAndGetterSnapshots() async throws {
        let pipeline = ClassificationPipeline(estimator: GatedPipelineClassifier())
        pipeline.transformers = [PipelineOffset(value: 2)]
        var snapshot = pipeline.transformers
        snapshot.removeAll()
        #expect(snapshot.isEmpty)
        #expect(pipeline.transformers.count == 1)
        #expect(try await pipeline.predict(features: [[0], [1]]) == [1, 1])
    }

    @Test("Legacy row preprocessing can impute NaN before estimator validation")
    func legacyNaNPreprocessing() async throws {
        let pipeline = ClassificationPipeline(transformers: [Imputer()], estimator: GatedPipelineClassifier())
        try await pipeline.fit(features: [[Double.nan], [1]], targets: [0, 1])
        #expect(try await pipeline.predict(features: [[Double.nan], [1]]) == [1, 1])
    }
}
