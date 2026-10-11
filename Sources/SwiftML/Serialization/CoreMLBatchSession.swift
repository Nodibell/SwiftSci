import SwiftDataFrame
import SwiftPreprocessing

/// One scoped model serving sequential batch runs with independent workspace admission.
/// Create this owner with `CoreMLBatchPipeline.withSession`.
public actor CoreMLBatchSession {
    private var pool: CoreMLMatrixPool?
    private var preprocessing: StandardPreprocessingPlan?
    private let configuration: CoreMLBatchPipeline.Configuration
    private let workBudget: MemoryBudget
    private var running = false
    private var closed = false
    private var closeWaiters: [CheckedContinuation<Void, Never>] = []

    init(pool: CoreMLMatrixPool, preprocessing: StandardPreprocessingPlan,
         configuration: CoreMLBatchPipeline.Configuration, workBudget: MemoryBudget) {
        self.pool = pool
        self.preprocessing = preprocessing
        self.configuration = configuration
        self.workBudget = workBudget
    }

    /// Admits one complete batch window until this source ends and consumption finishes.
    /// Bound the source to yield capacity between runs. Sequence numbers restart at zero.
    /// Loading and consumption follow `CoreMLBatchPipeline.run`'s ownership contract.
    /// Overlapping runs throw before admission; use separate sessions for independent callers.
    /// A failed run drains its work and leaves the session available for another run.
    public func run(load: @Sendable (CoreMLBatchPipeline.Request) async throws -> PreparedNumericBatch?,
                    consume: @Sendable (Int, CoreMLPrediction) async throws -> Void) async throws {
        guard !closed, !running, let pool, let preprocessing else {
            throw SwiftMLError.invalidParameter("Core ML batch session is closed or already running")
        }
        running = true
        do {
            try await workBudget.withReservation(MemoryEstimate(capacities: [configuration.windowBytes])) {
                try Task.checkCancellation()
                try await CoreMLBatchPipeline.runPrepared(pool: pool, preprocessing: preprocessing,
                    configuration: self.configuration, load: load, consume: consume)
            }
            finishRun()
        } catch {
            finishRun()
            throw error
        }
    }

    // Scope exit drains the entire run, including callbacks, before the pool can close.
    func close() async {
        closed = true
        if running { await withCheckedContinuation { closeWaiters.append($0) } }
        pool = nil
        preprocessing = nil
    }

    private func finishRun() {
        running = false
        let waiters = closeWaiters
        closeWaiters = []
        waiters.forEach { $0.resume() }
    }
}
