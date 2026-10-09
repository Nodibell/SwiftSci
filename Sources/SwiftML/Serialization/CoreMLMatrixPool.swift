import CoreML
import Foundation
import SwiftDataFrame
import SwiftPreprocessing

/// A scoped, persistent pool for fixed floating-point matrix inference.
/// Float16 input and output require macOS 15 or later.
/// Call `withPool` to reserve capacity before loading a model. Only owned results escape prediction.
public actor CoreMLMatrixPool {
    /// Conservative lifetime and per-request allowances. These are estimates, not an RSS limit.
    public struct Configuration: Sendable {
        /// Maximum simultaneous native predictions. Queued inputs remain caller-owned.
        public let maximumConcurrentPredictions: Int
        /// Total allowance for model loading, retained model state, buffers, padding, and pool overhead.
        public let retainedBytes: Int
        /// Per-slot allowance for admitted input, owned output, fallback output, and framework workspace.
        public let requestBytes: Int

        /// Requires positive capacities and concurrency. Overflow is rejected before admission.
        public init(maximumConcurrentPredictions: Int = 1, retainedBytes: Int, requestBytes: Int) throws {
            guard maximumConcurrentPredictions > 0, retainedBytes > 0, requestBytes > 0 else {
                throw MemoryAdmissionError.invalidCapacity
            }
            self.maximumConcurrentPredictions = maximumConcurrentPredictions
            self.retainedBytes = retainedBytes
            self.requestBytes = requestBytes
            _ = try quota
        }

        var quota: MemoryEstimate {
            get throws {
                try MemoryEstimate(capacities: [retainedBytes,
                    coreMLByteCount(maximumConcurrentPredictions, requestBytes)])
            }
        }
    }

    // Counts successful returned predictions. Identity matches prove reuse of the supplied
    // output object, not absence of internal framework copies or adoption of other wrappers.
    package private(set) var completedPredictions = 0
    package private(set) var outputBackingIdentityMatches = 0

    private var session: CoreMLPredictionSession?
    private var idle: [CoreMLMatrixBuffers]
    private let slots: MemoryBudget
    private let requestBytes: Int
    private(set) var closed = false
    private(set) var pending = 0
    private var closeWaiters: [CheckedContinuation<Void, Never>] = []

    private init(session: sending CoreMLPredictionSession, buffers: [CoreMLMatrixBuffers],
                 configuration: Configuration) throws {
        self.session = session
        idle = buffers
        slots = try MemoryBudget(limit: configuration.maximumConcurrentPredictions)
        requestBytes = configuration.requestBytes
    }

    /// Reserves retainedBytes + maximumConcurrentPredictions * requestBytes for the entire scope.
    ///
    /// Model loading occurs after admission. The retained estimate must include model and framework
    /// allocations in addition to the checked buffer capacities. No allocator-enforced cap is provided.
    /// Predictions reuse private buffers, copy outputs, and drain before scope exit releases the quota.
    /// Returned results and queued caller inputs need separate lifetime accounting.
    ///
    /// Use structured child tasks and await them inside `operation`. Acquire competing pools as
    /// siblings; nesting a second acquisition under a held quota can cause caller hold-and-wait.
    /// Escaped pool references reject predictions after the scope closes. An error or cancellation
    /// drains submitted work before releasing its capacity; Core ML controls device interruption.
    public static func withPool<Result: Sendable>(compiledModelURL: URL, inputColumns: [String],
        outputName: String, computeUnits: MLComputeUnits = .all, budget: MemoryBudget,
        configuration: Configuration,
        operation: @Sendable (CoreMLMatrixPool) async throws -> Result) async throws -> Result {
        try await budget.withReservation(configuration.quota) {
            try Task.checkCancellation()
            let session = try CoreMLPredictionSession(compiledModelURL: compiledModelURL,
                inputColumns: inputColumns, outputName: outputName, computeUnits: computeUnits, inputLayout: .matrix)
            let capacity = try session.matrixBufferCapacity(count: configuration.maximumConcurrentPredictions)
            guard capacity <= configuration.retainedBytes else {
                throw MemoryAdmissionError.exceedsLimit(required: capacity, limit: configuration.retainedBytes)
            }
            let buffers = try (0..<configuration.maximumConcurrentPredictions).map { _ in
                try CoreMLMatrixBuffers(input: session.inputArray!, output: session.outputArray!)
            }
            let pool = try CoreMLMatrixPool(session: session, buffers: buffers, configuration: configuration)
            do {
                try Task.checkCancellation()
                let result = try await operation(pool)
                await pool.close()
                return result
            } catch {
                await pool.close()
                throw error
            }
        }
    }

    /// Predicts with one exclusive slot. Inputs follow the configured column names in any order.
    /// Invalid shape or an input/output estimate above requestBytes is rejected before queuing.
    /// The remaining request allowance must cover model-specific temporary allocations and overhead.
    public func predict(_ input: PreparedNumericBatch) async throws -> CoreMLPrediction {
        guard !closed, session != nil else { throw SwiftMLError.invalidParameter("Core ML matrix pool is closed") }
        try session!.validateRequest(input, maximumBatchSize: 1)
        let indices = try session!.columnIndices(in: input)
        return try await predict(.columns(input, indices: indices))
    }

    /// Converts input once using this pool's model type and column order.
    /// The input has a separate lifetime reservation and may outlive this pool.
    /// Budget headroom must cover both the pool quota and prepared input; acquisition can wait.
    /// Keep the source batch accounted for until preparation completes. Results need separate accounting.
    public func prepare(_ input: PreparedNumericBatch, budget: MemoryBudget) async throws -> CoreMLPreparedMatrix {
        guard !closed, session != nil else { throw SwiftMLError.invalidParameter("Core ML matrix pool is closed") }
        try session!.validateRequest(input, maximumBatchSize: 1)
        let indices = try session!.columnIndices(in: input)
        let allowance = try CoreMLPreparedMatrix.allowance(schema: session!.inputArray!, names: session!.inputColumns)
        let reservation = try await budget.acquire(allowance)
        do {
            try Task.checkCancellation()
            guard !closed, let session else { throw SwiftMLError.invalidParameter("Core ML matrix pool is closed") }
            return try CoreMLPreparedMatrix(input: input, indices: indices, session: session,
                reservation: reservation, allowance: allowance)
        } catch {
            await reservation.finish()
            throw error
        }
    }

    // Experimental package-only boundary for the fused preprocessing benchmark.
    // The source provides row identity; packed values must already follow model column order.
    package func preparePackedForTrial(_ values: [Float16], source: PreparedNumericBatch,
                                      budget: MemoryBudget) async throws -> CoreMLPreparedMatrix {
        guard !closed, session != nil else {
            throw SwiftMLError.invalidParameter("Core ML matrix pool is closed")
        }
        try session!.validateRequest(source, maximumBatchSize: 1)
        let schema = session!.inputArray!
        guard schema.dataType == .float16, source.columnNames == session!.inputColumns,
              values.count == (try coreMLByteCount(schema.shape[0], schema.width)),
              values.allSatisfy(\.isFinite) else {
            throw SwiftMLError.invalidParameter("Packed trial input requires finite Float16 values in the exact model shape and column order")
        }
        let allowance = try CoreMLPreparedMatrix.allowance(schema: schema, names: session!.inputColumns)
        let reservation = try await budget.acquire(allowance)
        do {
            try Task.checkCancellation()
            guard !closed, let session else {
                throw SwiftMLError.invalidParameter("Core ML matrix pool is closed")
            }
            return try CoreMLPreparedMatrix(trialPacked: values, input: source, session: session,
                reservation: reservation, allowance: allowance)
        } catch {
            await reservation.finish()
            throw error
        }
    }

    /// Reuses a converted input matching this pool's shape, element type, and column order.
    /// Each request copies its bytes into an exclusive slot without repeating numeric conversion.
    public func predict(_ input: CoreMLPreparedMatrix) async throws -> CoreMLPrediction {
        guard !closed, let session else { throw SwiftMLError.invalidParameter("Core ML matrix pool is closed") }
        try input.validate(for: session)
        return try await predict(.prepared(input))
    }

    private func predict(_ input: CoreMLMatrixRequest) async throws -> CoreMLPrediction {
        let required = try session!.matrixRequestBytes(input)
        guard required <= requestBytes else {
            throw MemoryAdmissionError.exceedsLimit(required: required, limit: requestBytes)
        }
        return try await withSlot { pool, buffer in
            let output = try await pool.execute(input, buffer: buffer)
            pool.completedPredictions += 1
            if output.backingMatched { pool.outputBackingIdentityMatches += 1 }
            return output.prediction
        }
    }

    // A slot remains exclusively owned until its operation completes, including cancellation.
    func withSlot<Result: Sendable>(operation: @Sendable (isolated CoreMLMatrixPool, CoreMLMatrixBuffers)
        async throws -> Result) async throws -> Result {
        guard !closed else { throw SwiftMLError.invalidParameter("Core ML matrix pool is closed") }
        pending += 1
        var token: MemoryReservation?
        var buffer: CoreMLMatrixBuffers?
        do {
            token = try await slots.acquire(MemoryEstimate(capacities: [1]))
            guard !closed, let available = idle.popLast() else {
                throw SwiftMLError.invalidParameter("Core ML matrix pool is closed")
            }
            buffer = available
            let result = try await operation(self, available)
            await finish(buffer: buffer, token: token)
            return result
        } catch {
            await finish(buffer: buffer, token: token)
            throw error
        }
    }

    private func execute(_ input: CoreMLMatrixRequest,
                         buffer: CoreMLMatrixBuffers) async throws -> (prediction: CoreMLPrediction, backingMatched: Bool) {
        guard let session else { throw SwiftMLError.invalidParameter("Core ML matrix pool is closed") }
        return try await session.runPooledMatrix(input, buffers: buffer, isolation: self)
    }

    private func finish(buffer: CoreMLMatrixBuffers?, token: MemoryReservation?) async {
        if let buffer {
            if closed { buffer.release() } else { idle.append(buffer) }
        }
        await token?.finish()
        pending -= 1
        if closed && pending == 0 { completeClose() }
    }

    private func close() async {
        closed = true
        idle.forEach { $0.release() }
        idle = []
        if pending == 0 { completeClose(); return }
        await withCheckedContinuation { closeWaiters.append($0) }
    }

    private func completeClose() {
        session = nil
        let waiters = closeWaiters
        closeWaiters = []
        waiters.forEach { $0.resume() }
    }
}
