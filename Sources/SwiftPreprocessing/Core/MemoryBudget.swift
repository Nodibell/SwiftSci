import Foundation

/// Invalid estimates and requests that cannot fit the configured reservation budget.
public enum MemoryAdmissionError: Error, Equatable {
    /// A capacity is negative, overflows Int, or the budget limit is not positive.
    case invalidCapacity
    /// The request exceeds the entire budget and cannot become eligible by waiting.
    case exceedsLimit(required: Int, limit: Int)
}

/// A whole-operation peak estimate. Include retained inputs, copies, workspace and outputs.
/// This describes participating work; it is not a measurement of total process memory.
public struct MemoryEstimate: Sendable {
    /// Sum of the supplied capacities, in bytes.
    public let bytes: Int
    /// Adds nonnegative capacities, rejecting overflow before admission.
    public init(capacities: [Int]) throws {
        var total = 0
        for capacity in capacities {
            guard capacity >= 0 else { throw MemoryAdmissionError.invalidCapacity }
            let (next, overflow) = total.addingReportingOverflow(capacity)
            guard !overflow else { throw MemoryAdmissionError.invalidCapacity }
            total = next
        }
        bytes = total
    }
}

/// Admits whole peak estimates in FIFO order under a shared byte limit.
/// Reservations do not intercept allocators or cap unrelated memory or persistent model state.
public actor MemoryBudget {
    /// Maximum total bytes reserved by active operations.
    public let limit: Int
    private var active: [UUID: Int] = [:]
    private struct Waiter {
        let id: UUID
        let bytes: Int
        let continuation: CheckedContinuation<Void, any Error>
    }
    private var waiters: [Waiter] = []
    private var reserved = 0
    /// Largest observed sum of active reservations.
    public private(set) var peak = 0
    /// Current sum of active reservations.
    public var reservedBytes: Int { reserved }
    /// Number of requests waiting for a complete reservation.
    public var queuedCount: Int { waiters.count }

    /// Creates a budget with a strictly positive byte limit.
    public init(limit: Int) throws {
        guard limit > 0 else { throw MemoryAdmissionError.invalidCapacity }
        self.limit = limit
    }

    /// Reserves the complete estimate or waits without partially reserving bytes.
    /// Cancellation removes queued requests or returns a grant before throwing.
    public func acquire(_ plan: MemoryEstimate) async throws -> MemoryReservation {
        try Task.checkCancellation()
        guard plan.bytes <= limit else { throw MemoryAdmissionError.exceedsLimit(required: plan.bytes, limit: limit) }
        let id = UUID()
        if waiters.isEmpty && plan.bytes <= limit - reserved {
            grant(id, bytes: plan.bytes)
        } else {
            try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { continuation in
                    waiters.append(Waiter(id: id, bytes: plan.bytes, continuation: continuation))
                }
            } onCancel: {
                Task { await self.cancel(id) }
            }
        }
        // Cancellation may race with a successful grant. Return its bytes before throwing.
        do { try Task.checkCancellation() } catch { release(id); throw error }
        return MemoryReservation(budget: self, id: id)
    }

    private func grant(_ id: UUID, bytes: Int) {
        active[id] = bytes
        reserved += bytes
        peak = max(peak, reserved)
    }
    private func drain() {
        while let first = waiters.first, first.bytes <= limit - reserved {
            waiters.removeFirst()
            grant(first.id, bytes: first.bytes)
            first.continuation.resume()
        }
    }
    private func cancel(_ id: UUID) {
        if let index = waiters.firstIndex(where: { $0.id == id }) {
            waiters.remove(at: index).continuation.resume(throwing: CancellationError())
            drain()
        }
    }
    fileprivate func release(_ id: UUID) {
        guard let bytes = active.removeValue(forKey: id) else { return }
        reserved -= bytes
        drain()
    }

    /// Runs an operation and releases its reservation on success or error.
    /// Complete submitted device work before leaving this scope. Returned persistent allocations
    /// are no longer reserved; include them in estimates for subsequent participating work.
    public func withReservation<R: Sendable>(_ plan: MemoryEstimate,
        operation: @Sendable () async throws -> R) async throws -> R {
        let reservation = try await acquire(plan)
        do {
            let result = try await operation()
            await reservation.finish()
            return result
        } catch {
            await reservation.finish()
            throw error
        }
    }
}

/// A reservation released explicitly or when its final reference is destroyed.
/// Prefer scoped execution so release follows operation and device completion.
public final class MemoryReservation: Sendable {
    private let budget: MemoryBudget
    private let id: UUID
    fileprivate init(budget: MemoryBudget, id: UUID) { self.budget = budget; self.id = id }
    /// Releases this reservation once. Repeated calls are harmless.
    public func finish() async { await budget.release(id) }
    deinit {
        let budget = budget, id = id
        Task { await budget.release(id) }
    }
}
