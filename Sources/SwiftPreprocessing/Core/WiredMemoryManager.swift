import Foundation

/// Legacy task-count limiter retained for source compatibility.
///
/// Use `MemoryBudget` for estimated-byte admission. For MLX wired-memory limits,
/// use `MLX.WiredMemoryManager`. Neither is a drop-in task-count replacement.
@available(*, deprecated, message: "Legacy task-count limiter. Use MemoryBudget for byte admission or MLX.WiredMemoryManager for MLX wired limits; their policies differ.")
public actor WiredMemoryManager {
    /// Shared legacy limiter with a task-count limit derived from active processor count.
    public static let shared = WiredMemoryManager(maxConcurrentTasks: max(2, ProcessInfo.processInfo.activeProcessorCount))
    
    private let maxConcurrentTasks: Int
    private var activeTasksCount = 0
    
    private var nextContinuationId = 0
    private var suspensionQueue: [(id: Int, continuation: CheckedContinuation<Void, any Error>)] = []
    
    /// Initializes the manager with a concurrency limit.
    public init(maxConcurrentTasks: Int) {
        self.maxConcurrentTasks = maxConcurrentTasks
    }
    
    /// Acquires a ticket to run a memory-intensive GPU/CPU calculation.
    /// If the concurrency limit is reached, this method suspends asynchronously until a ticket is released.
    /// Supports Swift task cancellation.
    /// - Throws: `CancellationError` if acquisition is cancelled before a ticket is granted.
    public func acquireTicket() async throws -> WiredMemoryTicket {
        try Task.checkCancellation()
        
        if activeTasksCount < maxConcurrentTasks {
            activeTasksCount += 1
            return WiredMemoryTicket(manager: self)
        }
        
        let id = nextContinuationId
        nextContinuationId += 1
        
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                suspensionQueue.append((id: id, continuation: continuation))
            }
        } onCancel: {
            Task {
                await self.cancelAcquire(id: id)
            }
        }
        
        return WiredMemoryTicket(manager: self)
    }
    
    /// Scoped helper that executes an operation within an acquired memory ticket,
    /// ensuring the ticket is always cleaned up and cache is cleared.
    /// - Parameters:
    ///   - operation: Work that completes before the ticket is released.
    /// - Throws: An acquisition error or an error from the operation.
    public func withTicket<T: Sendable>(_ operation: () async throws -> T) async throws -> T {
        let ticket = try await acquireTicket()
        let result: T
        do {
            result = try await operation()
        } catch {
            await ticket.finish()
            throw error
        }
        await ticket.finish()
        return result
    }
    
    /// Scoped helper that executes an operation with direct access to an acquired memory ticket,
    /// ensuring the ticket is always cleaned up and cache is cleared upon completion.
    /// - Parameters:
    ///   - operation: Work that completes before the ticket is released.
    /// - Throws: An acquisition error or an error from the operation.
    public func withTicket<T: Sendable>(_ operation: (WiredMemoryTicket) async throws -> T) async throws -> T {
        let ticket = try await acquireTicket()
        let result: T
        do {
            result = try await operation(ticket)
        } catch {
            await ticket.finish()
            throw error
        }
        await ticket.finish()
        return result
    }
    
    /// Handles task cancellation by removing the continuation from the suspension queue and throwing CancellationError.
    /// - Parameters:
    ///   - id: Unique element identifier or key.
    public func cancelAcquire(id: Int) {
        if let idx = suspensionQueue.firstIndex(where: { $0.id == id }) {
            let item = suspensionQueue.remove(at: idx)
            item.continuation.resume(throwing: CancellationError())
        }
    }
    
    /// Releases an active ticket and resumes the next suspended task if any.
    public func releaseTicket() {
        activeTasksCount -= 1
        if !suspensionQueue.isEmpty {
            activeTasksCount += 1
            let next = suspensionQueue.removeFirst()
            next.continuation.resume()
        }
    }
    
    /// Gets the current number of active concurrent tasks.
    public func activeTasks() -> Int {
        return activeTasksCount
    }
}
