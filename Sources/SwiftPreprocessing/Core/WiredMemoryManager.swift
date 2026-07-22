import Foundation

/// Actor-based manager that controls concurrent allocations and tasks in Apple Silicon Unified Memory.
public actor WiredMemoryManager {
    /// Global shared memory manager with a default limit scaling with active processor count.
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
    
    /// Handles task cancellation by removing the continuation from the suspension queue and throwing CancellationError.
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
