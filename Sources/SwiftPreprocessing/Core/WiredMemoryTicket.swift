import Foundation
import os
#if canImport(MLX)
import MLX
#endif

/// A legacy concurrency slot, without a byte reservation.
///
/// ## Concurrency & Resource Cleanup
/// `WiredMemoryTicket` controls concurrency limits for high-memory operations on Apple Silicon.
/// Always prefer structured execution via `WiredMemoryManager.withTicket(_:)` or explicitly call
/// ``finish()`` after submitted work completes. Cache clearing does not synchronize GPU work.
///
/// ## Thread Safety
/// Thread-safe and conforms to `Sendable`. State transitions are protected by an internal lock.
@available(*, deprecated, message: "Legacy concurrency slot. Use MemoryReservation for byte admission or MLX.WiredMemoryTicket for MLX wired limits; their policies differ.")
public final class WiredMemoryTicket: Sendable {
    private let manager: WiredMemoryManager
    private let releasedState = OSAllocatedUnfairLock(initialState: false)
    
    internal init(manager: WiredMemoryManager) {
        self.manager = manager
    }
    
    /// Clears the MLX cache and releases the slot once. Call only after submitted work completes.
    public func finish() async {
        let alreadyReleased = releasedState.withLock { state in
            if !state {
                state = true
                return false
            }
            return true
        }
        
        if !alreadyReleased {
            #if canImport(MLX)
            MLX.Memory.clearCache()
            #endif
            await manager.releaseTicket()
        }
    }
    
    deinit {
        let manager = self.manager
        let stateLock = self.releasedState
        
        let needsRelease = stateLock.withLock { state in
            if !state {
                state = true
                return true
            }
            return false
        }
        
        if needsRelease {
            // Deinitialization releases the slot without changing the process-wide cache.
            Task {
                await manager.releaseTicket()
            }
        }
    }
}

