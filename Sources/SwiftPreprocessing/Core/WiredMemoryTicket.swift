import Foundation
import os
#if canImport(MLX)
import MLX
#endif

/// A ticket representing a reservation of memory and a concurrency slot.
///
/// ## Concurrency & Resource Cleanup
/// `WiredMemoryTicket` controls concurrency limits for high-memory operations on Apple Silicon.
/// Always prefer structured execution via `WiredMemoryManager.withTicket(_:)` or explicitly call
/// ``finish()`` when operations conclude to synchronize MLX GPU caches.
///
/// ## Thread Safety
/// Thread-safe and conforms to `Sendable`. State transitions are protected by an internal lock.
public final class WiredMemoryTicket: Sendable {
    private let manager: WiredMemoryManager
    private let releasedState = OSAllocatedUnfairLock(initialState: false)
    
    internal init(manager: WiredMemoryManager) {
        self.manager = manager
    }
    
    /// Concludes the operation, flushes MLX memory cache safely, and releases the concurrency slot.
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
            // Only release concurrency slot in uncoordinated deinit;
            // avoid clearing global GPU cache out-of-band to prevent active buffer corruption.
            Task {
                await manager.releaseTicket()
            }
        }
    }
}

