import Foundation
import os
import MLX

/// A ticket representing a reservation of memory and concurrency slot.
/// When finished or deinitialized, it flushes the GPU memory cache and releases the slot.
public final class WiredMemoryTicket: Sendable {
    private let manager: WiredMemoryManager
    private let releasedState = OSAllocatedUnfairLock(initialState: false)
    
    internal init(manager: WiredMemoryManager) {
        self.manager = manager
    }
    
    /// Concludes the operation, flushes MLX memory cache, and releases the slot.
    public func finish() async {
        let alreadyReleased = releasedState.withLock { state in
            if !state {
                state = true
                return false
            }
            return true
        }
        
        if !alreadyReleased {
            MLX.Memory.clearCache()
            await manager.releaseTicket()
        }
    }
    
    deinit {
        let manager = self.manager
        let stateLock = self.releasedState
        
        Task {
            let alreadyReleased = stateLock.withLock { state in
                if !state {
                    state = true
                    return false
                }
                return true
            }
            if !alreadyReleased {
                MLX.Memory.clearCache()
                await manager.releaseTicket()
            }
        }
    }
}
